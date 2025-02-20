import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.utils import check_X_y, check_array
from tqdm.notebook import tqdm


class CDNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_iterations=100, burn_in=50, random_state=None):
        """
        Initialize the Conditional Dependency Network classifier.

        Parameters
        ----------
        base_estimator : scikit-learn binary classifier, default=LogisticRegression()
            The estimator used to model the conditional probability for each label.
        n_iterations : int, default=100
            Total number of Gibbs sampling iterations during inference.
        burn_in : int, default=50
            Number of initial iterations to discard for burn-in.
        random_state : int or None, default=None
            Seed for random number generation.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression()
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, Y: numpy.ndarray):
        """
        Fit the CDN classifier.

        For each label, a binary classifier is trained to predict that label using
        the original features augmented by all other labels (i.e., the label’s Markov blanket).

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training input samples.
        Y : numpy.ndarray of shape (n_samples, n_labels)
            Binary indicator matrix of labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = numpy.array(Y)
        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels
        self.n_features_ = X.shape[1]

        # Dictionary to store a classifier for each label
        self.classifiers_ = {}

        # Train a binary classifier for each label j
        for j in tqdm(range(n_labels), desc="Training for each label"):
            # Construct augmented features: [X, Y_without_label_j]
            other_labels = numpy.delete(Y, j, axis=1)
            X_aug = numpy.hstack((X, other_labels))
            clf = clone(self.base_estimator)
            if hasattr(clf, "fit"):
                clf.fit(X_aug, Y[:, j])
            else:
                raise Exception("Base estimator does not have fit function")
            self.classifiers_[j] = clf

        return self

    def _gibbs_sampling(self, X_instance: numpy.ndarray) -> numpy.ndarray:
        """
        Perform Gibbs sampling for a single instance.

        Starting from a random initialization of the label vector, the sampler iteratively
        updates each label (in random order) using the conditional probability model.
        Samples collected after the burn-in period are averaged to estimate the marginal probabilities.

        Parameters
        ----------
        X_instance : numpy.ndarray of shape (n_features,)
            The feature vector for one instance.

        Returns
        -------
        prob_estimates : numpy.ndarray of shape (n_labels,)
            Estimated probability for each label.
        """
        rng = numpy.random.RandomState(self.random_state)
        n_labels = self.n_labels_
        # Initialize label vector randomly (0 or 1)
        current_state = rng.randint(0, 2, size=n_labels)
        sample_sum = numpy.zeros(n_labels)
        count = 0

        for iteration in range(self.n_iterations):
            # Randomly permute label indices for update order
            label_order = rng.permutation(n_labels)
            for j in label_order:
                # Build augmented feature: [X_instance, current_state excluding label j]
                other_labels = numpy.delete(current_state, j)
                X_aug = numpy.hstack((X_instance, other_labels)).reshape(1, -1)
                # Obtain probability that label j is 1
                if hasattr(self.classifiers_[j], "predict_proba"):
                    p_j = self.classifiers_[j].predict_proba(X_aug)[0, 1]
                else:
                    raise Exception("j-th classifier does not have predict_proba function")
                # Sample new value for label j from a Bernoulli distribution
                current_state[j] = 1 if rng.rand() < p_j else 0
            # After burn-in, accumulate the sample
            if iteration >= self.burn_in:
                sample_sum += current_state
                count += 1

        # Average collected samples to estimate marginal probabilities
        prob_estimates = sample_sum / count if count > 0 else current_state
        return prob_estimates

    def predict_proba(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Predict probability estimates for each label using Gibbs sampling.

        For each instance, the Gibbs sampler is run to approximate the marginal probability
        for each label.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_labels)
            Estimated probabilities for each label.
        """
        X = check_array(X)
        if hasattr(X, "shape"):
            n_samples = X.shape[0]
        else:
            raise Exception("Input samples do not have shape function")
        proba = numpy.zeros((n_samples, self.n_labels_))

        for i in range(n_samples):
            proba[i, :] = self._gibbs_sampling(X[i])
        return proba

    def predict(self, X: numpy.ndarray, threshold: float = 0.5) -> numpy.ndarray:
        """
        Predict the multi-label outputs for each instance.

        A label is predicted as relevant (1) if its estimated probability exceeds 0.5.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples, n_labels)
            Binary predictions for each label.
        """
        proba = self.predict_proba(X)
        predictions = (proba > threshold).astype(int)
        return predictions

    def evaluate(self, X: numpy.ndarray, Y_true: numpy.ndarray, threshold: float = 0.5) -> dict:
        """
        Evaluate the classifier using standard multi-label metrics.

        The function computes:
          - Subset Accuracy: fraction of samples with exactly correct label set.
          - Micro-averaged F1 score.
          - Hamming Loss.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input samples.
        Y_true : numpy.ndarray of shape (n_samples, n_labels)
            True binary indicator matrix of labels.

        Returns
        -------
        metrics : dict
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        Y_true = numpy.array(Y_true)
        Y_pred = self.predict(X, threshold)
        acc = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average="micro")
        hamming = hamming_loss(Y_true, Y_pred)
        return {"accuracy": acc, "f1_micro": f1, "hamming_loss": hamming}
