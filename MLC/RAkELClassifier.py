import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.utils.validation import check_array, check_X_y


class RAkELClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, k=3, n_estimators=None, random_state=None):
        """
        Initialize the Random k-Labelsets (RAkEL) classifier.

        Parameters
        ----------
        base_estimator : classifier object, default=LogisticRegression(max_iter=1000)
            The multi-class classifier to be used on each k-labelset LP problem.
        k : int, default=3
            The number of labels in each randomly chosen label subset.
        n_estimators : int or None, default=None
            The number of LP classifiers to ensemble. If None, it is set to 2 * q, where q is the number of labels.
        random_state : int or None, default=None
            Seed for random number generation.
        """
        if base_estimator is None:
            base_estimator = LogisticRegression(max_iter=1000)
        self.base_estimator = base_estimator
        self.k = k
        self.n_estimators = n_estimators  # Will be set in fit() if None.
        self.random_state = random_state

    def fit(self, X, Y):
        """
        Fit the RAkEL classifier.

        For each ensemble member, a random subset of k labels is selected and an LP classifier is trained.
        The multi-label targets are transformed by taking the intersection with the chosen label subset
        and mapping each unique combination to a unique class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.
        Y : array-like of shape (n_samples, n_labels)
            The binary indicator matrix for labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = numpy.array(Y)
        n_samples, q = Y.shape
        self.q_ = q

        # Set ensemble size if not provided.
        if self.n_estimators is None:
            self.n_estimators = 2 * q

        self.ensemble_ = []
        rng = numpy.random.RandomState(self.random_state)

        # For each ensemble member, select a random k-labelset and train an LP classifier.
        for r in range(self.n_estimators):
            # Randomly choose k unique label indices from 0 to q-1.
            subset = rng.choice(q, size=self.k, replace=False)
            subset = numpy.sort(subset)  # For consistency.

            # Transform the multi-label target: extract only the labels in the chosen subset.
            targets = [tuple(Y[i, subset]) for i in range(n_samples)]

            # Map each unique label combination to a unique class.
            unique_targets = list(set(targets))
            mapping = {target: idx for idx, target in enumerate(unique_targets)}
            reverse_mapping = {idx: numpy.array(target) for target, idx in mapping.items()}

            # Create the multi-class target for the current LP classifier.
            Y_transformed = numpy.array([mapping[target] for target in targets])

            # Train a clone of the base estimator on (X, Y_transformed).
            clf = clone(self.base_estimator)
            clf.fit(X, Y_transformed)

            # Save the ensemble component.
            self.ensemble_.append({
                'subset': subset,  # The indices of labels for this classifier.
                'mapping': mapping,  # Mapping from label tuple to class index.
                'reverse_mapping': reverse_mapping,  # Mapping from class index back to label tuple.
                'classifier': clf
            })

        return self

    def predict(self, X):
        """
        Predict multi-label outputs for the given data.

        Each ensemble member votes on the labels in its k-labelset. For each label in the full set,
        we count the number of votes (μ) and the number of classifiers that consider that label (τ).
        A label is predicted as relevant if μ/τ > 0.5.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_labels)
            The predicted binary label matrix.
        """
        X = check_array(X)
        n_samples = X.shape[0]
        # Initialize vote counts and classifier inclusion counts.
        votes = numpy.zeros((n_samples, self.q_))
        tau = numpy.zeros((n_samples, self.q_))

        # For each ensemble member:
        for component in self.ensemble_:
            subset = component['subset']
            clf = component['classifier']
            reverse_mapping = component['reverse_mapping']

            # Predict the class for each instance.
            preds = clf.predict(X)
            # Map predictions back to binary vectors for the chosen k-labelset.
            pred_binary = numpy.array([reverse_mapping[pred] for pred in preds])  # Shape: (n_samples, k)

            # For each label in the subset, update votes and tau.
            for idx, label in enumerate(subset):
                votes[:, label] += pred_binary[:, idx]  # Vote of 1 if predicted relevant.
                tau[:, label] += 1  # This classifier contributes to the vote count for label.

        # Compute vote ratio; avoid division by zero.
        ratio = numpy.divide(votes, tau, out=numpy.zeros_like(votes), where=tau != 0)
        # Predict label as relevant if the vote ratio exceeds 0.5.
        Y_pred = (ratio > 0.5).astype(int)
        return Y_pred

    def predict_proba(self, X):
        """
        Predict marginal probabilities for each label.

        For each label, the probability is computed as the ratio of votes (from ensemble members where
        the label is present) to the number of ensemble members that consider that label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_proba : ndarray of shape (n_samples, n_labels)
            The estimated probability for each label.
        """
        X = check_array(X)
        n_samples = X.shape[0]
        votes = numpy.zeros((n_samples, self.q_))
        tau = numpy.zeros((n_samples, self.q_))

        for component in self.ensemble_:
            subset = component['subset']
            clf = component['classifier']
            reverse_mapping = component['reverse_mapping']

            preds = clf.predict(X)
            pred_binary = numpy.array([reverse_mapping[pred] for pred in preds])

            for idx, label in enumerate(subset):
                votes[:, label] += pred_binary[:, idx]
                tau[:, label] += 1

        Y_proba = numpy.divide(votes, tau, out=numpy.zeros_like(votes), where=tau != 0)
        return Y_proba

    def evaluate(self, X, Y_true):
        """
        Evaluate the RAkEL classifier using common multi-label metrics.

        Metrics computed include:
          - Subset Accuracy
          - Hamming Loss
          - Micro-averaged F1 Score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Y_true : array-like of shape (n_samples, n_labels)
            The true binary label matrix.

        Returns
        -------
        metrics : dict
            A dictionary containing 'accuracy', 'hamming_loss', and 'f1_micro'.
        """
        Y_true = numpy.array(Y_true)
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y_true, Y_pred)
        hamming = hamming_loss(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average='micro')
        return {"accuracy": accuracy, "hamming_loss": hamming, "f1_micro": f1}
