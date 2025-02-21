from typing import Optional, cast

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from typing_extensions import TypeVar

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class CDNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: ClassifierMixin = LogisticRegression(),
        n_iterations: int = 100,
        burn_in: int = 50,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Conditional Dependency Network classifier.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The estimator used to model the conditional probability for each label.
        n_iterations : int
            Total number of Gibbs sampling iterations during inference.
        burn_in : int
            Number of initial iterations to discard for burn-in.
        random_state : Optional[None]
            Seed for random number generation.
        """
        self.classifiers_ = None
        self.n_features_ = None
        self.n_labels_ = None
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.random_state = random_state

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CDNClassifier":
        """
        Fit the CDN classifier.

        For each label, a binary classifier is trained to predict that label using
        the original features augmented by all other labels (i.e., the label's Markov blanket).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix of labels.

        Returns
        -------
        self : "CDNClassifier"
            Fitted estimator.
        """
        # Validate inputs
        Y = numpy.array(Y)
        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels
        self.n_features_ = X.shape[1]

        # Dictionary to store a classifier for each label
        self.classifiers_ = {}
        T = TypeVar("T", bound=ClassifierMixin)

        # Train a binary classifier for each label j
        for j in range(n_labels):
            # Construct augmented features: [X, Y_without_label_j]
            other_labels = numpy.delete(Y, j, axis=1)
            X_aug = numpy.hstack((X, other_labels))
            clf: T = cast(T, clone(self.base_estimator))
            clf.fit(X_aug, Y[:, j])
            self.classifiers_[j] = clf
        return self

    def _gibbs_sampling(self, X_instance: ArrayLike) -> ArrayLike:
        """
        Perform Gibbs sampling for a single instance.

        Starting from a random initialization of the label vector, the sampler iteratively
        updates each label (in random order) using the conditional probability model.
        Samples collected after the burn-in period are averaged to estimate the marginal probabilities.

        Parameters
        ----------
        X_instance : ArrayLike of shape (n_features,)
            The feature vector for one instance.

        Returns
        -------
        prob_estimates : ArrayLike of shape (n_labels,)
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
                p_j = self.classifiers_[j].predict_proba(X_aug)[0, 1]
                # Sample new value for label j from a Bernoulli distribution
                current_state[j] = 1 if rng.rand() < p_j else 0
            # After burn-in, accumulate the sample
            if iteration >= self.burn_in:
                sample_sum += current_state
                count += 1

        # Average collected samples to estimate marginal probabilities
        prob_estimates = sample_sum / count if count > 0 else current_state
        return prob_estimates

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict probability estimates for each label using Gibbs sampling.

        For each instance, the Gibbs sampler is run to approximate the marginal probability
        for each label.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ArrayLike of shape (n_samples, n_labels)
            Estimated probabilities for each label.
        """
        n_samples = X.shape[0]
        proba = numpy.zeros((n_samples, self.n_labels_))

        for i in range(n_samples):
            proba[i, :] = self._gibbs_sampling(X[i])
        return proba

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> ArrayLike:
        """
        Predict the multi-label outputs for each instance.

        A label is predicted as relevant (1) if its estimated probability exceeds 0.5.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.
        threshold : float
            Threshold probability.

        Returns
        -------
        predictions : ArrayLike of shape (n_samples, n_labels)
            Binary predictions for each label.
        """
        proba = self.predict_proba(X)
        predictions = (proba > threshold).astype(int)
        return predictions

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(
        self, X: ArrayLike, Y: ArrayLike, threshold: float = 0.5
    ) -> dict[str, float]:
        """
        Evaluate the classifier using standard multi-label metrics.

        The function computes:
          - Subset Accuracy: fraction of samples with exactly correct label set.
          - Micro-averaged F1 score.
          - Hamming Loss.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            True binary indicator matrix of labels.
        threshold : float
            Threshold probability.

        Returns
        -------
        metrics : dict[str, float]
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        return assess(Y, self.predict(X, threshold))
