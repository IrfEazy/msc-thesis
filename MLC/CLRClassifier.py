from typing import cast

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from typing_extensions import TypeVar

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class CLRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: ClassifierMixin = LogisticRegression()):
        """
        Parameters
        ----------
        base_estimator : scikit-learn classifier, default=LogisticRegression()
            The base binary classifier to be used for both pairwise and auxiliary classifiers.
        """
        self.q_ = None
        self.pairwise_classifiers_ = None
        self.aux_classifiers_ = None
        self.base_estimator = base_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CLRClassifier":
        """
        Fit the Calibrated Label Ranking classifier.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The training input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix with 1 indicating that the label is relevant.

        Returns
        -------
        self : CLRClassifier
        """
        # Validate inputs
        Y = numpy.array(Y)
        n_samples, self.q_ = Y.shape

        # Initialize dictionaries for pairwise and auxiliary classifiers
        self.pairwise_classifiers_ = {}
        self.aux_classifiers_ = {}
        T = TypeVar("T", bound=ClassifierMixin)

        # Train pairwise classifiers for each pair (j, k), j < k
        for j in range(self.q_):
            for k in range(j + 1, self.q_):
                X_pair = []
                y_pair = []
                # For each instance, consider it only if labels differ
                for i in range(n_samples):
                    if Y[i, j] != Y[i, k]:
                        X_pair.append(X[i])
                        # Define target: 1 if label j is relevant, 0 if label k is relevant
                        y_pair.append(1 if Y[i, j] == 1 else 0)
                if len(X_pair) > 0:
                    X_pair = numpy.array(X_pair)
                    y_pair = numpy.array(y_pair)
                    clf: T = cast(T, clone(self.base_estimator))
                    clf.fit(X_pair, y_pair)
                    self.pairwise_classifiers_[(j, k)] = clf

        # Train auxiliary classifiers (for each label vs. the virtual label)
        for j in range(self.q_):
            clf_aux: T = cast(T, clone(self.base_estimator))
            clf_aux.fit(X, Y[:, j])
            self.aux_classifiers_[j] = clf_aux
        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Compute calibrated scores for each label.

        For each instance, the pairwise classifiers produce a normalized vote count (between 0 and 1)
        for each label. In parallel, the auxiliary classifiers produce a probability of label relevance.
        The calibrated score is the difference: (normalized vote) - (auxiliary probability).

        Parameters:
        -----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        calibrated_scores : ArrayLike of shape (n_samples, n_labels)
            Calibrated scores for each label. A positive score suggests relevance.
        """
        n_samples = X.shape[0]
        votes = numpy.zeros((n_samples, self.q_))

        # Aggregate votes from pairwise classifiers
        for (j, k), clf in self.pairwise_classifiers_.items():
            predictions = clf.predict(X)
            # For each sample, if prediction is 1 then label j wins; otherwise label k wins.
            for i, pred in enumerate(predictions):
                if pred == 1:
                    votes[i, j] += 1
                else:
                    votes[i, k] += 1

        # Normalize votes: maximum votes per label is (q - 1)
        norm_votes = votes / (self.q_ - 1)

        # Obtain auxiliary probabilities for each label
        aux_probs = numpy.zeros((n_samples, self.q_))
        for j, clf in self.aux_classifiers_.items():
            # Assuming the estimator has a predict_proba method;
            # otherwise one might use decision_function and calibrate it.
            aux_probs[:, j] = clf.predict_proba(X)[:, 1]

        # Calibrated score: if norm_votes exceed the auxiliary probability, label is deemed relevant.
        calibrated_scores = norm_votes - aux_probs
        return calibrated_scores

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the multi-label set for each instance.

        A label is predicted as relevant if its calibrated score is positive.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix of predicted labels.
        """
        calibrated_scores = self.predict_proba(X)
        predictions = (calibrated_scores > 0).astype(int)
        return predictions

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
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

        Returns
        -------
        metrics : dict[str, float]
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        return assess(Y, self.predict(X))
