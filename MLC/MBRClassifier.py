from typing import cast, TypeVar

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class MBRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: ClassifierMixin = LogisticRegression(max_iter=1000),
        meta_estimator: ClassifierMixin = LogisticRegression(max_iter=1000),
    ):
        """
        Parameters
        ----------
        base_estimator : ClassifierMixin
            The classifier used in the first (BR) stage.
        meta_estimator : ClassifierMixin
            The classifier used in the second (meta) stage.
        """
        self.meta_classifiers_ = None
        self.first_stage_classifiers_ = None
        self.n_labels_ = None
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "MBRClassifier":
        """
        Fit the Meta Binary Relevance classifier.

        This method trains |L| binary classifiers in the first stage (BR) using the entire training set.
        Then, it obtains first-stage predictions to augment the original feature space, and trains |L|
        meta classifiers on the extended features.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The training input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            The binary indicator matrix for labels.

        Returns
        -------
        self : "MBRClassifier"
            Returns self.
        """
        Y = numpy.array(Y)
        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels

        # -----------------------------
        # First Stage: Binary Relevance
        # -----------------------------
        self.first_stage_classifiers_ = []
        T = TypeVar("T", bound=ClassifierMixin)
        for i in range(n_labels):
            clf: T = cast(T, clone(self.base_estimator))
            clf.fit(X, Y[:, i])
            self.first_stage_classifiers_.append(clf)

        # Generate first-stage predictions on training data.
        # We assume that the estimator has a predict_proba method.
        first_stage_predictions = numpy.zeros((n_samples, n_labels))
        for i in range(n_labels):
            # Use probability for the positive class
            first_stage_predictions[:, i] = self.first_stage_classifiers_[
                i
            ].predict_proba(X)[:, 1]

        # Augment the original features with the first-stage predictions.
        X_meta = numpy.hstack((X, first_stage_predictions))

        # -----------------------------------------
        # Second Stage: Meta-Level Binary Relevance
        # -----------------------------------------
        self.meta_classifiers_ = []
        for i in range(n_labels):
            clf: T = cast(T, clone(self.meta_estimator))
            clf.fit(X_meta, Y[:, i])
            self.meta_classifiers_.append(clf)
        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict probability estimates for each label.

        First, the first-stage classifiers are applied to X to generate predictions.
        These predictions augment the original feature space to form X_meta.
        Then, the meta classifiers provide the final probability estimates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        probabilities : ArrayLike of shape (n_samples, n_labels)
            Probability estimates for each label.
        """
        n_samples = X.shape[0]

        # First-stage predictions on new data.
        first_stage_predictions = numpy.zeros((n_samples, self.n_labels_))
        for i in range(self.n_labels_):
            first_stage_predictions[:, i] = self.first_stage_classifiers_[
                i
            ].predict_proba(X)[:, 1]

        # Augment original features with first-stage predictions.
        X_meta = numpy.hstack((X, first_stage_predictions))

        # Second-stage (meta) predictions.
        probabilities = numpy.zeros((n_samples, self.n_labels_))
        for i in range(self.n_labels_):
            probabilities[:, i] = self.meta_classifiers_[i].predict_proba(X_meta)[:, 1]
        return probabilities

    def predict(self, X: ArrayLike):
        """
        Predict the multi-label outputs for each instance.

        Labels are predicted as relevant (1) if the estimated probability exceeds 0.5.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : ArrayLike of shape (n_samples, n_labels)
            Binary predictions for each label.
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities > 0.5).astype(int)
        return predictions

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike):
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
        metrics : dict
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        return assess(Y, self.predict(X))
