from typing import cast, TypeVar

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class LPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, base_estimator: ClassifierMixin = LogisticRegression(max_iter=1000)
    ):
        """
        Initialize the Label Powerset classifier.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The multi-class classifier to be used on the transformed problem.
        """
        self.classifier_ = None
        self.class_to_label_ = None
        self.class_label_matrix_ = None
        self.label_to_class_ = None
        self.n_labels_ = None
        self.base_estimator = base_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "LPClassifier":
        """
        Fit the Label Powerset classifier.

        The multi-label training set is transformed into a multi-class training set by mapping
        each unique label set to a unique class. A multi-class classifier is then trained on this data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input feature matrix.
        Y : ArrayLike of shape (n_samples, n_labels)
            The binary indicator matrix for labels.

        Returns
        -------
        self : "LPClassifier"
            Fitted estimator.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = numpy.array(Y)
        self.n_labels_ = Y.shape[1]

        # Map each label vector to a tuple so it can be used as a dict key.
        label_tuples = [tuple(row) for row in Y]
        # Create a mapping: unique label-set tuple -> unique class index.
        unique_label_tuples = list(set(label_tuples))
        self.label_to_class_ = {
            label_tuple: idx for idx, label_tuple in enumerate(unique_label_tuples)
        }
        self.class_to_label_ = {
            idx: numpy.array(label_tuple)
            for label_tuple, idx in self.label_to_class_.items()
        }

        # Transform multi-label targets into a single multi-class target.
        Y_transformed = numpy.array([self.label_to_class_[tuple(row)] for row in Y])

        # Train the multi-class classifier.
        T = TypeVar("T", bound=ClassifierMixin)
        self.classifier_: T = cast(T, clone(self.base_estimator))
        self.classifier_.fit(X, Y_transformed)

        # Precompute a matrix of label sets for each class.
        n_classes = len(self.class_to_label_)
        self.class_label_matrix_ = numpy.zeros((n_classes, self.n_labels_), dtype=int)
        for class_idx, label_arr in self.class_to_label_.items():
            self.class_label_matrix_[class_idx, :] = label_arr
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict multi-label outputs for the given data.

        For each instance, the multi-class classifier is used to predict a class,
        which is then mapped back to the corresponding multi-label set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_pred : ArrayLike of shape (n_samples, n_labels)
            The predicted binary label matrix.
        """
        Y_class_pred = self.classifier_.predict(X)
        Y_pred = numpy.array([self.class_to_label_[cls] for cls in Y_class_pred])
        return Y_pred

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict marginal probabilities for each label.

        For each instance, the multi-class classifier outputs probabilities for each class.
        The marginal probability for a given label is computed by summing the probabilities of
        all classes whose corresponding label set has that label present.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y_proba : ArrayLike of shape (n_samples, n_labels)
            The estimated probability for each label.
        """
        # Get multi-class probability estimates.
        proba = self.classifier_.predict_proba(X)  # shape: (n_samples, n_classes)
        # Compute marginal probabilities: dot the probability matrix with the class label matrix.
        Y_proba = numpy.dot(proba, self.class_label_matrix_)
        return Y_proba

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate the Label Powerset classifier using standard multi-label metrics.

        The following metrics are computed:
          - Subset Accuracy
          - Hamming Loss
          - Micro-averaged F1 Score

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input samples.
        Y : ArrayLike of shape (n_samples, n_labels)
            The true binary indicator matrix for labels.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary containing accuracy, hamming loss, and micro F1 score.
        """
        return assess(Y, self.predict(X))
