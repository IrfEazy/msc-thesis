from typing import TypeVar, cast

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class BRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: ClassifierMixin = LogisticRegression()):
        """
        Initialize the Binary Relevance classifier.

        Parameters:
        base_estimator: ClassifierMixin
            The base classifier to use for each binary problem.
        """
        self.classifiers_ = None
        self.base_estimator = base_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "BRClassifier":
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
        self : "BRClassifier"
        """
        n_labels = Y.shape[1]
        self.classifiers_ = []
        T = TypeVar("T", bound=ClassifierMixin)

        for i in range(n_labels):
            clf: T = cast(T, clone(self.base_estimator))
            clf.fit(X, Y[:, i])
            self.classifiers_.append(clf)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict labels for the given data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input features.

        Returns
        -------
        Y_pred : ArrayLike of shape (n_samples, n_labels)
            The predicted binary label matrix.
        """
        n_samples = X.shape[0]
        n_labels = len(self.classifiers_)
        Y_pred = numpy.zeros((n_samples, n_labels))
        T = TypeVar("T", bound=ClassifierMixin)

        for i, clf in enumerate(self.classifiers_):
            clf: T = cast(T, clf)
            Y_pred[:, i] = clf.predict(X)
        return Y_pred

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict label probabilities for the given data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input features.

        Returns
        -------
        Y_proba : ArrayLike of shape (n_samples, n_labels)
            The predicted probability matrix.
        """
        n_samples = X.shape[0]
        n_labels = len(self.classifiers_)
        Y_proba = numpy.zeros((n_samples, n_labels))
        T = TypeVar("T", bound=ClassifierMixin)

        for i, clf in enumerate(self.classifiers_):
            clf: T = cast(T, clf)
            Y_proba[:, i] = clf.predict_proba(X)[:, 1]
        return Y_proba

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate the model on the given data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input features.
        Y : ArrayLike of shape (n_samples, n_labels)
            The true binary label matrix.

        Returns
        -------
        metrics : dict[str, float]
            Dictionary containing accuracy, micro F1 score, and hamming loss.
        """
        return assess(Y, self.predict(X))
