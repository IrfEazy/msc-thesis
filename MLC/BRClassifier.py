from typing import TypeVar, cast

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

from preconditions import check_same_rows, check_binary_matrices


class BRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: ClassifierMixin = LogisticRegression()):
        """
        Initialize the Binary Relevance classifier.

        Parameters:
        base_estimator: ClassifierMixin
            The base classifier to use for each binary problem.
        """
        self.classifiers_ = None
        self.base_classifier = base_estimator

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
        self : object
        """
        n_labels = Y.shape[1]
        self.classifiers_ = []
        T = TypeVar("T", bound=ClassifierMixin)

        for i in range(n_labels):
            clf: T = cast(T, clone(self.base_classifier))
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

        for i, clf in enumerate(self.classifiers_):
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

        for i, clf in enumerate(self.classifiers_):
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
            A dictionary containing evaluation metrics.
        """
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        f1 = f1_score(Y, Y_pred, average="micro")
        hamming = hamming_loss(Y, Y_pred)
        return {"accuracy": accuracy, "f1_micro": f1, "hamming_loss": hamming}
