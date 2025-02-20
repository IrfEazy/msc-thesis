from typing import cast

import numpy
from numpy._typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from tqdm.notebook import tqdm
from typing_extensions import TypeVar

from MLC.preconditions import check_same_rows, check_binary_matrices


class CCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: ClassifierMixin = LogisticRegression(), order=None):
        """
        Initialize the Classifier Chain.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The base classifier to be used for each binary classification task.
        order
            The order of labels in the chain. If None, the natural order is used.
        """
        self.order_ = None
        self.base_estimator = base_estimator
        self.order = order
        self.chain = []

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CCClassifier":
        """
        Fit the classifier chain on the training data.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        Y : ArrayLike
            Label matrix of shape (n_samples, n_labels).
        """
        n_samples, n_labels = Y.shape
        self.order_ = self.order if self.order is not None else range(n_labels)
        X_extended = numpy.copy(X)
        T = TypeVar("T", bound=ClassifierMixin)

        for i in self.order_:
            clf: T = cast(T, clone(self.base_estimator))
            clf.fit(X_extended, Y[:, i])
            self.chain.append(clf)
            # Augment feature space with the current label's predictions
            predictions = clf.predict(X_extended).reshape(-1, 1)
            X_extended = numpy.hstack((X_extended, predictions))
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict labels for the given data.

        Parameters
        ----------
        X: ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Y_pred: ArrayLike
            Predicted label matrix of shape (n_samples, n_labels).
        """
        n_samples = X.shape[0]
        X_extended = numpy.copy(X)
        Y_pred = numpy.zeros((n_samples, len(self.chain)))
        T = TypeVar("T", bound=ClassifierMixin)

        for i, clf in enumerate(tqdm(self.chain, desc="Predicting for each classifier")):
            clf: T = cast(T, clf)
            Y_pred[:, i] = clf.predict(X_extended)
            X_extended = numpy.hstack((X_extended, Y_pred[:, i].reshape(-1, 1)))

        return Y_pred

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict label probabilities for the given data.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Y_proba : ArrayLike
            Predicted label probability matrix of shape (n_samples, n_labels).
        """
        n_samples = X.shape[0]
        X_extended = numpy.copy(X)
        Y_proba = numpy.zeros((n_samples, len(self.chain)))
        T = TypeVar("T", bound=ClassifierMixin)

        for i, clf in enumerate(tqdm(self.chain, desc="Predicting for each classifier")):
            clf: T = cast(T, clf)
            Y_proba[:, i] = clf.predict_proba(X_extended)[:, 1]
            X_extended = numpy.hstack((X_extended, Y_proba[:, i].reshape(-1, 1)))
        return Y_proba

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y_true: ArrayLike) -> dict[str, float]:
        """
        Evaluate the classifier chain on the given test data.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        Y_true : ArrayLike
            True label matrix of shape (n_samples, n_labels).

        Returns
        -------
        metrics : dict[str, float]
            Dictionary containing evaluation metrics.
        """
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average="micro")
        hamming = hamming_loss(Y_true, Y_pred)
        return {"accuracy": accuracy, "f1_micro": f1, "hamming_loss": hamming}
