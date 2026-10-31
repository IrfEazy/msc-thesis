import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from ._base import resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class LPClassifier(BaseEstimator, ClassifierMixin):
    """Label Powerset: every distinct labelset becomes one class of a
    multi-class problem (Zhang & Zhou 2014, p.20, Eq. 15-17).

    Parameters
    ----------
    base_estimator : estimator, default None
        Multi-class classifier cloned for the transformed problem. None means
        a fresh ``LogisticRegression(max_iter=1000)``.

    Attributes
    ----------
    labelsets_ : ndarray of shape (n_classes, n_labels)
        The distinct training labelsets in lexicographic order; the row index
        is the class id used for the multi-class problem.
    classifier_ : fitted multi-class estimator, or None
        None when the training data has a single distinct labelset (the
        prediction is then constant and nothing is fitted).
    n_labels_ : int

    Notes
    -----
    ``predict_proba`` returns marginal label probabilities: the class
    probabilities times the labelset matrix. An estimator without
    ``predict_proba`` contributes one-hot class indicators.
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None):
        self.base_estimator = base_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "LPClassifier":
        """
        Fit the Label Powerset classifier.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix.

        Returns
        -------
        self : "LPClassifier"
        """
        X = numpy.asarray(X)
        Y = numpy.asarray(Y).astype(int)
        self.labelsets_, y_class = numpy.unique(Y, axis=0, return_inverse=True)
        y_class = numpy.asarray(y_class).ravel()
        self.n_labels_ = Y.shape[1]
        if self.labelsets_.shape[0] < 2:
            self.classifier_ = None
        else:
            self.classifier_ = clone(resolve_estimator(self.base_estimator)).fit(X, y_class)
        return self

    def _class_proba(self, X):
        """Probability per row of ``labelsets_`` (columns follow class ids)."""
        n = X.shape[0]
        proba = numpy.zeros((n, self.labelsets_.shape[0]))
        if self.classifier_ is None:
            proba[:, 0] = 1.0
            return proba
        classes = numpy.asarray(self.classifier_.classes_).astype(int)
        if hasattr(self.classifier_, "predict_proba"):
            proba[:, classes] = self.classifier_.predict_proba(X)
        else:
            pred = numpy.asarray(self.classifier_.predict(X)).astype(int)
            proba[numpy.arange(n), pred] = 1.0
        return proba

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict the labelset of each sample.

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
            Always one of the training labelsets.
        """
        X = numpy.asarray(X)
        if self.classifier_ is None:
            return numpy.tile(self.labelsets_[0], (X.shape[0], 1)).astype(int)
        pred = numpy.asarray(self.classifier_.predict(X)).astype(int)
        return self.labelsets_[pred].astype(int)

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Marginal probability of each label (sum of the probabilities of the
        labelsets that contain it).

        Returns
        -------
        Y_proba : ndarray of shape (n_samples, n_labels), values in [0, 1]
        """
        X = numpy.asarray(X)
        return self._class_proba(X) @ self.labelsets_

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate on the given data; see ``functions.assess`` (AUC uses the
        marginal probabilities of ``predict_proba``).
        """
        return assess(Y, self.predict(X), self.predict_proba(X))
