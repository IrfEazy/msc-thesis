import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import fit_binary, positive_score, resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class BRClassifier(BaseEstimator, ClassifierMixin):
    """Binary Relevance: one independent binary classifier per label
    (Zhang & Zhou 2014, Fig. 3).

    Parameters
    ----------
    base_estimator : estimator, default None
        Binary classifier cloned for each label. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    t_criterion : bool, default False
        If True, a sample with no predicted label gets the label with the
        highest score (T-criterion). The paper's BR does not apply it.

    Attributes
    ----------
    estimators_ : list of fitted binary classifiers, one per label
        A label with a constant training column gets a constant predictor.
    n_labels_ : int
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None, t_criterion=False):
        self.base_estimator = base_estimator
        self.t_criterion = t_criterion

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "BRClassifier":
        """
        Fit one binary classifier per label.

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
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        template = resolve_estimator(self.base_estimator)
        self.estimators_ = [fit_binary(template, X, Y[:, i]) for i in range(Y.shape[1])]
        self.n_labels_ = Y.shape[1]
        return self

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Positive-class score of each label.

        Returns
        -------
        Y_proba : ndarray of shape (n_samples, n_labels)
            Probability of each label (decision-function value for estimators
            without ``predict_proba``).
        """
        X = numpy.asarray(X)
        return numpy.column_stack([positive_score(e, X) for e in self.estimators_])

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict labels for the given data.

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
            The predicted binary label matrix. Estimators with
            ``predict_proba`` are thresholded at 0.5; others use their own
            ``predict``.
        """
        X = numpy.asarray(X)
        cols = []
        for e in self.estimators_:
            if hasattr(e, "predict_proba"):
                cols.append(positive_score(e, X) > 0.5)
            else:
                cols.append(numpy.asarray(e.predict(X)) == 1)
        Y_pred = numpy.column_stack(cols).astype(int)
        if self.t_criterion:
            empty = Y_pred.sum(axis=1) == 0
            if empty.any():
                best = self.predict_proba(X[empty]).argmax(axis=1)
                Y_pred[numpy.flatnonzero(empty), best] = 1
        return Y_pred

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate the model on the given data.

        Returns
        -------
        metrics : dict
            See ``functions.assess``; AUC uses ``predict_proba`` scores.
        """
        return assess(Y, self.predict(X), self.predict_proba(X))
