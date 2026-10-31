import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import fit_binary, positive_score, resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


def _binary_predict(est, X) -> numpy.ndarray:
    """0/1 prediction of a fitted binary estimator (score > 0.5 when it has
    ``predict_proba``, else its own ``predict``)."""
    if hasattr(est, "predict_proba"):
        return (positive_score(est, X) > 0.5).astype(int)
    return (numpy.asarray(est.predict(X)) == 1).astype(int)


class CCClassifier(BaseEstimator, ClassifierMixin):
    """Classifier Chain (Zhang & Zhou 2014, Fig. 4, Eq. 6-8).

    Link j is trained on ``[X, true labels of the earlier chain positions]``;
    at prediction time the earlier links' binary predictions are appended
    instead.

    Parameters
    ----------
    base_estimator : estimator, default None
        Binary classifier cloned for each link. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    order : None, "random" or sequence of int, default None
        Label order of the chain. None is the natural order; a sequence must
        be a permutation of ``range(n_labels)``; "random" draws a permutation
        from ``random_state`` at every fit.
    random_state : int or None, default None
        Seed of the "random" order.

    Attributes
    ----------
    estimators_ : list of fitted binary classifiers, in chain order
    order_ : ndarray of int, the chain order used (label index per link)
    n_labels_ : int

    Notes
    -----
    Outputs of ``predict`` and ``predict_proba`` are in the original label
    column order. ``predict_proba`` returns each link's positive score while
    chaining the binary predictions (consistently with training).
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None, order=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state

    def _resolve_order(self, q):
        if self.order is None:
            return numpy.arange(q)
        if isinstance(self.order, str):
            if self.order != "random":
                raise ValueError("order must be None, 'random' or a permutation of the label indices")
            return numpy.random.RandomState(self.random_state).permutation(q)
        order = numpy.asarray(self.order)
        if order.ndim != 1 or order.size != q or not numpy.array_equal(numpy.sort(order), numpy.arange(q)):
            raise ValueError(f"order must be a permutation of range({q})")
        return order.astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CCClassifier":
        """
        Fit the chain: link j sees X and the true labels of earlier links.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
        Y : ArrayLike of shape (n_samples, n_labels)

        Returns
        -------
        self : "CCClassifier"
        """
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        q = Y.shape[1]
        order = self._resolve_order(q)
        template = resolve_estimator(self.base_estimator)
        estimators = []
        X_ext = X
        for label in order:
            estimators.append(fit_binary(template, X_ext, Y[:, label]))
            X_ext = numpy.hstack((X_ext, Y[:, [label]]))
        self.estimators_ = estimators
        self.order_ = order
        self.n_labels_ = q
        return self

    def _chain(self, X):
        X = numpy.asarray(X)
        n = X.shape[0]
        Y_pred = numpy.zeros((n, self.n_labels_), dtype=int)
        Y_score = numpy.zeros((n, self.n_labels_))
        X_ext = X
        for est, label in zip(self.estimators_, self.order_):
            Y_score[:, label] = positive_score(est, X_ext)
            Y_pred[:, label] = _binary_predict(est, X_ext)
            X_ext = numpy.hstack((X_ext, Y_pred[:, [label]]))
        return Y_pred, Y_score

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict labels, chaining the binary predictions.

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels), original label order
        """
        return self._chain(X)[0]

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Positive-class score of each link, chaining binary predictions.

        Returns
        -------
        Y_score : ndarray of shape (n_samples, n_labels), original label order
        """
        return self._chain(X)[1]

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate on the given data; see ``functions.assess`` (AUC uses
        ``predict_proba`` scores).
        """
        Y_pred, Y_score = self._chain(X)
        return assess(Y, Y_pred, Y_score)
