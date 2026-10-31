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


class CLRClassifier(BaseEstimator, ClassifierMixin):
    """Calibrated Label Ranking (Zhang & Zhou 2014, Fig. 5, Eq. 9-14).

    Pairwise models g_jk (j < k) are trained only on the rows where exactly
    one of the two labels is relevant (target: label j relevant). q auxiliary
    BR models g_jV (label j against the virtual label) are trained on all rows.
    Votes: zeta(j) counts the pairwise wins of j (binary predictions);
    zeta*(j) = zeta(j) + [g_jV(x) = 1]; zeta*(V) = sum_j [g_jV(x) = 0].
    Label j is predicted iff zeta*(j) > zeta*(V).

    Parameters
    ----------
    base_estimator : estimator, default None
        Binary classifier cloned for every pairwise and auxiliary model. None
        means a fresh ``LogisticRegression(max_iter=1000)``.

    Attributes
    ----------
    pairwise_estimators_ : dict {(j, k): fitted classifier}
        A pair with no row where the labels differ has no model and casts no
        vote. A single-class pair target gives a constant predictor.
    aux_estimators_ : list of q fitted classifiers (label vs virtual label)
    n_labels_ : int

    Notes
    -----
    ``predict_proba`` returns the normalised vote scores zeta*(j) / q in
    [0, 1]; they are vote fractions, not calibrated probabilities.
    """

    def __init__(self, base_estimator: ClassifierMixin | None = None):
        self.base_estimator = base_estimator

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "CLRClassifier":
        """
        Fit the pairwise and auxiliary models.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
        Y : ArrayLike of shape (n_samples, n_labels)

        Returns
        -------
        self : CLRClassifier
        """
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        q = Y.shape[1]
        template = resolve_estimator(self.base_estimator)
        pairwise = {}
        for j in range(q):
            for k in range(j + 1, q):
                mask = Y[:, j] != Y[:, k]
                if not mask.any():
                    continue
                pairwise[(j, k)] = fit_binary(template, X[mask], Y[mask, j])
        self.pairwise_estimators_ = pairwise
        self.aux_estimators_ = [fit_binary(template, X, Y[:, j]) for j in range(q)]
        self.n_labels_ = q
        return self

    def _votes(self, X):
        """Return (zeta*(j) of shape (n, q), zeta*(V) of shape (n,))."""
        X = numpy.asarray(X)
        n, q = X.shape[0], self.n_labels_
        zeta = numpy.zeros((n, q), dtype=int)
        for (j, k), est in self.pairwise_estimators_.items():
            p = _binary_predict(est, X)
            zeta[:, j] += p
            zeta[:, k] += 1 - p
        aux = numpy.column_stack([_binary_predict(e, X) for e in self.aux_estimators_])
        return zeta + aux, (1 - aux).sum(axis=1)

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Normalised vote scores zeta*(j) / q, shape (n_samples, n_labels).
        Vote fractions, not probabilities.
        """
        return self._votes(X)[0] / self.n_labels_

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict label j iff zeta*(j) > zeta*(V) (strictly).

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
        """
        zeta, zeta_v = self._votes(X)
        return (zeta > zeta_v[:, None]).astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate on the given data; see ``functions.assess`` (AUC uses the
        normalised vote scores).
        """
        return assess(Y, self.predict(X), self.predict_proba(X))
