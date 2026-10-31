import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin

from .LPClassifier import LPClassifier
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class PStClassifier(BaseEstimator, ClassifierMixin):
    """Pruned Sets: Label Powerset after pruning rare labelsets (Read et al.
    2008, as described by Bogatinovski et al. 2022).

    A labelset is frequent when it occurs at least ``pruning_value`` times in
    the training data. Rows with a frequent labelset are kept. Each pruned
    row is reintroduced once per kept subset: its frequent, non-empty, proper
    sub-labelsets are ranked by count (descending), size (descending) and then
    lexicographically, and the top ``max_reintroduced`` become new rows with
    the same features. A pruned row without a frequent proper subset is
    dropped. An LP classifier is then trained on the result. When every
    labelset occurs at least ``pruning_value`` times, nothing is pruned and
    PSt equals LP.

    Parameters
    ----------
    base_estimator : estimator, default None
        Multi-class classifier for the LP step (None: LogisticRegression).
    pruning_value : int, default 2
        Minimum labelset count p to be frequent (count >= p).
    max_reintroduced : int, default 1
        Maximum number b of subsets reintroduced per pruned row.

    Attributes
    ----------
    lp_ : fitted LPClassifier
    n_labels_ : int

    Raises
    ------
    ValueError
        On fit, if no row has a frequent labelset (lower ``pruning_value``).
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        pruning_value: int = 2,
        max_reintroduced: int = 1,
    ):
        self.base_estimator = base_estimator
        self.pruning_value = pruning_value
        self.max_reintroduced = max_reintroduced

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "PStClassifier":
        """
        Prune rare labelsets, reintroduce their frequent subsets and fit LP.

        Returns
        -------
        self : "PStClassifier"
        """
        if self.pruning_value < 1:
            raise ValueError("pruning_value must be >= 1")
        if self.max_reintroduced < 0:
            raise ValueError("max_reintroduced must be >= 0")
        X = numpy.asarray(X)
        Y = numpy.asarray(Y).astype(int)
        sets, inverse, counts = numpy.unique(
            Y, axis=0, return_inverse=True, return_counts=True
        )
        inverse = numpy.asarray(inverse).ravel()
        frequent = counts >= self.pruning_value
        if not frequent.any():
            raise ValueError(
                f"no labelset occurs at least pruning_value={self.pruning_value} "
                f"times (maximum count {counts.max()}); use a lower pruning_value"
            )
        # Frequent non-empty labelsets, ranked: count desc, size desc, lexicographic.
        cands = [
            (-int(counts[i]), -int(sets[i].sum()), tuple(sets[i]), i)
            for i in numpy.flatnonzero(frequent)
            if sets[i].sum() > 0
        ]
        cands.sort(key=lambda c: c[:3])

        keep_rows, new_rows, new_sets = [], [], []
        for r in range(Y.shape[0]):
            cls = inverse[r]
            if frequent[cls]:
                keep_rows.append(r)
                continue
            row = Y[r]
            taken = 0
            for _, _, _, i in cands:
                if taken >= self.max_reintroduced:
                    break
                s = sets[i]
                if s.sum() < row.sum() and (s <= row).all():
                    new_rows.append(r)
                    new_sets.append(s)
                    taken += 1
        X_new = numpy.concatenate([X[keep_rows], X[new_rows]]) if new_rows else X[keep_rows]
        Y_new = (
            numpy.concatenate([Y[keep_rows], numpy.array(new_sets)])
            if new_rows
            else Y[keep_rows]
        )
        self.n_labels_ = Y.shape[1]
        self.lp_ = LPClassifier(self.base_estimator).fit(X_new, Y_new)
        return self

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """Predict a retained labelset per sample; int array (n, n_labels)."""
        return self.lp_.predict(X)

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """Marginal label probabilities of the LP step, shape (n, n_labels)."""
        return self.lp_.predict_proba(X)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """Evaluate with ``functions.assess`` (AUC from ``predict_proba``)."""
        return assess(Y, self.predict(X), self.predict_proba(X))
