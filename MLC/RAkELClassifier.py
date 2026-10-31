import warnings
from itertools import combinations
from math import comb

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin

from .LPClassifier import LPClassifier
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class RAkELClassifier(BaseEstimator, ClassifierMixin):
    """Random k-Labelsets, disjoint-free variant RAkELo (Zhang & Zhou 2014,
    Fig. 6, Eq. 18-20): an ensemble of LP classifiers, each trained on a
    random k-labelset; label j is relevant when mu_j / tau_j > threshold,
    where tau_j counts the members whose labelset contains j and mu_j the
    members that predict it.

    Parameters
    ----------
    base_estimator : estimator, default None
        Multi-class classifier of each LP member (None: LogisticRegression).
    k : int, default 3
        Labelset size; clipped to the number of labels with a warning.
    n_estimators : int or None, default None
        Number of members m; None means 2 * n_labels. Capped at the number of
        distinct k-labelsets.
    random_state : int or None
        Seed for labelset sampling.

    Attributes
    ----------
    k_ : effective labelset size min(k, n_labels)
    n_estimators_ : effective number of members
    labelsets_ : list of sorted tuples, the distinct label subsets
    estimators_ : list of fitted LPClassifier, one per labelset
    n_labels_ : int

    Notes
    -----
    Every label is covered first (shuffled permutation split into k_-groups,
    the last group topped up with random other labels), then further distinct
    labelsets are drawn at random. If ``n_estimators`` is smaller than the
    number of groups needed, some labels stay uncovered (a warning is issued)
    and are never predicted. ``predict_proba`` returns mu/tau vote fractions,
    not calibrated probabilities.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        k: int = 3,
        n_estimators: int | None = None,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _sample_labelsets(self, q, k, m, rng):
        chosen, seen = [], set()

        def add(s):
            s = tuple(sorted(int(v) for v in s))
            if s not in seen:
                seen.add(s)
                chosen.append(s)

        perm = rng.permutation(q)
        for start in range(0, q, k):
            if len(chosen) >= m:
                break
            group = list(perm[start : start + k])
            if len(group) < k:
                others = [v for v in range(q) if v not in group]
                group += list(rng.choice(others, size=k - len(group), replace=False))
            add(group)
        if comb(q, k) <= 20000:
            rest = [s for s in combinations(range(q), k) if s not in seen]
            for i in rng.permutation(len(rest)):
                if len(chosen) >= m:
                    break
                add(rest[i])
        else:
            while len(chosen) < m:
                add(rng.choice(q, size=k, replace=False))
        return chosen[:m]

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "RAkELClassifier":
        """
        Sample distinct k-labelsets and fit one LP classifier on each.

        Returns
        -------
        self : "RAkELClassifier"
        """
        if self.k < 1:
            raise ValueError("k must be >= 1")
        X = numpy.asarray(X)
        Y = numpy.asarray(Y).astype(int)
        q = Y.shape[1]
        k = min(self.k, q)
        if k < self.k:
            warnings.warn(f"k={self.k} exceeds the number of labels; using k={k}", stacklevel=4)
        m = self.n_estimators if self.n_estimators is not None else 2 * q
        if m < 1:
            raise ValueError("n_estimators must be >= 1")
        m = min(m, comb(q, k))
        rng = numpy.random.RandomState(self.random_state)
        labelsets = self._sample_labelsets(q, k, m, rng)
        if len(set(v for s in labelsets for v in s)) < q:
            warnings.warn("n_estimators is too small to cover every label")
        self.estimators_ = [
            LPClassifier(self.base_estimator).fit(X, Y[:, list(s)]) for s in labelsets
        ]
        self.labelsets_ = labelsets
        self.k_ = k
        self.n_estimators_ = len(labelsets)
        self.n_labels_ = q
        return self

    def _vote_fractions(self, X):
        X = numpy.asarray(X)
        mu = numpy.zeros((X.shape[0], self.n_labels_))
        tau = numpy.zeros_like(mu)
        for s, est in zip(self.labelsets_, self.estimators_):
            idx = list(s)
            mu[:, idx] += est.predict(X)
            tau[:, idx] += 1
        return numpy.divide(mu, tau, out=numpy.zeros_like(mu), where=tau > 0)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> numpy.ndarray:
        """
        Label j is predicted when mu_j / tau_j > threshold (strictly).

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
        """
        return (self._vote_fractions(X) > threshold).astype(int)

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """Vote fractions mu/tau per label (0 for an uncovered label); these
        are not calibrated probabilities."""
        return self._vote_fractions(X)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """Evaluate with ``functions.assess`` (AUC from the vote fractions)."""
        return assess(Y, self.predict(X), self.predict_proba(X))
