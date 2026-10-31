import numpy
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClassifierMixin

from ._base import fit_binary, positive_score, resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


class Node:
    """Node of the HOMER hierarchy.

    Attributes
    ----------
    labels : list of int
        Indices of the labels represented by the node.
    children : list of Node
        Child nodes; their label sets partition ``labels``.
    models : list
        One fitted binary classifier per child, predicting the child's
        meta-label (OR of its labels).
    is_leaf : bool
        True for a single-label node (no models).
    """

    def __init__(self, labels):
        self.labels = labels
        self.models = []
        self.children = []
        self.is_leaf = False


class HOMERClassifier(BaseEstimator, ClassifierMixin):
    """HOMER: hierarchy of multi-label learners (Tsoumakas et al. 2008).

    The label set is split recursively into ``n_clusters`` groups by balanced
    k-means over the label indicator columns (each label is a point whose
    coordinates are its values over all training rows). Each internal node
    trains one binary model per child on the child's meta-label (OR of the
    child's labels), using only the training rows that have at least one
    label of the node (all rows at the root). Prediction goes top-down and
    descends into a child only if its meta-label score reaches ``threshold``.

    Parameters
    ----------
    base_estimator : estimator, default None
        Binary classifier cloned for each meta-label. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    n_clusters : int, default 2
        Number of children per node (k of the balanced k-means), at least 2
        (fit raises ValueError otherwise). A node with
        fewer labels is split into single labels. Children have floor(n/k)
        or ceil(n/k) labels.
    threshold : float, default 0.5
        A meta-label score at or above it lets the traversal descend, and a
        leaf score at or above it makes the label relevant.
    random_state : int or None, default None
        Seeds the k-means initialisation; a fixed value gives the same tree.
    n_iter : int, default 10
        Fixed number of k-means iterations.

    Attributes
    ----------
    root_ : Node
        Root of the hierarchy.
    n_labels_ : int
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        n_clusters: int = 2,
        threshold: float = 0.5,
        random_state: int | None = None,
        n_iter: int = 10,
    ):
        self.base_estimator = base_estimator
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.random_state = random_state
        self.n_iter = n_iter

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "HOMERClassifier":
        """
        Build the label hierarchy and train the meta-label models.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary label matrix.

        Returns
        -------
        self : "HOMERClassifier"
        """
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be at least 2, got {self.n_clusters}.")
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        template = resolve_estimator(self.base_estimator)
        rng = numpy.random.RandomState(self.random_state)
        self.n_labels_ = Y.shape[1]
        self.root_ = self._build_tree(
            X, Y, list(range(self.n_labels_)), template, rng, True
        )
        return self

    def _split_labels(self, Y, label_indices, rng):
        """
        Partition a label set into balanced groups (balanced k-means).

        Labels are points (their columns of ``Y`` over all training rows).
        Centres start from k distinct random labels; each of the ``n_iter``
        iterations assigns labels to centres with capacities floor(n/k) or
        ceil(n/k) (optimal assignment, centres replicated per capacity slot)
        and recomputes the centres.

        Returns
        -------
        groups : list of sorted label-index lists
            Disjoint, covering ``label_indices``, ordered by smallest label.
        """
        m = len(label_indices)
        k = min(self.n_clusters, m)
        if k >= m:
            return [[lbl] for lbl in label_indices]
        points = Y[:, label_indices].T.astype(float)
        caps = numpy.full(k, m // k)
        caps[: m % k] += 1
        slots = numpy.repeat(numpy.arange(k), caps)
        centres = points[rng.choice(m, size=k, replace=False)]
        assign = slots[:m]
        for _ in range(self.n_iter):
            dist = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            rows, cols = linear_sum_assignment(dist[:, slots])
            assign = numpy.empty(m, dtype=int)
            assign[rows] = slots[cols]
            centres = numpy.array([points[assign == c].mean(axis=0) for c in range(k)])
        groups = [
            sorted(label_indices[i] for i in range(m) if assign[i] == c)
            for c in range(k)
        ]
        return sorted(groups, key=lambda g: g[0])

    def _build_tree(self, X, Y, label_indices, template, rng, is_root=False):
        """Recursively build the hierarchy (see the class docstring)."""
        node = Node(label_indices)
        if len(label_indices) == 1 and not is_root:
            node.is_leaf = True
            return node
        if is_root:
            mask = numpy.ones(Y.shape[0], dtype=bool)
        else:
            mask = Y[:, label_indices].any(axis=1)
        X_sub, Y_sub = X[mask], Y[mask]
        if len(label_indices) == 1:
            groups = [list(label_indices)]
        else:
            groups = self._split_labels(Y, label_indices, rng)
        for group in groups:
            mu = Y_sub[:, group].any(axis=1).astype(int)
            node.models.append(fit_binary(template, X_sub, mu))
            node.children.append(self._build_tree(X, Y, group, template, rng))
        return node

    def _traverse(self, X, node, active, out, reached_out):
        """Fill ``out`` with pruned top-down scores and ``reached_out`` with
        the reached-leaf mask for the rows in ``active``."""
        for child, model in zip(node.children, node.models):
            idx = numpy.flatnonzero(active)
            if idx.size == 0:
                return
            p = numpy.zeros(X.shape[0])
            p[idx] = positive_score(model, X[idx])
            reached = active & (p >= self.threshold)
            if child.is_leaf:
                out[reached, child.labels[0]] = p[reached]
                reached_out[reached, child.labels[0]] = True
            else:
                self._traverse(X, child, reached, out, reached_out)

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Pruned top-down scores (not calibrated probabilities).

        A label's score is the meta-label score of its leaf, if every
        meta-label on the path from the root reached ``threshold``; otherwise
        0.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_labels)
        """
        return self._scores(X)[0]

    def _scores(self, X):
        """(pruned scores, reached-leaf mask), both (n_samples, n_labels)."""
        X = numpy.asarray(X)
        out = numpy.zeros((X.shape[0], self.n_labels_))
        reached = numpy.zeros(out.shape, dtype=bool)
        self._traverse(X, self.root_, numpy.ones(X.shape[0], dtype=bool), out, reached)
        return out, reached

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict labels: 1 for the labels whose leaf was reached, i.e. every
        meta-label on the path scored at least ``threshold`` (pruned labels
        are never predicted, even with ``threshold=0``).

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
        """
        return self._scores(X)[1].astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate the model; see ``functions.assess`` (AUC uses the scores).
        """
        scores, reached = self._scores(X)
        return assess(Y, reached.astype(int), scores)
