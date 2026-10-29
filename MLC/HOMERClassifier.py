from typing import List, Optional

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


# Helper Node class to represent a node in the hierarchy.
class Node:
    def __init__(self, labels: List[int]):
        """
        Initialize a node in the hierarchy.

        Parameters
        ----------
        labels : List[int]
            The indices of the labels represented by this node.

        Attributes
        ----------
        children : List[Node]
            Child nodes, whose label sets partition `labels`.
        models : list
            One entry per child: a fitted binary classifier of the child's
            meta-label, or a constant probability (float).
        """
        self.labels = labels
        self.models = []
        self.children = []
        self.is_leaf = False


class HOMERClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: ClassifierMixin = LogisticRegression(max_iter=1000),
        n_clusters: int = 2,
        threshold: float = 0.5,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the HOMER classifier.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The base classifier cloned for each meta-label model.
        n_clusters : int
            The maximum number of clusters to use for label set splitting.
        threshold : float
            The probability threshold for a label to be considered relevant.
        random_state : Optional[int]
            The random seed to use for reproducibility.
        """
        self.root = None
        self.n_labels_ = None
        self.X_ = None
        self.Y_ = None
        self.base_estimator = base_estimator
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.random_state = random_state

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "HOMERClassifier":
        """
        Build the label hierarchy and train the child meta-label models of each node.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary label matrix.

        Returns
        -------
        self : "HOMERClassifier"
            The fitted HOMER classifier.
        """
        self.Y_ = numpy.array(Y)
        self.X_ = X
        self.n_labels_ = self.Y_.shape[1]
        # Build the tree starting from the full label set.
        self.root = self._build_tree(X, self.Y_, list(range(self.n_labels_)), True)
        return self

    def _fit_meta_model(self, X: ArrayLike, mu: ArrayLike):
        """
        Fit a binary model for a meta-label, or return a constant probability.

        Parameters
        ----------
        X : ArrayLike of shape (n_subset, n_features)
            Training samples of the node.
        mu : ArrayLike of shape (n_subset,)
            Binary meta-label values.

        Returns
        -------
        model : float or fitted estimator
            0.0 for an empty subset or an all-zero meta-label, 1.0 for an
            all-one meta-label, otherwise a fitted clone of the base estimator.
        """
        if mu.shape[0] == 0 or not mu.any():
            return 0.0
        if mu.all():
            return 1.0
        return clone(self.base_estimator).fit(X, mu)

    @staticmethod
    def _model_proba(model, x: ArrayLike) -> float:
        """
        Probability of the positive meta-label for a single instance.

        Parameters
        ----------
        model : float or fitted estimator
            A constant probability or a fitted binary classifier.
        x : ArrayLike of shape (1, n_features)
            The input feature vector.

        Returns
        -------
        p : float
            Probability of class 1.
        """
        if isinstance(model, float):
            return model
        column = list(model.classes_).index(1)
        return float(model.predict_proba(x)[0][column])

    def _split_labels(self, Y: ArrayLike, label_indices: List[int]) -> List[List[int]]:
        """
        Partition a label set into child label sets.

        Parameters
        ----------
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary label matrix.
        label_indices : List[int]
            The indices of the labels to split (at least two).

        Returns
        -------
        groups : List[List[int]]
            Non-empty, disjoint groups covering label_indices.
        """
        n_labels_node = len(label_indices)
        n_clusters = min(self.n_clusters, n_labels_node)
        if n_labels_node == n_clusters:
            return [[lbl] for lbl in label_indices]
        # Represent each label by its column vector (over all training samples)
        # and cluster them into n_clusters groups.
        label_matrix = Y[:, label_indices].T
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(label_matrix)
        groups = [
            [label_indices[i] for i in range(n_labels_node) if clusters[i] == cl]
            for cl in numpy.unique(clusters)
        ]
        if len(groups) < 2:
            # Degenerate clustering (e.g. identical columns): even contiguous split.
            groups = [
                [label_indices[i] for i in part]
                for part in numpy.array_split(numpy.arange(n_labels_node), n_clusters)
            ]
        return groups

    def _build_tree(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        label_indices: List[int],
        is_root: bool = False,
    ) -> Node:
        """
        Recursively build the hierarchy.

        Each internal node trains one binary model per child on the child's
        meta-label (the OR of Y over the child's labels), using the samples
        with at least one positive label in the node (all samples at the root).
        Leaf nodes train no model.

        Parameters
        ----------
        X : ArrayLike
            Input feature matrix.
        Y : ArrayLike
            Binary label matrix.
        label_indices : List[int]
            The indices of the labels to consider for this node.
        is_root : bool
            Whether the node is the root (trained on all samples).

        Returns
        -------
        node : Node
            The node representing the current label set.
        """
        node = Node(label_indices)
        if len(label_indices) == 1 and not is_root:
            node.is_leaf = True
            return node

        if is_root:
            mask = numpy.ones(Y.shape[0], dtype=bool)
        else:
            mask = numpy.any(Y[:, label_indices] == 1, axis=1)
        X_sub = numpy.array(X)[mask]
        Y_sub = Y[mask]

        if len(label_indices) == 1:
            groups = [list(label_indices)]
        else:
            groups = self._split_labels(Y, label_indices)
        for group in groups:
            mu = numpy.any(Y_sub[:, group] == 1, axis=1).astype(int)
            node.models.append(self._fit_meta_model(X_sub, mu))
            node.children.append(self._build_tree(X, Y, group))
        return node

    def _traverse_predict(self, x: ArrayLike, node: Node) -> ArrayLike:
        """
        Recursively traverse the hierarchy for a single instance to obtain label probabilities.

        Parameters
        ----------
        x : ArrayLike of shape (n_features,)
            The input feature vector.
        node : Node
            The current node in the hierarchy.

        Returns
        -------
        prob : ArrayLike of shape (n_labels,)
            Predicted probabilities for each label in the global label space.
            A label has the probability of its leaf at its parent if the
            traversal reached it, otherwise 0.0.
        """
        prob = numpy.zeros(self.n_labels_)
        x_reshaped = x.reshape(1, -1)
        for child, model in zip(node.children, node.models):
            p = self._model_proba(model, x_reshaped)
            if p < self.threshold:
                continue
            if child.is_leaf:
                prob[child.labels[0]] = p
            else:
                # Child label sets are disjoint, so the results do not overlap.
                prob = numpy.maximum(prob, self._traverse_predict(x, child))
        return prob

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict marginal probabilities for each label.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        probabilities : ArrayLike of shape (n_samples, n_labels)
            Predicted probabilities for each label in the global label space.
        """
        n_samples = X.shape[0]
        probabilities = numpy.zeros((n_samples, self.n_labels_))
        for i in range(n_samples):
            probabilities[i] = self._traverse_predict(X[i], self.root)
        return probabilities

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict binary labels for each instance.

        A label is predicted as relevant if its predicted probability is at least the threshold.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        Y_pred : ArrayLike of shape (n_samples, n_labels)
            Predicted binary labels for each instance.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate the HOMER classifier using common multi-label metrics.

        Metrics include:
          - Subset Accuracy
          - Hamming Loss
          - Micro-averaged F1 Score

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input feature matrix.
        Y : ArrayLike of shape (n_samples, n_labels)
            The binary label matrix.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary containing the computed evaluation
            metrics (accuracy, hamming loss, and F1 score).
        """
        return assess(Y, self.predict(X))
