from typing import List, Optional, cast

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.multioutput import MultiOutputClassifier
from tqdm.notebook import tqdm
from typing_extensions import TypeVar

from MLC.preconditions import check_same_rows, check_binary_matrices


# Helper Node class to represent a node in the hierarchy.
class Node:
    def __init__(self, labels: List[int]):
        """
        Parameters
        ----------
        labels : List[int]
            The indices (with respect to the global label space) that are assigned to this node.
        """
        self.labels = labels  # list of label indices in this node
        self.clf = None  # classifier trained on samples for these labels
        self.children = []  # child nodes (if any)
        self.is_leaf = False  # True if this node contains a single label


class HOMERClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            base_estimator: ClassifierMixin = LogisticRegression(max_iter=1000),
            n_clusters: int = 2,
            threshold: float = 0.5,
            random_state: Optional[int] = None
    ):
        """
        Initialize the HOMER classifier.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The base learner to be used at each node.
        n_clusters : int
            The number of children to split into at each internal node.
        threshold : float
            Decision threshold for determining label relevance.
        random_state : Optional[int]
            Seed for random number generation.
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
        Build the label hierarchy and train a classifier at each node.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.
        Y : ArrayLike of shape (n_samples, n_labels)
            Binary indicator matrix for labels.

        Returns
        -------
        self : "HOMERClassifier"
            Fitted HOMER classifier.
        """
        self.Y_ = numpy.array(Y)
        self.X_ = X
        self.n_labels_ = self.Y_.shape[1]
        # Build the tree starting from the full label set.
        self.root = self._build_tree(X, self.Y_, list(range(self.n_labels_)))
        return self

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def _build_tree(self, X: ArrayLike, Y: ArrayLike, label_indices: List[int]) -> Node:
        """
        Recursively build the hierarchy.

        Parameters
        ----------
        X : ArrayLike
            Training features.
        Y : ArrayLike
            Training labels.
        label_indices : List[int]
            The indices of the labels (from the global label space) to be handled at this node.

        Returns
        -------
        node : Node
            The constructed node (with classifier and child nodes, if any).
        """
        node = Node(label_indices)
        # Select samples having at least one positive label among those in this node.
        mask = numpy.any(Y[:, label_indices] == 1, axis=1)
        X_sub = numpy.array(X[mask])
        Y_sub = numpy.array(Y[mask][:, label_indices])
        T = TypeVar("T", bound=ClassifierMixin)

        # Train a classifier for this node if there are samples.
        if X_sub.shape[0] > 0:
            if len(label_indices) == 1:
                clf: T = cast(T, clone(self.base_estimator))
                clf.fit(X_sub, Y_sub.ravel())
                node.clf = clf
            else:
                clf = MultiOutputClassifier(clone(self.base_estimator))
                clf.fit(X_sub, Y_sub)
                node.clf = clf
        else:
            node.clf = None

        # If only one label is present, mark as leaf.
        if len(label_indices) == 1:
            node.is_leaf = True
            return node

        # Otherwise, split the label set using clustering.
        n_labels_node = len(label_indices)
        n_clusters = min(self.n_clusters, n_labels_node)
        if n_labels_node == n_clusters:
            # Each label becomes its own child.
            for lbl in label_indices:
                child = self._build_tree(X, Y, [lbl])
                node.children.append(child)
        else:
            # Represent each label by its column vector (over all training samples)
            # and cluster them into n_clusters groups.
            label_matrix = Y[:, label_indices].T  # shape: (n_labels_node, n_samples)
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            clusters = kmeans.fit_predict(label_matrix)
            for cl in numpy.unique(clusters):
                child_labels = [label_indices[i] for i in range(n_labels_node) if clusters[i] == cl]
                child = self._build_tree(X, Y, child_labels)
                node.children.append(child)
        node.is_leaf = False
        return node

    def _traverse_predict(self, x: ArrayLike, node: Node) -> ArrayLike:
        """
        Recursively traverse the hierarchy for a single instance to obtain label probabilities.

        Parameters
        ----------
        x : ArrayLike of shape (n_features,)
            A single input instance.
        node : Node
            The current node in the hierarchy.

        Returns
        -------
        prob : ArrayLike of shape (n_labels,)
            A vector of predicted probabilities (in the global label space) for instance x.
        """
        prob = numpy.zeros(self.n_labels_)

        # Get this node's prediction (if a classifier was trained).
        if node.clf is not None:
            x_reshaped = x.reshape(1, -1)
            if len(node.labels) == 1:
                # For a single-label node, obtain probability for positive class.
                p = node.clf.predict_proba(x_reshaped)[0][0]
                prob[node.labels[0]] = p
            else:
                # For a multi-label node, node.clf is a MultiOutputClassifier.
                proba_list = node.clf.predict_proba(x_reshaped)
                for i, lbl in enumerate(node.labels):
                    p = proba_list[i][0][1]
                    prob[lbl] = p

        # Traverse children if available.
        if not node.is_leaf and node.children:
            for child in node.children:
                # Proceed to child if any label in its set has a probability above the threshold.
                if any(prob[lbl] >= self.threshold for lbl in child.labels):
                    child_prob = self._traverse_predict(x, child)
                    # Combine probabilities by taking the maximum value.
                    prob = numpy.maximum(prob, child_prob)
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
        Y_proba : ArrayLike of shape (n_samples, n_labels)
            Predicted probability estimates for each label.
        """
        n_samples = X.shape[0]
        Y_proba = numpy.zeros((n_samples, self.n_labels_))
        for i in tqdm(range(n_samples), desc="Predicting for each sample"):
            Y_proba[i] = self._traverse_predict(X[i], self.root)
        return Y_proba

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict binary labels for each instance.

        A label is predicted as relevant if its predicted probability exceeds the threshold.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        Y_pred : ArrayLike of shape (n_samples, n_labels)
            The binary label matrix.
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y_true: ArrayLike) -> dict[str, float]:
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
        Y_true : ArrayLike of shape (n_samples, n_labels)
            The true binary label matrix.

        Returns
        -------
        metrics : dict[str, float]
            A dictionary with keys 'accuracy', 'hamming_loss', and 'f1_micro'.
        """
        Y_pred = self.predict(X)
        acc = accuracy_score(Y_true, Y_pred)
        hamming = hamming_loss(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average='micro')
        return {"accuracy": acc, "hamming_loss": hamming, "f1_micro": f1}
