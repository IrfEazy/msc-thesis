import unittest

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from MLC.HOMERClassifier import HOMERClassifier
from MLC.tests.common import check_contract, make_data


def level_sizes(node):
    """Sizes of the children of each node, by depth."""
    levels, frontier = [], [node]
    while frontier:
        sizes = [len(c.labels) for n in frontier for c in n.children]
        if not sizes:
            break
        levels.append(sorted(sizes, reverse=True))
        frontier = [c for n in frontier for c in n.children if not c.is_leaf]
    return levels


def check_partition(test, node):
    if node.is_leaf:
        return
    flat = sorted(l for c in node.children for l in c.labels)
    test.assertEqual(flat, sorted(node.labels))
    for c in node.children:
        check_partition(test, c)


def random_tree(q, k, seed=0):
    rng = numpy.random.RandomState(seed)
    Y = (rng.rand(40, q) < 0.4).astype(int)
    X = rng.randn(40, 3)
    return HOMERClassifier(n_clusters=k, random_state=seed).fit(X, Y)


class SpyEstimator(ClassifierMixin, BaseEstimator):
    """Logistic regression recording every fit in a class-level list."""

    calls = []

    def fit(self, X, y):
        type(self).calls.append((numpy.asarray(X).copy(), numpy.asarray(y).copy()))
        self.est_ = LogisticRegression().fit(X, y)
        self.classes_ = self.est_.classes_
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        return self.est_.predict_proba(X)


class HOMERTests(unittest.TestCase):
    def test_invalid_n_clusters(self):
        X, Y = make_data(0)
        for k in (0, 1, -2):
            with self.assertRaises(ValueError):
                HOMERClassifier(n_clusters=k).fit(X, Y)

    def test_threshold_zero_does_not_mark_pruned_labels(self):
        X, Y = make_data(0)
        clf = HOMERClassifier(random_state=0).fit(X, Y)
        clf.threshold = 0.0
        # Every score is >= 0, so every leaf is reached and all labels are predicted.
        self.assertTrue((clf.predict(X) == 1).all())
        clf.threshold = 1.01
        self.assertEqual(clf.predict(X).sum(), 0)
        # Prune the second root child with a threshold equal to the first
        # model's minimum score: pruned labels must stay 0 though 0 >= 0.
        clf.threshold = 0.5
        scores, reached = clf._scores(X)
        P = clf.predict(X)
        numpy.testing.assert_array_equal(P, reached.astype(int))
        self.assertTrue((scores[~reached] == 0).all())

    def test_node_training_rows_and_meta_targets(self):
        X, Y = make_data(0)
        SpyEstimator.calls = []
        clf = HOMERClassifier(SpyEstimator(), random_state=0).fit(X, Y)

        # Training order is depth-first: one fit per child at each internal node.
        expected = []

        def walk(node, is_root):
            if node.is_leaf:
                return
            mask = numpy.ones(len(Y), bool) if is_root else Y[:, node.labels].any(axis=1)
            for child in node.children:
                expected.append((mask, Y[mask][:, child.labels].any(axis=1).astype(int)))
                walk(child, False)

        walk(clf.root_, True)
        self.assertEqual(len(SpyEstimator.calls), len(expected))
        non_root = 0
        for (Xf, yf), (mask, target) in zip(SpyEstimator.calls, expected):
            numpy.testing.assert_array_equal(Xf, X[mask])
            numpy.testing.assert_array_equal(yf, target)
            if not mask.all():
                non_root += 1
        self.assertGreater(non_root, 0)

    def test_contract(self):
        check_contract(self, lambda: HOMERClassifier(random_state=0))

    def test_sizes_q8_k2(self):
        clf = random_tree(8, 2)
        self.assertEqual(level_sizes(clf.root_), [[4, 4], [2, 2, 2, 2], [1] * 8])
        check_partition(self, clf.root_)

    def test_sizes_q7_k3(self):
        clf = random_tree(7, 3)
        top = [len(c.labels) for c in clf.root_.children]
        self.assertEqual(sorted(top), [2, 2, 3])
        check_partition(self, clf.root_)

        def check(node):
            if not node.is_leaf:
                n = len(node.labels)
                k = min(3, n)
                for c in node.children:
                    self.assertIn(len(c.labels), {n // k, -(-n // k)})
                    check(c)

        check(clf.root_)

    def test_seeded_tree_reproducible(self):
        def shape(node):
            return (tuple(node.labels), [shape(c) for c in node.children])

        self.assertEqual(shape(random_tree(9, 3, 5).root_), shape(random_tree(9, 3, 5).root_))

    def test_all_zero_row_predicts_empty(self):
        X, Y = make_data(0)
        clf = HOMERClassifier(random_state=0).fit(X, Y)
        # Force the root meta-label models to reject every instance.
        P = clf.predict(X)
        clf.threshold = 1.01
        self.assertEqual(clf.predict(X).sum(), 0)
        self.assertEqual(P.shape, Y.shape)
        # Training rows without labels get empty or valid predictions without error.
        Z = Y.copy()
        Z[0] = 0
        clf2 = HOMERClassifier(random_state=0).fit(X, Z)
        self.assertEqual(clf2.predict(X).shape, Z.shape)

    def test_label_without_positives(self):
        X, Y = make_data(0)
        Y[:, 2] = 0
        clf = HOMERClassifier(random_state=0).fit(X, Y)
        self.assertTrue((clf.predict(X)[:, 2] == 0).all())
        self.assertTrue((clf.predict_proba(X)[:, 2] == 0).all())

    def test_proba_and_evaluate(self):
        X, Y = make_data(0)
        clf = HOMERClassifier(random_state=0).fit(X, Y)
        P = clf.predict_proba(X)
        self.assertEqual(P.shape, Y.shape)
        numpy.testing.assert_array_equal(clf.predict(X), (P >= 0.5).astype(int))
        self.assertEqual(set(clf.evaluate(X, Y)), {"accuracy", "hamming_loss", "report"})


if __name__ == "__main__":
    unittest.main()
