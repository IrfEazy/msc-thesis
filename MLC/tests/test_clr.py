import unittest

import numpy

from MLC.functions import assess
from sklearn.base import BaseEstimator, ClassifierMixin

from MLC.BRClassifier import BRClassifier
from MLC.CLRClassifier import CLRClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


class StubEstimator(ClassifierMixin, BaseEstimator):
    """Fixed predicted class (value 0/1) for every sample."""

    def __init__(self, value=0):
        self.value = value

    def fit(self, X, y):
        self.classes_ = numpy.array([0, 1])
        return self

    def predict(self, X):
        return numpy.full(len(X), self.value)

    def predict_proba(self, X):
        return numpy.tile([1.0 - self.value, float(self.value)], (len(X), 1))


def build(q, pair_values, aux_values):
    clr = CLRClassifier()
    clr.n_labels_ = q
    clr.pairwise_estimators_ = {k: StubEstimator(v) for k, v in pair_values.items()}
    clr.aux_estimators_ = [StubEstimator(v) for v in aux_values]
    return clr


class CLRTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: CLRClassifier())

    def test_virtual_label_vote_changes_set(self):
        X = numpy.zeros((2, 3))
        # q=3. Pairwise: 0 beats 1, 0 beats 2, 1 beats 2 -> zeta = [2, 1, 0]
        pairs = {(0, 1): 1, (0, 2): 1, (1, 2): 1}
        # aux: label0 relevant, label1 irrelevant, label2 irrelevant
        # zeta* = [3, 1, 0], zeta*(V) = 2 -> only label 0 predicted
        clr = build(3, pairs, [1, 0, 0])
        numpy.testing.assert_array_equal(clr.predict(X), [[1, 0, 0]] * 2)
        # all aux relevant: zeta* = [3, 2, 1], zeta*(V) = 0 -> all predicted
        clr = build(3, pairs, [1, 1, 1])
        numpy.testing.assert_array_equal(clr.predict(X), [[1, 1, 1]] * 2)
        # no aux relevant: zeta*(V) = 3 = zeta*(0) -> strict, nothing predicted
        clr = build(3, pairs, [0, 0, 0])
        numpy.testing.assert_array_equal(clr.predict(X), [[0, 0, 0]] * 2)
        numpy.testing.assert_allclose(clr.predict_proba(X)[0], [2 / 3, 1 / 3, 0])

    def test_vote_conservation(self):
        X, Y = make_data(0)
        clr = CLRClassifier().fit(X, Y)
        q = Y.shape[1]
        self.assertEqual(len(clr.pairwise_estimators_), q * (q - 1) // 2)
        zeta_star, _ = clr._votes(X)
        aux = numpy.column_stack([(e.predict(X) == 1) for e in clr.aux_estimators_])
        self.assertTrue(((zeta_star - aux).sum(axis=1) == q * (q - 1) // 2).all())

    def test_pairwise_training_rows(self):
        X, Y = make_data(0)
        RecordingEstimator.reset()
        CLRClassifier(RecordingEstimator()).fit(X, Y)
        mask = Y[:, 0] != Y[:, 1]
        Xp, yp = RecordingEstimator.calls[0]
        self.assertEqual(Xp.shape[0], mask.sum())
        numpy.testing.assert_array_equal(yp, Y[mask, 0])
        q = Y.shape[1]
        self.assertEqual(len(RecordingEstimator.calls), q * (q - 1) // 2 + q)
        self.assertEqual(RecordingEstimator.calls[-1][0].shape[0], X.shape[0])

    def test_empty_pair_skipped_and_q1(self):
        X, Y = make_data(0)
        Y[:, 1] = Y[:, 0]
        clr = CLRClassifier().fit(X, Y)
        self.assertNotIn((0, 1), clr.pairwise_estimators_)
        self.assertEqual(clr.predict(X).shape, Y.shape)
        clr1 = CLRClassifier().fit(X, Y[:, :1])
        P = clr1.predict(X)
        self.assertEqual(P.shape, (X.shape[0], 1))
        numpy.testing.assert_array_equal(P[:, 0], clr1.aux_estimators_[0].predict(X))

    def test_cardinality_not_overpredicted(self):
        X, Y = make_data(0, n_samples=300, n_features=10, n_classes=5)
        clr = CLRClassifier().fit(X, Y)
        P = clr.predict(X)
        ratio = P.sum(axis=1).mean() / Y.sum(axis=1).mean()
        self.assertLess(abs(ratio - 1), 0.3)
        from sklearn.metrics import f1_score
        f_clr = f1_score(Y, P, average="micro")
        f_br = f1_score(Y, BRClassifier().fit(X, Y).predict(X), average="micro")
        self.assertLess(abs(f_clr - f_br), 0.15)

    def test_evaluate_computes_votes_once(self):
        X, Y = make_data(0)
        clf = CLRClassifier().fit(X, Y)
        calls = []
        original = clf._votes
        clf._votes = lambda X: calls.append(1) or original(X)
        r = clf.evaluate(X, Y)
        self.assertEqual(len(calls), 1)
        self.assertEqual(r["accuracy"], assess(Y, clf.predict(X), clf.predict_proba(X))["accuracy"])

    def test_evaluate(self):
        X, Y = make_data(0)
        r = CLRClassifier().fit(X, Y).evaluate(X, Y)
        self.assertIn("auc", r["report"]["0"])


if __name__ == "__main__":
    unittest.main()
