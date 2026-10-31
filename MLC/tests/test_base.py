import unittest

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from MLC._base import (
    ConstantBinaryClassifier,
    binary_predict,
    fit_binary,
    positive_score,
    resolve_estimator,
)
from MLC.functions import assess
from MLC.tests.common import make_data


class HelperTests(unittest.TestCase):
    def test_resolve_estimator(self):
        a, b = resolve_estimator(None), resolve_estimator(None)
        self.assertIsNot(a, b)
        self.assertEqual(a.max_iter, 1000)
        est = LogisticRegression()
        self.assertIs(resolve_estimator(est), est)

    def test_constant_classifier(self):
        X = numpy.zeros((3, 2))
        c = ConstantBinaryClassifier(1).fit(X, None)
        self.assertEqual(list(c.classes_), [0, 1])
        self.assertEqual(c.predict(X).tolist(), [1, 1, 1])
        numpy.testing.assert_array_equal(c.predict_proba(X), [[0, 1]] * 3)

    def test_fit_binary(self):
        X, Y = make_data(0)
        tmpl = LogisticRegression()
        m = fit_binary(tmpl, X, Y[:, 0])
        self.assertIsNot(m, tmpl)
        self.assertFalse(hasattr(tmpl, "coef_"))
        c = fit_binary(tmpl, X, numpy.ones(len(X), dtype=int))
        self.assertIsInstance(c, ConstantBinaryClassifier)
        self.assertEqual(c.value, 1)

    def test_positive_score(self):
        X, Y = make_data(0)
        lr = fit_binary(LogisticRegression(), X, Y[:, 0])
        numpy.testing.assert_allclose(positive_score(lr, X), lr.predict_proba(X)[:, 1])
        svm = fit_binary(LinearSVC(), X, Y[:, 0])
        numpy.testing.assert_allclose(positive_score(svm, X), svm.decision_function(X))
        self.assertEqual(positive_score(ConstantBinaryClassifier(0), X).tolist(), [0.0] * len(X))

    def test_positive_score_class_order_and_missing_class(self):
        X = numpy.zeros((3, 2))

        class Reversed:
            classes_ = numpy.array([1, 0])

            def predict_proba(self, X):
                return numpy.tile([0.8, 0.2], (len(X), 1))

        class OnlyZero:
            classes_ = numpy.array([0])

            def predict_proba(self, X):
                return numpy.ones((len(X), 1))

        self.assertEqual(positive_score(Reversed(), X).tolist(), [0.8] * 3)
        self.assertEqual(positive_score(OnlyZero(), X).tolist(), [0.0] * 3)

    def test_binary_predict(self):
        X = numpy.zeros((2, 2))

        class Proba:
            classes_ = numpy.array([0, 1])

            def predict_proba(self, X):
                return numpy.array([[0.4, 0.6], [0.5, 0.5]])

        class Hard:
            def predict(self, X):
                return numpy.array([1, 0])

        self.assertEqual(binary_predict(Proba(), X).tolist(), [1, 0])
        self.assertEqual(binary_predict(Hard(), X).tolist(), [1, 0])


class AssessTests(unittest.TestCase):
    def setUp(self):
        rng = numpy.random.RandomState(0)
        self.Y = rng.randint(0, 2, (30, 4))
        self.P = rng.randint(0, 2, (30, 4))

    def test_accuracy_and_hamming(self):
        r = assess(self.Y, self.P)
        self.assertAlmostEqual(r["accuracy"], (self.Y == self.P).all(axis=1).mean())
        self.assertAlmostEqual(r["hamming_loss"], (self.Y != self.P).mean())

    def test_single_class_column(self):
        Y = self.Y.copy()
        Y[:, 1] = 0
        r = assess(Y, self.P, numpy.random.RandomState(1).rand(30, 4))
        self.assertTrue(numpy.isnan(r["report"]["1"]["auc"]))
        self.assertTrue(numpy.isfinite(r["report"]["macro avg"]["auc"]))
        self.assertTrue(numpy.isfinite(r["report"]["weighted avg"]["auc"]))

    def test_all_single_class(self):
        r = assess(numpy.zeros((5, 2), int), numpy.zeros((5, 2), int))
        self.assertTrue(numpy.isnan(r["report"]["macro avg"]["auc"]))
        self.assertTrue(numpy.isnan(r["report"]["samples avg"]["auc"]))

    def test_hand_computed_auc(self):
        Y = numpy.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        P = numpy.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        S = numpy.array([[0.9, 0.1], [0.4, 0.8], [0.6, 0.7], [0.1, 0.2]])
        rep = assess(Y, P, S)["report"]
        # col 0: positives {0.9, 0.4}, negatives {0.6, 0.1} -> 3/4 pairs ordered.
        self.assertAlmostEqual(rep["0"]["auc"], 0.75)
        # col 1: positives {0.8, 0.7}, negatives {0.1, 0.2} -> 1.0.
        self.assertAlmostEqual(rep["1"]["auc"], 1.0)
        self.assertAlmostEqual(rep["macro avg"]["auc"], 0.875)
        self.assertAlmostEqual(rep["weighted avg"]["auc"], 0.875)  # supports 2 and 2
        # micro: positives {0.9, 0.4, 0.8, 0.7}, negatives {0.1, 0.6, 0.1, 0.2}.
        # Ordered pairs: 0.9 beats 4, 0.4 beats 3, 0.8 beats 4, 0.7 beats 4 -> 15/16.
        self.assertAlmostEqual(rep["micro avg"]["auc"], 15 / 16)
        # samples: rows 0 and 2 have AUC 1 (0.9 > 0.1, 0.7 > 0.6); rows 1 and 3
        # have a single class and are skipped.
        self.assertAlmostEqual(rep["samples avg"]["auc"], 1.0)

    def test_score_auc_differs_from_hard(self):
        Y = numpy.array([[1], [1], [0], [0]])
        P = numpy.array([[1], [0], [0], [0]])
        S = numpy.array([[0.9], [0.6], [0.4], [0.1]])
        hard = assess(Y, P)["report"]["0"]["auc"]
        soft = assess(Y, P, S)["report"]["0"]["auc"]
        self.assertAlmostEqual(soft, 1.0)
        self.assertAlmostEqual(hard, 0.75)
        for k in ("micro avg", "macro avg", "weighted avg", "samples avg"):
            self.assertIn("auc", assess(Y, P, S)["report"][k])


if __name__ == "__main__":
    unittest.main()
