import unittest

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from MLC._base import (
    ConstantBinaryClassifier,
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
        only0 = LogisticRegression().fit(X[:2], [0, 1])
        only0.classes_ = numpy.array([0])
        only0.predict_proba = lambda X: numpy.ones((len(X), 1))
        self.assertEqual(positive_score(only0, X).tolist(), [0.0] * len(X))


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
