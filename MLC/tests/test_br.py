import unittest

import numpy
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from MLC.BRClassifier import BRClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


class BRTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: BRClassifier())

    def test_columns_match_single_clone(self):
        X, Y = make_data(0)
        est = LogisticRegression(max_iter=500)
        br = BRClassifier(est).fit(X, Y)
        P, S = br.predict(X), br.predict_proba(X)
        for i in range(Y.shape[1]):
            single = clone(est).fit(X, Y[:, i])
            numpy.testing.assert_array_equal(P[:, i], single.predict(X))
            numpy.testing.assert_allclose(S[:, i], single.predict_proba(X)[:, 1])

    def test_constant_column(self):
        X, Y = make_data(0)
        Y[:, 2] = 1
        br = BRClassifier().fit(X, Y)
        self.assertTrue((br.predict(X)[:, 2] == 1).all())
        self.assertTrue((br.predict_proba(X)[:, 2] == 1).all())

    def test_refit_and_recording(self):
        X, Y = make_data(0)
        RecordingEstimator.reset()
        br = BRClassifier(RecordingEstimator()).fit(X, Y)
        self.assertEqual(len(RecordingEstimator.calls), Y.shape[1])
        for i, (Xf, yf) in enumerate(RecordingEstimator.calls):
            numpy.testing.assert_array_equal(Xf, X)
            numpy.testing.assert_array_equal(yf, Y[:, i])
        br.fit(X, Y[:, :2])
        self.assertEqual(len(br.estimators_), 2)
        self.assertEqual(br.n_labels_, 2)

    def test_evaluate(self):
        X, Y = make_data(0)
        r = BRClassifier().fit(X, Y).evaluate(X, Y)
        self.assertEqual(set(r), {"accuracy", "hamming_loss", "report"})
        self.assertIn("auc", r["report"]["0"])

    def test_t_criterion(self):
        X, Y = make_data(0)
        # Rarely positive labels make the default leave empty rows.
        Y = Y.copy()
        Y[:, 1:] = 0
        Y[::10, 1:] = 1
        default = BRClassifier().fit(X, Y)
        P0 = default.predict(X)
        empty = P0.sum(axis=1) == 0
        self.assertTrue(empty.any())
        crit = BRClassifier(t_criterion=True).fit(X, Y)
        P1 = crit.predict(X)
        S = crit.predict_proba(X)
        self.assertTrue((P1.sum(axis=1) >= 1).all())
        # Non-empty rows are unchanged; empty rows get exactly the argmax label.
        numpy.testing.assert_array_equal(P1[~empty], P0[~empty])
        for r in numpy.flatnonzero(empty):
            expected = numpy.zeros(Y.shape[1], dtype=int)
            expected[S[r].argmax()] = 1
            numpy.testing.assert_array_equal(P1[r], expected)


if __name__ == "__main__":
    unittest.main()
