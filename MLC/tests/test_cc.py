import unittest

import numpy
from sklearn.linear_model import LogisticRegression

from MLC.BRClassifier import BRClassifier
from MLC.CCClassifier import CCClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


class CCTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: CCClassifier())

    def test_link_trained_on_true_labels(self):
        X, Y = make_data(0)
        RecordingEstimator.reset()
        CCClassifier(RecordingEstimator()).fit(X, Y)
        calls = RecordingEstimator.calls
        self.assertEqual(len(calls), Y.shape[1])
        self.assertEqual(calls[0][0].shape[1], X.shape[1])
        numpy.testing.assert_array_equal(calls[1][0][:, -1], Y[:, 0])
        numpy.testing.assert_array_equal(calls[2][0][:, -2:], Y[:, :2])

    def test_custom_order(self):
        X, Y = make_data(0)
        order = [2, 0, 3, 1]
        cc = CCClassifier(order=order).fit(X, Y)
        P, S = cc.predict(X), cc.predict_proba(X)
        self.assertEqual(P.shape, Y.shape)
        br = BRClassifier().fit(X, Y)
        numpy.testing.assert_array_equal(P[:, 2], br.predict(X)[:, 2])
        numpy.testing.assert_allclose(S[:, 2], br.predict_proba(X)[:, 2])
        # chained label 0 sees label 2's prediction: differs in training features
        RecordingEstimator.reset()
        CCClassifier(RecordingEstimator(), order=order).fit(X, Y)
        numpy.testing.assert_array_equal(RecordingEstimator.calls[1][0][:, -1], Y[:, 2])

    def test_invalid_order(self):
        X, Y = make_data(0)
        for bad in ([0, 1, 2], [0, 0, 1, 2], "foo"):
            with self.assertRaises(ValueError):
                CCClassifier(order=bad).fit(X, Y)

    def test_refit_keeps_length(self):
        X, Y = make_data(0)
        cc = CCClassifier().fit(X, Y)
        cc.fit(X, Y)
        self.assertEqual(len(cc.estimators_), Y.shape[1])

    def test_random_order_reproducible(self):
        X, Y = make_data(0)
        a = CCClassifier(order="random", random_state=3).fit(X, Y)
        b = CCClassifier(order="random", random_state=3).fit(X, Y)
        numpy.testing.assert_array_equal(a.order_, b.order_)
        numpy.testing.assert_array_equal(a.predict(X), b.predict(X))
        self.assertEqual(sorted(a.order_), list(range(Y.shape[1])))

    def test_evaluate(self):
        X, Y = make_data(0)
        r = CCClassifier(LogisticRegression(max_iter=500)).fit(X, Y).evaluate(X, Y)
        self.assertIn("auc", r["report"]["0"])


if __name__ == "__main__":
    unittest.main()
