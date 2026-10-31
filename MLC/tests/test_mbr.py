import unittest

import numpy

from MLC.MBRClassifier import MBRClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


def meta_features(**kw):
    """Stage-2 training X of the first label, via a recording meta estimator."""
    X, Y = make_data(0)
    RecordingEstimator.reset()
    clf = MBRClassifier(meta_estimator=RecordingEstimator(), **kw).fit(X, Y)
    return clf, RecordingEstimator.calls[0][0], X, Y


class MBRTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: MBRClassifier(random_state=0))  # None reseeds folds on each fit

    def test_meta_columns(self):
        _, Xm, X, Y = meta_features()
        self.assertEqual(Xm.shape, (X.shape[0], X.shape[1] + Y.shape[1]))
        numpy.testing.assert_array_equal(Xm[:, : X.shape[1]], X)

    def test_oof_differs_from_in_sample_and_reproducible(self):
        _, oof1, _, _ = meta_features(cv=5, random_state=1)
        _, oof2, _, _ = meta_features(cv=5, random_state=1)
        _, ins, _, _ = meta_features(cv=None)
        numpy.testing.assert_array_equal(oof1, oof2)
        self.assertFalse(numpy.allclose(oof1, ins))

    def test_cv_none_in_sample(self):
        clf, Xm, X, _ = meta_features(cv=None)
        F = numpy.column_stack([e.predict_proba(X)[:, 1] for e in clf.first_stage_estimators_])
        numpy.testing.assert_allclose(Xm[:, X.shape[1]:], F)

    def test_constant_label_in_folds(self):
        X, Y = make_data(0)
        Y[:5, 1] = 0
        Y[5:, 1] = 0
        Y[0, 1] = 1  # rare positive: some folds see a single class
        P = MBRClassifier(random_state=0).fit(X, Y).predict(X)
        self.assertEqual(P.shape, Y.shape)

    def test_evaluate(self):
        X, Y = make_data(0)
        r = MBRClassifier(random_state=0).fit(X, Y).evaluate(X, Y)
        self.assertIn("auc", r["report"]["0"])


if __name__ == "__main__":
    unittest.main()
