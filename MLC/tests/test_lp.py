import unittest

import numpy

from MLC.LPClassifier import LPClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


def separable(reps=6):
    sets = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]])
    Y = numpy.repeat(sets, reps, axis=0)
    X = Y * 5.0 + numpy.tile(numpy.linspace(0, 0.1, reps), len(sets))[:, None]
    return X, Y


class LPTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: LPClassifier())

    def test_predictions_are_training_labelsets(self):
        X, Y = make_data(0)
        P = LPClassifier().fit(X, Y).predict(make_data(3)[0])
        train = {tuple(r) for r in Y}
        self.assertTrue(all(tuple(r) in train for r in P))

    def test_marginals(self):
        X, Y = make_data(0)
        lp = LPClassifier().fit(X, Y)
        S = lp.predict_proba(X)
        self.assertTrue(((S >= -1e-12) & (S <= 1 + 1e-12)).all())
        proba = lp.classifier_.predict_proba(X)
        for j in range(Y.shape[1]):
            cols = lp.labelsets_[:, j] == 1
            numpy.testing.assert_allclose(S[:, j], proba[:, cols].sum(axis=1))

    def test_sorted_labelsets(self):
        X, Y = make_data(0)
        ls = LPClassifier().fit(X, Y).labelsets_
        self.assertEqual([tuple(r) for r in ls], sorted(tuple(r) for r in ls))

    def test_separable_reproduces_Y(self):
        X, Y = separable()
        numpy.testing.assert_array_equal(LPClassifier().fit(X, Y).predict(X), Y)

    def test_single_labelset(self):
        X, _ = make_data(0)
        Y = numpy.tile([1, 0, 1], (X.shape[0], 1))
        lp = LPClassifier().fit(X, Y)
        numpy.testing.assert_array_equal(lp.predict(X), Y)
        numpy.testing.assert_array_equal(lp.predict_proba(X), Y)

    def test_estimator_without_predict_proba(self):
        from sklearn.svm import LinearSVC

        X, Y = separable()
        lp = LPClassifier(LinearSVC()).fit(X, Y)
        numpy.testing.assert_array_equal(lp.predict(X), Y)
        self.assertTrue(((lp.predict_proba(X) == 0) | (lp.predict_proba(X) == 1)).all())

    def test_fit_trains_on_class_ids(self):
        X, Y = separable()
        RecordingEstimator.reset()
        LPClassifier(RecordingEstimator()).fit(X, Y)
        self.assertEqual(len(RecordingEstimator.calls), 1)
        self.assertEqual(set(RecordingEstimator.calls[0][1]), {0, 1, 2, 3})

    def test_evaluate(self):
        X, Y = separable()
        r = LPClassifier().fit(X, Y).evaluate(X, Y)
        self.assertIsInstance(r, dict)


if __name__ == "__main__":
    unittest.main()
