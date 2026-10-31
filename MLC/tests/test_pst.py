import unittest

import numpy

from MLC.LPClassifier import LPClassifier
from MLC.PStClassifier import PStClassifier
from MLC.tests.common import RecordingEstimator, check_contract, make_data


def pruning_fixture():
    sets = {(1, 0, 0): 5, (0, 1, 0): 3, (0, 0, 1): 5, (1, 1, 0): 1}
    rows, Y = [], []
    for s, n in sets.items():
        for i in range(n):
            Y.append(s)
            rows.append(numpy.array(s, float) * 5 + i * 0.01)
    return numpy.array(rows), numpy.array(Y)


class PStTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: PStClassifier(pruning_value=1))

    def test_defaults(self):
        c = PStClassifier()
        self.assertEqual((c.pruning_value, c.max_reintroduced, c.base_estimator), (2, 1, None))

    def test_equals_lp_without_pruning(self):
        X, Y = make_data(0)
        P = PStClassifier(pruning_value=1).fit(X, Y).predict(X)
        numpy.testing.assert_array_equal(P, LPClassifier().fit(X, Y).predict(X))

    def test_rare_labelset_relabelled_and_never_predicted(self):
        X, Y = pruning_fixture()
        RecordingEstimator.reset()
        pst = PStClassifier(RecordingEstimator(), pruning_value=2).fit(X, Y)
        Xr, yr = RecordingEstimator.calls[0]
        self.assertEqual(Xr.shape[0], X.shape[0])  # rare row reintroduced once
        self.assertNotIn((1, 1, 0), {tuple(r) for r in pst.lp_.labelsets_})
        self.assertNotIn((1, 1, 0), {tuple(r) for r in pst.predict(X)})
        rare = numpy.flatnonzero((Y == [1, 1, 0]).all(axis=1))[0]
        cls = yr[(Xr == X[rare]).all(axis=1)]
        self.assertEqual(tuple(pst.lp_.labelsets_[cls[0]]), (1, 0, 0))

    def test_b1_invariant_to_permutation(self):
        X, Y = pruning_fixture()
        perm = numpy.random.RandomState(0).permutation(len(Y))
        out = []
        for idx in (numpy.arange(len(Y)), perm):
            RecordingEstimator.reset()
            pst = PStClassifier(RecordingEstimator(), pruning_value=2).fit(X[idx], Y[idx])
            Xr, yr = RecordingEstimator.calls[0]
            out.append(sorted((tuple(x), tuple(pst.lp_.labelsets_[c])) for x, c in zip(Xr, yr)))
        self.assertEqual(out[0], out[1])

    def test_b_larger_keeps_more(self):
        X, Y = pruning_fixture()
        Y2 = Y.copy()
        Y2[-1] = (1, 1, 1)  # rare row with three frequent subsets
        RecordingEstimator.reset()
        PStClassifier(RecordingEstimator(), 2, 1).fit(X, Y2)
        n1 = RecordingEstimator.calls[0][0].shape[0]
        RecordingEstimator.reset()
        PStClassifier(RecordingEstimator(), 2, 3).fit(X, Y2)
        n3 = RecordingEstimator.calls[0][0].shape[0]
        self.assertEqual(n3 - n1, 2)

    def test_no_survivors(self):
        X, Y = make_data(0)
        with self.assertRaisesRegex(ValueError, "lower pruning_value"):
            PStClassifier(pruning_value=10**6).fit(X, Y)

    def test_evaluate(self):
        X, Y = pruning_fixture()
        self.assertIsInstance(PStClassifier().fit(X, Y).evaluate(X, Y), dict)


if __name__ == "__main__":
    unittest.main()
