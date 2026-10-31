import unittest
import warnings
from math import comb

import numpy

from MLC.RAkELClassifier import RAkELClassifier
from MLC.tests.common import check_contract, make_data


class _Fixed:
    def __init__(self, out):
        self.out = numpy.asarray(out)

    def predict(self, X):
        return self.out


class RAkELTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: RAkELClassifier(random_state=0))

    def test_small_q(self):
        for q in (2, 3):
            X, Y = make_data(0, n_classes=q)
            r = RAkELClassifier(random_state=0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r.fit(X, Y)
            self.assertEqual(r.k_, min(3, q))
            self.assertEqual(r.n_estimators_, comb(q, r.k_))
            self.assertEqual(r.predict(X).shape, Y.shape)
            self.assertEqual(any("exceeds" in str(x.message) for x in w), q < 3)

    def test_k_warning_points_at_caller(self):
        X, Y = make_data(0, n_classes=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RAkELClassifier(k=5, random_state=0).fit(X, Y)
        clip = [x for x in w if "exceeds the number of labels" in str(x.message)]
        self.assertEqual(len(clip), 1)
        self.assertEqual(clip[0].filename, __file__)

    def test_labelsets_distinct_sized_covering(self):
        X, Y = make_data(0, n_classes=7)
        r = RAkELClassifier(k=3, n_estimators=10, random_state=1).fit(X, Y)
        self.assertEqual(len(set(r.labelsets_)), 10)
        self.assertTrue(all(len(s) == 3 and list(s) == sorted(s) for s in r.labelsets_))
        self.assertEqual({v for s in r.labelsets_ for v in s}, set(range(7)))

    def test_default_ensemble_size_and_cap(self):
        X, Y = make_data(0, n_classes=4)
        self.assertEqual(RAkELClassifier(random_state=0).fit(X, Y).n_estimators_, 4)  # C(4,3)
        X, Y = make_data(0, n_classes=5)
        self.assertEqual(RAkELClassifier(random_state=0).fit(X, Y).n_estimators_, 10)

    def test_params_unchanged(self):
        X, Y = make_data(0, n_classes=5)
        r = RAkELClassifier(k=7, random_state=0)
        before = r.get_params()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.fit(X, Y)
        self.assertEqual(r.get_params(), before)
        self.assertIsNone(r.n_estimators)
        self.assertEqual(r.k_, 5)

    def test_seeded_reproducibility(self):
        X, Y = make_data(0, n_classes=6)
        a = RAkELClassifier(random_state=3).fit(X, Y)
        b = RAkELClassifier(random_state=3).fit(X, Y)
        self.assertEqual(a.labelsets_, b.labelsets_)
        numpy.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_vote_fractions(self):
        r = RAkELClassifier()
        r.n_labels_ = 4
        r.labelsets_ = [(0, 1), (1, 2), (1, 3)]
        r.estimators_ = [
            _Fixed([[1, 1], [0, 0]]),
            _Fixed([[1, 0], [0, 1]]),
            _Fixed([[0, 0], [1, 1]]),
        ]
        X = numpy.zeros((2, 1))
        expected = numpy.array([[1, 2 / 3, 0, 0], [0, 1 / 3, 1, 1]])
        expected[0, 2] = 0.0
        # label 2 only in member 1: sample 0 -> 0, sample 1 -> 1; label 3 only in member 2
        expected = numpy.array([[1.0, 2 / 3, 0.0, 0.0], [0.0, 1 / 3, 1.0, 1.0]])
        numpy.testing.assert_allclose(r.predict_proba(X), expected)
        numpy.testing.assert_array_equal(r.predict(X), [[1, 1, 0, 0], [0, 0, 1, 1]])
        # strict threshold
        numpy.testing.assert_array_equal(r.predict(X, threshold=1.0), [[0, 0, 0, 0], [0, 0, 0, 0]])

    def test_evaluate(self):
        X, Y = make_data(0)
        self.assertIsInstance(RAkELClassifier(random_state=0).fit(X, Y).evaluate(X, Y), dict)


if __name__ == "__main__":
    unittest.main()
