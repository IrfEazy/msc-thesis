import unittest

import numpy

from MLC._base import positive_score

from MLC.CDNClassifier import CDNClassifier
from MLC.tests.common import check_contract, make_data


class CDNTests(unittest.TestCase):
    def test_contract(self):
        check_contract(self, lambda: CDNClassifier(n_iterations=10, burn_in=5, random_state=0))

    def test_reproducible_proba(self):
        X, Y = make_data(0)
        clf = CDNClassifier(n_iterations=12, burn_in=4, random_state=3).fit(X, Y)
        P = clf.predict_proba(X)
        self.assertEqual(P.shape, Y.shape)
        self.assertTrue(((P >= 0) & (P <= 1)).all())
        numpy.testing.assert_array_equal(P, clf.predict_proba(X))
        other = CDNClassifier(n_iterations=12, burn_in=4, random_state=3).fit(X, Y)
        numpy.testing.assert_array_equal(P, other.predict_proba(X))

    def test_matches_reference_loop(self):
        X, Y = make_data(0)
        clf = CDNClassifier(n_iterations=12, burn_in=4, random_state=3).fit(X, Y)
        n, q = X.shape[0], Y.shape[1]
        rng = numpy.random.RandomState(3)
        state = rng.randint(0, 2, size=(n, q))
        total = numpy.zeros((n, q))
        for it in range(12):
            for j in rng.permutation(q):
                X_aug = numpy.hstack((X, numpy.delete(state, j, axis=1)))
                p = positive_score(clf.estimators_[j], X_aug)
                state[:, j] = rng.random_sample(n) < p
            if it >= 4:
                total += state
        numpy.testing.assert_array_equal(clf.predict_proba(X), total / 8)

    def test_sequential_updates_within_sweep(self):
        class Copy:
            """Score = value of the first label column (right after X)."""

            classes_ = numpy.array([0, 1])

            def __init__(self, col):
                self.col = col

            def predict_proba(self, X):
                v = X[:, self.col].astype(float)
                return numpy.column_stack([1 - v, v])

        # Two labels, no features: X has 1 column, the other label is column 1.
        X = numpy.zeros((50, 1))
        clf = CDNClassifier(n_iterations=4, burn_in=0, random_state=0)
        clf.estimators_ = [Copy(1), Copy(1)]
        clf.n_labels_ = 2
        P = clf.predict_proba(X)
        # Reference sequential loop: the update of j uses the freshly updated
        # other label, so after the first step both labels are equal.
        rng = numpy.random.RandomState(0)
        state = rng.randint(0, 2, size=(50, 2))
        total = numpy.zeros((50, 2))
        for _ in range(4):
            for j in rng.permutation(2):
                p = state[:, 1 - j].astype(float)
                state[:, j] = rng.random_sample(50) < p
            total += state
        numpy.testing.assert_array_equal(P, total / 4)
        # A simultaneous update would swap the columns each sweep instead of
        # equalising them; sequential updates leave both labels identical.
        numpy.testing.assert_array_equal(P[:, 0], P[:, 1])

    def test_constant_zero_label(self):
        X, Y = make_data(0)
        Y[:, 1] = 0
        P = CDNClassifier(n_iterations=10, burn_in=5, random_state=0).fit(X, Y).predict_proba(X)
        self.assertTrue((P[:, 1] == 0).all())

    def test_invalid_burn_in(self):
        X, Y = make_data(0)
        for burn_in in (-1, 10, 11):
            with self.assertRaises(ValueError):
                CDNClassifier(n_iterations=10, burn_in=burn_in).fit(X, Y)

    def test_evaluate(self):
        X, Y = make_data(0)
        r = CDNClassifier(n_iterations=10, burn_in=5, random_state=0).fit(X, Y).evaluate(X, Y)
        self.assertEqual(set(r), {"accuracy", "hamming_loss", "report"})


if __name__ == "__main__":
    unittest.main()
