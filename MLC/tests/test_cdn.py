import unittest

import numpy

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
