import itertools
import unittest

import numpy as np

from MLC.functions import assess as mlc_assess
from notebooks import utils as nu


def _models(spec):
    # spec: {name: (macro_f1, [f1 per label])}
    out = {}
    for name, (macro, f1s) in spec.items():
        report = {str(i): {"f1-score": v} for i, v in enumerate(f1s)}
        report["macro avg"] = {"f1-score": macro}
        out[name] = {"model": f"m_{name}", "assess": {"report": report}}
    return out


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


class AssessTest(unittest.TestCase):
    def setUp(self):
        self.Y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]])
        self.P = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]])
        self.S = np.array(
            [[0.9, 0.2, 0.4], [0.1, 0.8, 0.7], [0.7, 0.4, 0.9], [0.3, 0.6, 0.5]]
        )

    def test_equals_mlc_with_scores(self):
        self.assertTrue(
            _same(nu.assess(self.Y, self.P, self.S), mlc_assess(self.Y, self.P, self.S))
        )

    def test_equals_mlc_without_scores(self):
        self.assertTrue(_same(nu.assess(self.Y, self.P), mlc_assess(self.Y, self.P)))

    def test_single_class_label_is_nan(self):
        r = nu.assess(self.Y, self.P, self.S)
        self.assertTrue(np.isnan(r["report"]["2"]["auc"]))

    def test_scores_change_auc(self):
        a = nu.assess(self.Y, self.P, self.S)["report"]["0"]["auc"]
        b = nu.assess(self.Y, self.P)["report"]["0"]["auc"]
        self.assertEqual(a, 1.0)
        self.assertEqual(b, 1.0)
        a1 = nu.assess(self.Y, self.P, self.S)["report"]["1"]["auc"]
        b1 = nu.assess(self.Y, self.P)["report"]["1"]["auc"]
        self.assertNotEqual(a1, b1)


class ExtractModelsTest(unittest.TestCase):
    def test_per_label_best(self):
        m = _models({"a": (0.5, [0.9, 0.1]), "b": (0.6, [0.2, 0.8])})
        r = nu.extract_models([0, 1], m, "f1-score")
        self.assertEqual(r["0"]["name"], "a")
        self.assertEqual(r["1"]["name"], "b")
        self.assertEqual(r["1"]["model"], "m_b")
        self.assertEqual(r["1"]["performance"], 0.8)

    def test_tie_independent_of_order(self):
        spec = {"c": (0.5, [0.7]), "a": (0.5, [0.7]), "b": (0.6, [0.7])}
        names = set()
        for perm in itertools.permutations(spec):
            r = nu.extract_models([0], _models({k: spec[k] for k in perm}), "f1-score")
            names.add(r["0"]["name"])
        self.assertEqual(names, {"b"})
        spec = {"c": (0.5, [0.7]), "a": (0.5, [0.7])}
        for perm in itertools.permutations(spec):
            r = nu.extract_models([0], _models({k: spec[k] for k in perm}), "f1-score")
            self.assertEqual(r["0"]["name"], "a")

    def test_nan_never_wins(self):
        nan = float("nan")
        for first in ("x", "y"):
            order = [first] + [k for k in ("x", "y") if k != first]
            spec = {"x": (0.9, [nan]), "y": (0.1, [0.2])}
            r = nu.extract_models([0], _models({k: spec[k] for k in order}), "f1-score")
            self.assertEqual(r["0"]["name"], "y")

    def test_all_nan_deterministic(self):
        nan = float("nan")
        r = nu.extract_models(
            [0], _models({"b": (0.5, [nan]), "a": (0.5, [nan])}), "f1-score"
        )
        self.assertEqual(r["0"]["name"], "a")


if __name__ == "__main__":
    unittest.main()
