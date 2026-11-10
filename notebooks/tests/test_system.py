import unittest

import numpy as np
import pandas as pd

from notebooks import system as sy
from notebooks import taxonomy as tx

CS = "computer security"
TAX = tx.default_taxonomy()
TOPC = TAX.classes("top")
OTHER = TAX.other


class StubFilter:
    def __init__(self, classes=(False, True), col=0):
        self.classes_ = np.array(classes)
        self.col = col  # feature column carrying the relevance flag

    def predict_proba(self, X):
        p = X[:, self.col].astype(float)
        pos = list(self.classes_).index(True)
        out = np.zeros((len(X), 2))
        out[:, pos], out[:, 1 - pos] = p, 1 - p
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class StubCat:
    """predict(X) = X[:, cols] > 0.5 where column j of the output reads feature col map[class]."""

    def __init__(self, level, classes, colmap, embedder="E"):
        self.level, self.classes, self.embedder = level, tuple(classes), embedder
        self.colmap = colmap
        self.calls = []

    def predict_proba(self, X):
        return np.stack([X[:, self.colmap[c]] for c in self.classes], axis=1).astype(float)

    def predict(self, X):
        self.calls.append(len(X))
        return (self.predict_proba(X) > 0.5).astype(np.int64)


# feature layout: col0 relevance, cols 1.. one per top class, then sub cols
TOPCOL = {c: 1 + j for j, c in enumerate(TOPC)}
SUBCOL, k = {}, 1 + len(TOPC)
for par in TAX.parents():
    for c in TAX.classes(par):
        SUBCOL[(par, c)] = k
        k += 1
NF = k


def feats_matrix(rows):
    """rows: list of dicts {rel, top:[..], sub:{(parent, cls)}}."""
    X = np.zeros((len(rows), NF))
    for i, r in enumerate(rows):
        X[i, 0] = r.get("rel", 1)
        for c in r.get("top", []):
            X[i, TOPCOL[c]] = 1
        for key in r.get("sub", []):
            X[i, SUBCOL[key]] = 1
    return X


def build(two=True, permute=False, filter_=None):
    top_cls = list(reversed(TOPC)) if permute else TOPC
    top = sy.CategoriserAdapter(StubCat("top", top_cls, TOPCOL), "top", "E", None, TAX)
    subs = {}
    for par in TAX.parents():
        cls = TAX.classes(par)
        cls = list(reversed(cls)) if permute else cls
        cmap = {c: SUBCOL[(par, c)] for c in cls}
        subs[par] = sy.CategoriserAdapter(StubCat(par, cls, cmap), par, "E", None, TAX)
    if two:
        return sy.TwoStepSystem(sy.FilterAdapter(filter_ or StubFilter(), "E"), top, subs, TAX)
    return sy.OneStepSystem(top, subs, TAX)


def provider(X, log=None):
    ids = [f"r{i}" for i in range(len(X))]

    def f(name, row_ids):
        if log is not None:
            log.append((name, list(row_ids)))
        return X[[ids.index(r) for r in row_ids]]
    return ids, f


def col(res, label):
    return res.P[:, res.labels.index(label)]


class LabelsAndGold(unittest.TestCase):
    def test_public_order_unique(self):
        labs = sy.public_labels(TAX)
        self.assertEqual(len(labs), len(set(labs)))
        self.assertEqual(labs[-1], OTHER)
        n_top = len(TOPC) - 1
        self.assertEqual(labs[:n_top], [c for c in TOPC if c != OTHER])
        exp = []
        for p in TAX.parents():
            exp += [c for c in TAX.classes(p) if c != OTHER]
        self.assertEqual(labs[n_top:-1], exp)

    def test_collision_raises(self):
        bad = tx.Taxonomy(other="other", top_classes=("a", "other"), top_map={}, sub_classes={"a": ("a", "other")},
                          sub_maps={"a": {}}, sha256="x")
        with self.assertRaises(sy.SystemCompositionError):
            sy.public_labels(bad)

    def test_gold_cases(self):
        L = sy.gold_public
        self.assertEqual(L([CS], False), [OTHER])  # irrelevant
        self.assertEqual(L(["football"], True), [OTHER])  # relevant, unmapped
        self.assertEqual(L(["network security"], True), [OTHER])  # unmapped parent not in watson
        self.assertEqual(L(["hardware"], True), ["hardware"])  # mapped top, parent only
        self.assertEqual(L([CS], True), [CS])  # sub other dropped, parent stays
        self.assertEqual(L([CS, "network security"], True), [CS, "network security"])  # top + sub
        self.assertEqual(L(["network security", "hardware"], True), ["hardware"])  # sub without mapped parent
        self.assertEqual(L(["hardware", CS, "network security"], True), [CS, "hardware", "network security"])
        self.assertEqual(L(["operating systems", "networking"], True), ["networking", "operating systems"])

    def test_gold_matrix(self):
        df = pd.DataFrame({"watson": [str({"categories": [{"label": "/technology and computing/computer security/network security", "score": .9}]}),
                                      str({"categories": []})], "relevant": [True, True]})
        M = sy.gold_matrix_frame(df)
        labs = sy.public_labels()
        self.assertEqual(M.shape, (2, len(labs)))
        self.assertEqual(M[0, labs.index("computer security")], 1)
        self.assertEqual(M[0, labs.index(OTHER)], 0)
        self.assertEqual(M[1, labs.index(OTHER)], 1)
        self.assertEqual(M[1].sum(), 1)


class Cascade(unittest.TestCase):
    def run_two(self, rows, **kw):
        sysm = build(**kw)
        ids, f = provider(feats_matrix(rows))
        return sysm, sysm.predict(ids, f)

    def test_filter_reject(self):
        _, r = self.run_two([{"rel": 0, "top": ["hardware"]}, {"rel": 1, "top": ["hardware"]}])
        self.assertEqual(list(r.filter_reject), [True, False])
        self.assertEqual(r.P[0].sum(), 1)
        self.assertEqual(col(r, OTHER)[0], 1)
        self.assertEqual(col(r, "hardware")[1], 1)
        self.assertTrue(np.isnan(r.top_scores[0]).all())
        self.assertEqual(r.top_pred[0].sum(), 0)

    def test_one_step_no_filter(self):
        sysm = build(two=False)
        ids, f = provider(feats_matrix([{"rel": 0, "top": ["hardware"]}]))
        r = sysm.predict(ids, f)
        self.assertFalse(r.filter_reject.any())
        self.assertEqual(col(r, "hardware")[0], 1)

    def test_empty_top_fallback(self):
        _, r = self.run_two([{"rel": 1}])
        self.assertTrue(r.empty_top_fallback[0])
        self.assertEqual(col(r, OTHER)[0], 1)
        self.assertEqual(r.P[0].sum(), 1)

    def test_other_conflict(self):
        _, r = self.run_two([{"rel": 1, "top": [OTHER, "software"]}])
        self.assertTrue(r.other_conflict[0])
        self.assertTrue(r.top_other[0])
        self.assertEqual(col(r, OTHER)[0], 0)
        self.assertEqual(col(r, "software")[0], 1)

    def test_top_other_only(self):
        _, r = self.run_two([{"rel": 1, "top": [OTHER]}])
        self.assertTrue(r.top_other[0])
        self.assertFalse(r.empty_top_fallback[0])
        self.assertEqual(col(r, OTHER)[0], 1)

    def test_sub_gated_on_predicted_parent(self):
        rows = [{"top": ["hardware"], "sub": [("computer security", "network security")]},
                {"top": ["computer security"], "sub": [("computer security", "network security")]}]
        sysm, r = self.run_two(rows)
        self.assertEqual(col(r, "network security")[0], 0)
        self.assertEqual(col(r, "network security")[1], 1)
        self.assertEqual(list(r.sub_applied["computer security"]), [False, True])
        self.assertTrue(np.isnan(r.sub_scores["computer security"][0]).all())
        self.assertEqual(sysm.subs["computer security"].categoriser.calls, [1])

    def test_row0_regression(self):
        subs = TAX.classes("computer security")
        a, b = [c for c in subs if c != OTHER][:2]
        rows = [{"top": ["computer security"], "sub": [("computer security", a)]},
                {"top": ["computer security"], "sub": [("computer security", b)]},
                {"top": ["computer security"], "sub": [("computer security", a), ("computer security", b)]}]
        sysm, r = self.run_two(rows)
        self.assertEqual(list(col(r, a)), [1, 0, 1])
        self.assertEqual(list(col(r, b)), [0, 1, 1])
        self.assertEqual(sysm.subs["computer security"].categoriser.calls, [3])

    def test_sub_other_dropped_and_parent_stays(self):
        rows = [{"top": ["computer security"], "sub": [("computer security", OTHER)]},
                {"top": ["computer security"], "sub": []}]
        _, r = self.run_two(rows)
        self.assertEqual(list(r.parent_without_child["computer security"]), [True, True])
        self.assertEqual(col(r, "computer security")[0], 1)
        self.assertEqual(r.P[0].sum(), 1)
        self.assertEqual(col(r, OTHER)[0], 0)

    def test_sub_none_conflict(self):
        real = [c for c in TAX.classes("software") if c != OTHER][0]
        rows = [{"top": ["software"], "sub": [("software", OTHER), ("software", real)]}]
        _, r = self.run_two(rows)
        self.assertTrue(r.sub_none_conflict["software"][0])
        self.assertFalse(r.parent_without_child["software"][0])
        self.assertEqual(col(r, real)[0], 1)

    def test_networking_passthrough(self):
        _, r = self.run_two([{"top": ["networking"]}])
        self.assertEqual(col(r, "networking")[0], 1)
        self.assertNotIn("networking", r.sub_pred)

    def test_permuted_class_order(self):
        rows = [{"top": ["hardware", "computer security"], "sub": [("computer security", "network security")]}]
        _, r1 = self.run_two(rows)
        _, r2 = self.run_two(rows, permute=True)
        np.testing.assert_array_equal(r1.P, r2.P)
        np.testing.assert_array_equal(r1.top_scores, r2.top_scores)

    def test_class_set_mismatch_raises(self):
        bad = StubCat("top", TOPC[:-1], TOPCOL)
        with self.assertRaises(sy.SystemCompositionError):
            sy.CategoriserAdapter(bad, "top", "E", None, TAX)

    def test_gold_independence(self):
        import inspect
        rows = [{"top": ["computer security"], "sub": [("computer security", "network security")]},
                {"rel": 0, "top": ["hardware"]}]
        ids, f = provider(feats_matrix(rows))
        for two in (True, False):
            sysm = build(two=two)
            self.assertEqual(list(inspect.signature(sysm.predict).parameters), ["row_ids", "features"])
            df = pd.DataFrame({"text": ["a", "b"], "relevant": [True, False], "watson": ["x", "y"]}, index=ids)
            r1 = sysm.predict(ids, f)
            gold_a = sy.gold_matrix([[CS]] * 2, [True, True])
            df2 = df.assign(relevant=[False, True], watson=["z", "w"], text=["c", "d"])
            gold_b = sy.gold_matrix([[]] * 2, [False, False])
            self.assertFalse((gold_a == gold_b).all())
            r2 = sysm.predict(list(df2.index), f)  # only ids and features reach the system
            np.testing.assert_array_equal(r1.P, r2.P)
            np.testing.assert_array_equal(r1.top_scores, r2.top_scores)
            np.testing.assert_array_equal(r1.filter_score, r2.filter_score)
            for p in r1.sub_scores:
                np.testing.assert_array_equal(r1.sub_scores[p], r2.sub_scores[p])

    def test_positive_class_lookup(self):
        for classes in [(False, True), (True, False), (0, 1), (1, 0)]:
            f = sy.FilterAdapter(StubFilter(classes), "E")
            X = np.array([[1.0], [0.0]])
            self.assertEqual(list(f.accept(X)), [True, False])
            np.testing.assert_allclose(f.score(X), [1.0, 0.0])
        with self.assertRaises(sy.SystemCompositionError):
            sy.FilterAdapter(StubFilter(("a", "b")), "E")

    def test_result_immutable(self):
        _, r = self.run_two([{"top": ["hardware"]}])
        with self.assertRaises(ValueError):
            r.P[0, 0] = 5
        with self.assertRaises(Exception):
            r.P = None

    def test_per_component_embedder_routing(self):
        top = sy.CategoriserAdapter(StubCat("top", TOPC, TOPCOL), "top", "ET", None, TAX)
        subs = {p: sy.CategoriserAdapter(StubCat(p, TAX.classes(p), {c: SUBCOL[(p, c)] for c in TAX.classes(p)}),
                                         p, "ES", {"sha": "x"}, TAX) for p in TAX.parents()}
        sysm = sy.TwoStepSystem(sy.FilterAdapter(StubFilter(), "EF"), top, subs, TAX)
        self.assertEqual(sysm.embedders, ["EF", "ES", "ET"])
        self.assertEqual(sysm.fingerprints, {"ES": {"sha": "x"}})
        X = feats_matrix([{"top": ["computer security"]}, {"top": ["hardware"]}])
        log = []
        ids, f = provider(X, log)
        sysm.predict(ids, f)
        names = [n for n, _ in log]
        self.assertEqual(names[0], "EF")
        self.assertEqual(names[1], "ET")
        self.assertEqual(set(names[2:]), {"ES"})
        self.assertEqual([i for n, i in log if n == "ES"], [["r0"], ["r1"]])


class Providers(unittest.TestCase):
    def test_no_heavy_imports(self):
        import subprocess, sys
        code = ("import sys; import notebooks.system; "
                "bad=[m for m in ('nltk','torch','sentence_transformers') if m in sys.modules]; "
                "sys.exit(1 if bad else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)

    def test_artifact_provider_routes_by_name(self):
        from unittest import mock
        calls = []

        class Art:
            def __init__(self, n):
                self.n = n

            def select(self, ids):
                return np.full((len(ids), 1), self.n)

        def fake_load(d, name, rows, fp, **kw):
            calls.append((name, fp))
            return Art(len(calls))

        from notebooks import embeddings as em
        prov = sy.ArtifactFeatures("d", "corp", {"ATT&CK-BERT": {"sha": "q"}}, allow_missing_fingerprint=True)
        with mock.patch.object(em, "load_artifact", fake_load):
            a = prov("ATT&CK-BERT", ["r1"])
            b = prov("SBERT", ["r1", "r2"])
            prov("ATT&CK-BERT", ["r1"])
        self.assertEqual(calls, [("corp__ATT_CK-BERT", {"sha": "q"}), ("corp__SBERT", None)])
        self.assertEqual(a.shape, (1, 1))
        self.assertEqual(b.shape, (2, 1))

    def test_missing_fingerprint_raises(self):
        from unittest import mock
        from notebooks import embeddings as em
        prov = sy.ArtifactFeatures("d", "corp", {"A": {"sha": "q"}})
        with mock.patch.object(em, "load_artifact", side_effect=AssertionError("must not load")):
            with self.assertRaises(sy.SystemCompositionError):
                prov("B", ["r1"])

    def test_fingerprint_mismatch_raises(self):
        from unittest import mock
        from notebooks import embeddings as em

        def fake_load(d, name, rows, fp, **kw):
            if fp != {"sha": "actual"}:
                raise ValueError("fingerprint mismatch")
            raise AssertionError("unexpected")
        prov = sy.ArtifactFeatures("d", "corp", {"A": {"sha": "expected"}})
        with mock.patch.object(em, "load_artifact", fake_load):
            with self.assertRaises(ValueError):
                prov("A", ["r1"])

    def test_artifact_features_helper_wires_fingerprints(self):
        from unittest import mock
        from notebooks import embeddings as em
        top = sy.CategoriserAdapter(StubCat("top", TOPC, TOPCOL), "top", "ET", {"sha": "t"}, TAX)
        subs = {p: sy.CategoriserAdapter(StubCat(p, TAX.classes(p), {c: SUBCOL[(p, c)] for c in TAX.classes(p)}),
                                         p, "ES", {"sha": "s"}, TAX) for p in TAX.parents()}
        sysm = sy.TwoStepSystem(sy.FilterAdapter(StubFilter(), "EF", {"sha": "f"}), top, subs, TAX)
        calls = []

        class Art:
            def select(self, ids):
                return np.zeros((len(ids), 1))

        def fake_load(d, name, rows, fp, **kw):
            calls.append((d, name, rows, fp))
            return Art()
        prov = sy.artifact_features(sysm, "dir", row_ids=["r1", "r2"])
        with mock.patch.object(em, "load_artifact", fake_load):
            for e in sysm.embedders:
                prov(e, ["r1"])
        self.assertEqual(calls, [("dir", "corpus__EF", ["r1", "r2"], {"sha": "f"}),
                                 ("dir", "corpus__ES", ["r1", "r2"], {"sha": "s"}),
                                 ("dir", "corpus__ET", ["r1", "r2"], {"sha": "t"})])

    def test_text_provider(self):
        from unittest import mock
        from notebooks import embeddings as em
        with mock.patch.object(em, "embed", lambda texts, e: np.full((len(texts), 1), float(len(e)))):
            prov = sy.TextFeatures({"a": "x", "b": "y"}, {"E1": "ab", "E2": "abc"})
            self.assertEqual(prov("E2", ["a", "b"]).tolist(), [[3.0], [3.0]])
            self.assertEqual(prov("E1", ["b"]).tolist(), [[2.0]])
            with self.assertRaises(sy.SystemCompositionError):
                prov("E3", ["a"])


class Loaders(unittest.TestCase):
    def _cats(self, levels=None, bad_level=None):
        def load(path, level, allow_non_production=False):
            cls = TAX.classes("top" if level in ("general", "one step") else level)
            cat = StubCat(level, cls, {c: 0 for c in cls}, embedder="X")
            if bad_level == level:
                cat.level = "wrong"
            return cat, {"embedder": {"name": f"E-{level}", "fingerprint": {"sha": level}}}
        return load

    def paths(self, top="one step"):
        return {lv: f"/p/{lv}" for lv in [top, *TAX.parents()]}

    def test_load_one_step_parses_sidecar(self):
        from unittest import mock
        from notebooks import categorisation as ct
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()):
            s = sy.load_one_step(self.paths(), taxonomy=TAX)
        self.assertIsInstance(s, sy.OneStepSystem)
        self.assertEqual(s.top.embedder, "E-one step")
        self.assertEqual((s.top.level, s.top.source_level), ("top", "one step"))
        self.assertEqual(s.fingerprints["E-one step"], {"sha": "one step"})
        for p in TAX.parents():
            self.assertEqual(s.subs[p].embedder, f"E-{p}")

    def test_load_two_step_parses_sidecar(self):
        from unittest import mock
        from notebooks import categorisation as ct, filtering as fl
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()), \
                mock.patch.object(fl, "load_deployed_filter",
                                  return_value=(StubFilter(), {"embedder": {"name": "EF", "fingerprint": {"sha": "f"}}})):
            s = sy.load_two_step("/p/filter", self.paths("general"), taxonomy=TAX)
        self.assertEqual(s.filter.embedder, "EF")
        self.assertEqual(s.fingerprints["EF"], {"sha": "f"})
        self.assertEqual(s.top.source_level, "general")

    def test_filter_without_embedder_needs_override(self):
        from unittest import mock
        from notebooks import categorisation as ct, filtering as fl
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()), \
                mock.patch.object(fl, "load_deployed_filter", return_value=(StubFilter(), {})):
            with self.assertRaises(sy.SystemCompositionError):
                sy.load_two_step("/p/filter", self.paths("general"), taxonomy=TAX)
            s = sy.load_two_step("/p/filter", self.paths("general"), taxonomy=TAX, filter_embedder="EF")
        self.assertEqual(s.filter.embedder, "EF")

    def test_wrong_level_raises(self):
        from unittest import mock
        from notebooks import categorisation as ct
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats(bad_level="hardware")):
            with self.assertRaises(sy.SystemCompositionError):
                sy.load_one_step(self.paths(), taxonomy=TAX)

    def test_two_step_needs_general_not_top(self):
        from unittest import mock
        from notebooks import categorisation as ct, filtering as fl
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()), \
                mock.patch.object(fl, "load_deployed_filter", return_value=(StubFilter(), {"embedder": "EF"})):
            with self.assertRaises(sy.SystemCompositionError):
                sy.load_two_step("/p/filter", self.paths("top"), taxonomy=TAX)

    def test_source_level_mismatch_raises(self):
        top_cls = TAX.classes("top")
        cat = StubCat("one step", top_cls, TOPCOL, embedder="E")
        sy.CategoriserAdapter(cat, "top", "E", None, TAX, source_level="one step")
        with self.assertRaises(sy.SystemCompositionError):
            sy.CategoriserAdapter(cat, "top", "E", None, TAX, source_level="general")
        with self.assertRaises(sy.SystemCompositionError):
            sy.CategoriserAdapter(cat, "top", "E", None, TAX)      # default source_level = level = "top"

    def test_hook_and_shared_subs(self):
        from unittest import mock
        from notebooks import categorisation as ct
        seen = []
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()):
            a = sy.load_one_step(self.paths(), taxonomy=TAX, hook=lambda c, p, m, ad: seen.append(c))
            b = sy.load_one_step({"one step": "/p/x"}, taxonomy=TAX, subs=a.subs)
        self.assertEqual(sorted(seen), sorted(["one step", *TAX.parents()]))
        self.assertIs(b.subs["hardware"], a.subs["hardware"])

    def test_missing_level_raises(self):
        from unittest import mock
        from notebooks import categorisation as ct
        paths = self.paths()
        del paths["hardware"]
        with mock.patch.object(ct, "load_deployed_categoriser", self._cats()):
            with self.assertRaises(sy.SystemCompositionError):
                sy.load_one_step(paths, taxonomy=TAX)


if __name__ == "__main__":
    unittest.main()
