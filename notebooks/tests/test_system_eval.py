import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

from notebooks import system as sy
from notebooks import system_eval as se
from notebooks import taxonomy as tx
from notebooks.tests import test_system as ts

OTHER = tx.default_taxonomy().other
TAXO = tx.Taxonomy(other=OTHER, top_classes=("a", "b", OTHER), top_map={}, sub_classes={"a": ("s1", OTHER)},
                   sub_maps={"a": {}}, sha256="test")


def esys(*a, **kw):
    kw.setdefault("taxonomy", TAXO)
    return se.evaluate_systems(*a, **kw)
LABELS = ("a", "b", "s1", OTHER)  # tops a, b; sub of parent a: s1 (+ sub other); public other
N = 6


def make(P, name="two_step", fscore=None, reject=None, top_scores=None, sub_scores=None, sub_applied=None):
    P = np.asarray(P, dtype=np.int64)
    n = len(P)
    z = np.zeros(n, bool)
    top_scores = np.full((n, 3), np.nan) if top_scores is None else np.asarray(top_scores, float)
    sub_scores = np.full((n, 2), np.nan) if sub_scores is None else np.asarray(sub_scores, float)
    applied = z if sub_applied is None else np.asarray(sub_applied, bool)
    return sy.SystemResult(
        system=name, row_ids=tuple(f"r{i}" for i in range(n)), labels=LABELS, P=P,
        filter_score=np.full(n, np.nan) if fscore is None else np.asarray(fscore, float),
        top_classes=("a", "b", OTHER), top_pred=np.zeros((n, 3), np.int64), top_scores=top_scores,
        sub_classes={"a": ("s1", OTHER)}, sub_pred={"a": np.zeros((n, 2), np.int64)},
        sub_scores={"a": sub_scores}, sub_applied={"a": applied},
        filter_reject=z if reject is None else np.asarray(reject, bool),
        top_other=z, empty_top_fallback=z, other_conflict=z,
        sub_none_conflict={"a": z}, parent_without_child={"a": z})


GOLD = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])
GROUPS = np.array(["g0", "g0", "g1", "g2", "g2", "g3"])
REL = GOLD[:, 3] == 0
PRED = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])


class TestMetrics(unittest.TestCase):
    def ev(self, results=None, **kw):
        results = results or {"x": make(PRED, "x")}
        kw.setdefault("B", 20)
        kw.setdefault("seed", 1)
        return esys(results, GOLD, REL, GROUPS, **kw)

    def test_hand_computed_micro_macro(self):
        e2e = self.ev()["systems"]["x"]["end_to_end"]
        # tp: a=3,b=0,s1=1,other=1 ; fp: b=1,s1=1,other=1 ; fn: b=2,other=1
        tp, fp, fn = 5, 3, 3
        self.assertAlmostEqual(e2e["micro_precision"]["value"], tp / (tp + fp))
        self.assertAlmostEqual(e2e["micro_recall"]["value"], tp / (tp + fn))
        self.assertAlmostEqual(e2e["micro_f1"]["value"], 2 * tp / (2 * tp + fp + fn))
        f1 = [1.0, 0.0, 2 / 3, 0.5]
        self.assertAlmostEqual(e2e["macro_f1"]["value"], sum(f1) / 4)
        self.assertAlmostEqual(e2e["subset_accuracy"]["value"], 2 / 6)  # rows 0 and 3
        self.assertAlmostEqual(e2e["hamming_loss"]["value"], 6 / 24)

    def test_macro_denominator_fixed(self):
        gold = GOLD.copy()
        pred = PRED.copy()
        pred[:, 2] = 0
        gold[:, 2] = 0  # label s1: no support, no prediction -> counts as 0
        r = esys({"x": make(pred, "x")}, gold, REL, GROUPS, B=5, seed=0)
        pl = r["systems"]["x"]["per_label"]["s1"]
        self.assertEqual(pl["f1"]["value"], 0.0)
        e = r["systems"]["x"]["end_to_end"]["macro_f1"]["value"]
        f = [r["systems"]["x"]["per_label"][c]["f1"]["value"] for c in LABELS]
        self.assertAlmostEqual(e, sum(f) / 4)

    def test_projections_columns(self):
        pr = self.ev()["systems"]["x"]["projections"]
        # top_only: a, b, other ; sub_only: s1 ; mapped_only: a, b, s1
        def micro(cols):
            Y, P = GOLD[:, cols], PRED[:, cols]
            tp, fp, fn = (Y * P).sum(), ((1 - Y) * P).sum(), (Y * (1 - P)).sum()
            return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        self.assertAlmostEqual(pr["top_only"]["micro_f1"]["value"], micro([0, 1, 3]))
        self.assertAlmostEqual(pr["sub_only"]["micro_f1"]["value"], micro([2]))
        self.assertAlmostEqual(pr["mapped_only"]["micro_f1"]["value"], micro([0, 1, 2]))

    def test_multiplicity_matches_integer_weights_and_row_repetition(self):
        counts = se.group_draws(4, 3, 7)[0:1]
        gidx = np.unique(GROUPS, return_inverse=True)[1]
        W = se.row_weights(counts, gidx)
        rows = np.repeat(np.arange(N), W[0].astype(int))
        Y, P = GOLD[rows], PRED[rows]
        tp, fp, fn = (Y * P).sum(), ((1 - Y) * P).sum(), (Y * (1 - P)).sum()
        micro_rep = 2 * tp / (2 * tp + fp + fn)
        Yf, Pf = GOLD.astype(float), PRED.astype(float)
        tpw, fpw, fnw = (W @ (Yf * Pf)).sum(), (W @ ((1 - Yf) * Pf)).sum(), (W @ (Yf * (1 - Pf))).sum()
        self.assertAlmostEqual(micro_rep, 2 * tpw / (2 * tpw + fpw + fnw))
        # weighted AUC equals sklearn with sample_weight
        score = np.array([0.9, 0.2, 0.2, 0.5, 0.7, 0.1])
        y = np.array([1, 0, 1, 0, 1, 0])
        if W[0][y == 1].sum() and W[0][y == 0].sum():
            self.assertAlmostEqual(se._Auc(score, y)(W)[0], roc_auc_score(y, score, sample_weight=W[0]))
        W1 = np.array([[2.0, 1, 3, 0, 1, 2]])
        self.assertAlmostEqual(se._Auc(score, y)(W1)[0], roc_auc_score(y, score, sample_weight=W1[0]))

    def test_identical_systems_zero_difference_and_determinism(self):
        res = {"x": make(PRED, "x"), "y": make(PRED, "y")}
        a = self.ev(res, B=30, seed=3)
        for k, v in a["paired"]["x_minus_y"].items():
            self.assertEqual(v["value"], 0.0)
            self.assertEqual(v["ci"], [0.0, 0.0])
        b = self.ev(res, B=30, seed=3)
        self.assertEqual(a, b)
        # same draws for both systems: identical metric CIs
        self.assertEqual(a["systems"]["x"]["end_to_end"]["micro_f1"], a["systems"]["y"]["end_to_end"]["micro_f1"])
        c = self.ev(res, B=30, seed=4)
        self.assertNotEqual(a["systems"]["x"]["end_to_end"]["micro_f1"]["ci"],
                            c["systems"]["x"]["end_to_end"]["micro_f1"]["ci"])

    def test_mismatched_results_raise(self):
        bad = make(PRED[:5], "y")
        with self.assertRaises(se.SystemEvalError):
            esys({"x": make(PRED, "x"), "y": bad}, GOLD, REL, GROUPS, B=2, seed=0)

    def test_filter_fn_split_and_relevance(self):
        # gold relevant rows: 0,1,2,5 ; rejected: row 1 (mapped gold), row 2 (mapped), row 3 (irrelevant)
        gold = GOLD.copy()
        gold[4] = [0, 0, 0, 1]
        rel = np.array([1, 1, 1, 0, 1, 1], bool)  # row 4 relevant but unmapped (gold other)
        rej = np.array([0, 1, 0, 1, 1, 0], bool)
        r = make(PRED, "two_step", fscore=[.9, .2, .6, .1, .3, .8], reject=rej)
        out = esys({"two_step": r}, gold, rel, GROUPS, B=10, seed=0)["systems"]["two_step"]["filter"]
        self.assertEqual(out["confusion"], [[1, 0], [2, 3]])
        self.assertEqual(out["dropped_relevant_mapped"]["value"], 1.0)
        self.assertEqual(out["dropped_relevant_unmapped"]["value"], 1.0)
        self.assertAlmostEqual(out["recall"]["value"], 3 / 5)
        self.assertAlmostEqual(out["false_negative_rate"]["value"], 2 / 5)
        self.assertAlmostEqual(out["auc"]["value"], roc_auc_score(rel, [.9, .2, .6, .1, .3, .8]))

    def test_auc_none_for_single_class_and_macro_counts(self):
        ts = np.array([[.9, .1, .1], [.8, .2, .1], [.1, .7, .2], [.1, .1, .9], [.2, .2, .8], [.7, .6, .1]])
        r = make(PRED, "x", top_scores=ts)
        top = esys({"x": r}, GOLD, REL, GROUPS, B=15, seed=0)["systems"]["x"]["stage_auc"]["top"]
        for j, c in enumerate(("a", "b", OTHER)):
            self.assertAlmostEqual(top["per_class"][c]["value"], roc_auc_score(GOLD[:, [0, 1, 3][j]], ts[:, j]))
        self.assertEqual(top["macro"]["n_undefined"], 0)
        # single gold class on the population: s1 scores only where sub ran, rows 1-2 (gold s1: 0,0)
        sub_sc = np.array([[np.nan] * 2, [.4, .6], [.2, .8]] + [[np.nan] * 2] * 3)
        r2 = make(PRED, "x", sub_scores=sub_sc, sub_applied=[0, 1, 1, 0, 0, 0])
        sub = esys({"x": r2}, GOLD, REL, GROUPS, B=15, seed=0)["systems"]["x"]["stage_auc"]["sub"]
        self.assertIsNone(sub["a"]["per_class"]["s1"]["value"])
        self.assertEqual(sub["a"]["per_class"]["s1"]["n_valid"], 0)
        self.assertIsNone(sub["a"]["macro"]["value"])
        self.assertEqual(sub["a"]["macro"]["n_undefined"], 1)

    def test_oracle_block_and_json(self):
        import json
        pred = np.array([[1, 0], [0, 1], [1, 1]])
        gold = np.array([[1, 0], [0, 1], [1, 0]])
        sc = np.array([[.9, .1], [.2, .8], [.7, .6]])
        o = {"top": (pred, sc, gold, ["g0", "g1", "g3"], ["a", "b"])}
        out = esys({"x": make(PRED, "x")}, GOLD, REL, GROUPS, oracle=o, B=10, seed=0)
        blk = out["oracle"]["top"]
        self.assertEqual(blk["n"], 3)
        self.assertAlmostEqual(blk["per_class"]["a"]["auc"]["value"], 1.0)
        self.assertAlmostEqual(blk["micro_f1"]["value"], 2 * 3 / (2 * 3 + 1 + 0))
        json.dumps(out)
        with self.assertRaises(se.SystemEvalError):
            esys({"x": make(PRED, "x")}, GOLD, REL, GROUPS, B=2,
                                oracle={"top": (pred, sc, gold, ["g0", "g1", "zz"], ["a", "b"])})

    def test_ci_matches_independent_percentile_of_seeded_draws(self):
        B, seed = 40, 5
        out = self.ev(B=B, seed=seed)["systems"]["x"]["end_to_end"]["micro_f1"]
        ug, gidx = np.unique(GROUPS, return_inverse=True)
        counts = se.group_draws(len(ug), B, seed)
        vals = []
        for b in range(B):
            rows = np.repeat(np.arange(N), counts[b][gidx])  # every row of a drawn group, with multiplicity
            Y, P = GOLD[rows], PRED[rows]
            tp, fp, fn = (Y * P).sum(), ((1 - Y) * P).sum(), (Y * (1 - P)).sum()
            vals.append(2 * tp / (2 * tp + fp + fn))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        self.assertAlmostEqual(out["ci"][0], lo)
        self.assertAlmostEqual(out["ci"][1], hi)
        self.assertEqual(out["n_valid"], B)

    def test_support_groups_implicit_relevance_and_routing(self):
        r = self.ev()["systems"]["x"]
        # gold groups per label: a -> g0,g3 ; b -> g1,g3 ; s1 -> g0 ; other -> g2
        sg = {c: r["per_label"][c]["support_groups"] for c in LABELS}
        self.assertEqual(sg, {"a": 2, "b": 2, "s1": 1, OTHER: 1})
        self.assertEqual(r["per_label"]["a"]["support_rows"]["value"], 3.0)
        # implicit relevance: predicted mapped = rows 0,1,4,5 ; gold relevant = rows 0,1,2,5
        # tp=3 (0,1,5) fp=1 (4) fn=1 (2) tn=1 (3)
        ir = r["implicit_relevance"]
        self.assertAlmostEqual(ir["precision"]["value"], 3 / 4)
        self.assertAlmostEqual(ir["recall"]["value"], 3 / 4)
        self.assertAlmostEqual(ir["f1"]["value"], 3 / 4)
        self.assertAlmostEqual(ir["balanced_accuracy"]["value"], 0.5 * (3 / 4 + 1 / 2))
        # routing passthrough
        z = np.zeros(N, bool)
        res = make(PRED, "x")
        res = sy.SystemResult(**{**res.__dict__, "filter_reject": np.array([1, 1, 0, 0, 0, 0], bool),
                                 "top_other": np.array([0, 0, 0, 1, 1, 0], bool),
                                 "sub_none_conflict": {"a": np.array([0, 0, 1, 0, 0, 0], bool)},
                                 "sub_applied": {"a": np.array([1, 1, 1, 0, 0, 0], bool)}})
        rt = self.ev({"x": res})["systems"]["x"]["routing"]
        self.assertEqual(rt["filter_reject"], {"count": 2, "rate": 2 / 6})
        self.assertEqual(rt["top_other"], {"count": 2, "rate": 2 / 6})
        self.assertEqual(rt["sub_none_conflict"]["a"], {"count": 1, "rate": 1 / 6})
        self.assertEqual(rt["sub_applied"]["a"], {"count": 3, "rate": 3 / 6})
        self.assertEqual(rt["other_conflict"]["count"], 0)
        del z

    def test_empty_projection_is_undefined(self):
        res = make(PRED, "x")
        res = sy.SystemResult(**{**res.__dict__, "sub_classes": {"a": (OTHER,)}, "sub_pred": {"a": np.zeros((N, 1), np.int64)},
                                 "sub_scores": {"a": np.full((N, 1), np.nan)}})
        so = self.ev({"x": res})["systems"]["x"]["projections"]["sub_only"]
        self.assertIsNone(so["micro_f1"]["value"])
        self.assertIsNone(so["macro_f1"]["value"])
        self.assertEqual(so["micro_f1"]["ci"], [None, None])

    def test_speed_large(self):
        rng = np.random.default_rng(0)
        n, L = 4300, 20

        def valid(prob):
            M = (rng.random((n, L)) < prob).astype(int)
            M[:, 2] &= M[:, 0]                      # s1 implies a
            M[:, -1] = 0
            M[M.sum(axis=1) == 0, -1] = 1           # other iff no mapped label
            return M
        Y, P = valid(.15), valid(.15)
        labels = ("a", "b", "s1") + tuple(f"l{i}" for i in range(L - 4)) + (OTHER,)
        r = make(P[:, :4], "x")
        r = sy.SystemResult(**{**r.__dict__, "labels": labels, "P": P, "row_ids": tuple(map(str, range(n))),
                               "filter_score": np.full(n, np.nan), "filter_reject": np.zeros(n, bool),
                               "top_other": np.zeros(n, bool), "empty_top_fallback": np.zeros(n, bool),
                               "other_conflict": np.zeros(n, bool),
                               "top_scores": rng.random((n, 3)),
                               "sub_scores": {"a": rng.random((n, 2))}, "sub_applied": {"a": rng.random(n) < .5}})
        out = esys({"x": r, "y": r}, Y, Y[:, -1] == 0, rng.integers(0, 2000, n), B=300, seed=0)
        self.assertEqual(out["systems"]["x"]["end_to_end"]["micro_f1"]["n_valid"], 300)


CS = "computer security"
HW = "hardware"


class TestOracleInputs(unittest.TestCase):
    TAX = tx.default_taxonomy()
    # rows: 0 CS+sub, 1 hardware, 2 relevant unmapped, 3 CS without sub, 4 irrelevant
    WL = [[CS, "network security"], [HW], ["football"], [CS], []]
    REL = [True, True, True, True, False]
    GRP = ["g0", "g1", "g2", "g3", "g4"]

    def setUp(self):
        rows = [{"rel": 1, "top": [CS], "sub": [(CS, "network security")]}, {"rel": 1, "top": [HW]},
                {"rel": 1, "top": [OTHER]}, {"rel": 1, "top": [CS], "sub": [(CS, OTHER)]}, {"rel": 0}]
        self.X = ts.feats_matrix(rows)
        self.log = []
        self.ids, self.feat = ts.provider(self.X, self.log)
        self.system = ts.build(two=True)

    def test_populations_and_gold(self):
        out = se.oracle_inputs(self.system, self.ids, self.feat, self.WL, self.REL, self.GRP, self.TAX)
        # no filter: only the component embeddings were requested; top on gold-relevant rows only
        self.assertEqual(self.log[0], ("E", ["r0", "r1", "r2", "r3"]))
        self.assertEqual(self.log[1:], [("E", ["r0", "r3"]), ("E", ["r1"])])  # sub CS: gold-parent rows; sub hardware: r1
        self.assertEqual(set(out), {"top", f"sub:{CS}", f"sub:{HW}"})
        pred, score, gold, grp, classes = out["top"]
        self.assertEqual(grp, ["g0", "g1", "g2", "g3"])
        self.assertEqual(classes, list(self.TAX.classes("top")))
        exp = np.zeros((4, len(classes)), int)
        for i, labs in enumerate([[CS], [HW], [OTHER], [CS]]):
            for c in labs:
                exp[i, classes.index(c)] = 1
        np.testing.assert_array_equal(gold, exp)
        self.assertEqual(pred.shape, exp.shape)
        pred, score, gold, grp, classes = out[f"sub:{CS}"]
        self.assertEqual(grp, ["g0", "g3"])
        self.assertEqual(classes, list(self.TAX.classes(CS)))
        exp = np.zeros((2, len(classes)), int)
        exp[0, classes.index("network security")] = 1
        exp[1, classes.index(OTHER)] = 1  # sub other kept as a class
        np.testing.assert_array_equal(gold, exp)
        self.assertEqual(pred.shape, exp.shape)

    def test_accepted_by_evaluate_systems(self):
        out = se.oracle_inputs(self.system, self.ids, self.feat, self.WL, self.REL, self.GRP, self.TAX)
        res = self.system.predict(self.ids, self.feat)
        gold = sy.gold_matrix(self.WL, self.REL)
        ev = se.evaluate_systems({"two_step": res}, gold, np.array(self.REL), np.array(self.GRP), oracle=out, B=5, seed=0)
        self.assertEqual(ev["oracle"]["top"]["n"], 4)
        self.assertEqual(ev["oracle"][f"sub:{CS}"]["n"], 2)

    def test_missing_gold_label_raises_naming_label_and_stage(self):
        top = SimpleNamespace(embedder="E", classes=[c for c in self.TAX.classes("top") if c != CS],
                              predict=lambda X: np.zeros((len(X), 1)), predict_proba=lambda X: np.zeros((len(X), 1)))
        bad = SimpleNamespace(top=top, subs={}, taxonomy=self.TAX)
        with self.assertRaises(se.SystemEvalError) as cm:
            se.oracle_inputs(bad, self.ids, self.feat, self.WL, self.REL, self.GRP, self.TAX)
        self.assertIn(CS, str(cm.exception))
        self.assertIn("top", str(cm.exception))


# --------------------------------------------------------------------------- rigour tests
NAN3, NAN2 = [np.nan] * 3, [np.nan] * 2
GOLD8 = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0],
                  [1, 0, 1, 0], [0, 1, 0, 0]])
GROUPS8 = np.array(["g0", "g0", "g1", "g2", "g2", "g3", "g0", "g4"])  # unequal sizes 3,1,2,1,1
REL8 = GOLD8[:, 3] == 0
PX = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0],
               [1, 0, 0, 0], [0, 1, 0, 0]])
PY = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0],
               [1, 0, 1, 0], [1, 1, 0, 0]])


def systems8():
    x = make(PX, "x", fscore=[.9, .8, .6, .1, .1, .7, .9, .5], reject=[0, 0, 0, 1, 1, 0, 0, 0],
             top_scores=[[.9, .1, .1], [.8, .2, .1], [.1, .7, .2], NAN3, NAN3, [.7, .7, .1], [.9, .3, .1], [.1, .7, .2]],
             sub_scores=[[.7, .3], [.2, .8], NAN2, NAN2, NAN2, [.5, .5], [.7, .3], NAN2],
             sub_applied=[1, 1, 0, 0, 0, 1, 1, 0])
    y = make(PY, "y", fscore=[.5, .5, .4, .2, .6, .9, .7, .8], reject=[0, 0, 0, 1, 0, 0, 0, 0],
             top_scores=[[.6, .4, .1], [.6, .6, .3], [.2, .5, .5], NAN3, [.3, .3, .4], [.8, .2, .2], [.5, .5, .1], [.2, .9, .1]],
             sub_scores=[NAN2, [.5, .5], [.5, .5], NAN2, NAN2, NAN2, [.9, .1], [.1, .9]],
             sub_applied=[0, 1, 1, 0, 0, 0, 1, 1])
    return {"x": x, "y": y}


ORACLE8 = {"top": (np.array([[1, 0], [0, 1], [1, 0]]), np.array([[.9, .1], [.4, .4], [.4, .4]]),
                   np.array([[1, 0], [1, 0], [1, 1]]), ["g0", "g0", "g3"], ["a", "b"]),
           "sub:a": (np.array([[1, 0], [0, 1]]), np.array([[.5, .5], [.5, .5]]),
                     np.array([[1, 0], [0, 1]]), ["g0", "g0"], ["s1", OTHER])}


def repeat_result(r, rows):
    d = dict(r.__dict__)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v[rows]
        elif isinstance(v, dict) and k not in ("sub_classes",):
            d[k] = {p: a[rows] for p, a in v.items()}
    d["row_ids"] = tuple(r.row_ids[i] for i in rows)
    return sy.SystemResult(**d)


def f1_parts(Y, P):
    tp, fp, fn = (Y * P).sum(0), ((1 - Y) * P).sum(0), (Y * (1 - P)).sum(0)
    return tp, fp, fn


def e2e_metrics(Y, P):
    tp, fp, fn = f1_parts(Y, P)
    den = 2 * tp + fp + fn
    per = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    return micro, per.mean(), float((Y == P).all(axis=1).mean())


class TestRigour(unittest.TestCase):
    def draws(self, B, seed):
        ug, gidx = np.unique(GROUPS8, return_inverse=True)
        return se.group_draws(len(ug), B, seed), gidx

    def test_weighted_auc_is_pair_statistic_with_half_ties(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            n = 12
            score = rng.integers(0, 4, n) / 4.0                  # many ties
            y = rng.integers(0, 2, n)
            w = rng.integers(0, 4, n).astype(float)               # zero and unequal weights
            if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
                self.assertTrue(np.isnan(se._Auc(score, y)(w[None, :])[0]))
                continue
            num = sum(w[i] * w[j] * ((score[i] > score[j]) + 0.5 * (score[i] == score[j]))
                      for i in range(n) for j in range(n) if y[i] == 1 and y[j] == 0)
            den = w[y == 1].sum() * w[y == 0].sum()
            got = se._Auc(score, y)(w[None, :])[0]
            self.assertAlmostEqual(got, num / den)
            rows = np.repeat(np.arange(n), w.astype(int))
            self.assertAlmostEqual(got, roc_auc_score(y[rows], score[rows]))

    def test_every_metric_equals_literal_row_repetition(self):
        compared = 0
        saw_zero = saw_multi = False
        for seed in range(6):
            counts, gidx = self.draws(1, seed)
            c = counts[0]
            saw_zero |= bool((c == 0).any())
            saw_multi |= bool((c >= 2).any())
            rows = np.repeat(np.arange(len(GROUPS8)), c[gidx])
            res = systems8()
            orig = esys(res, GOLD8, REL8, GROUPS8, oracle=ORACLE8, B=1, seed=seed, compare=[("x", "y")])
            ug = np.unique(GROUPS8).tolist()
            orc = {k: (p[np.repeat(np.arange(len(g)), [c[ug.index(x)] for x in g])],
                       sc[np.repeat(np.arange(len(g)), [c[ug.index(x)] for x in g])],
                       go[np.repeat(np.arange(len(g)), [c[ug.index(x)] for x in g])],
                       [x for x in g for _ in range(c[ug.index(x)])], cl)
                   for k, (p, sc, go, g, cl) in ORACLE8.items()}
            rep = esys({k: repeat_result(v, rows) for k, v in res.items()}, GOLD8[rows], REL8[rows], GROUPS8[rows],
                       oracle=orc, B=0, seed=0, compare=[("x", "y")])

            def walk(o, r, path):
                nonlocal compared
                if "ci" in o and "value" in o:
                    lo, hi = o["ci"]
                    self.assertEqual(lo, hi, path)
                    if lo is None:
                        if "macro" in path or "macro_auc" in path:
                            return          # fixed-set rule: invalid replicate, repeated data averages fewer
                        self.assertIsNone(r["value"], path)
                    else:
                        self.assertAlmostEqual(lo, r["value"], msg=path)
                    compared += 1
                    return
                for k, v in o.items():
                    if isinstance(v, dict) and k in r:
                        walk(v, r[k], path + "/" + k)
            for part in ("systems", "paired", "oracle"):
                walk(orig[part], rep[part], part)
        self.assertTrue(saw_zero and saw_multi)
        self.assertGreater(compared, 300)

    def test_non_identical_paired_differences_verified_independently(self):
        B, seed = 25, 11
        out = esys(systems8(), GOLD8, REL8, GROUPS8, B=B, seed=seed, compare=[("x", "y")])["paired"]["x_minus_y"]
        counts, gidx = self.draws(B, seed)
        diffs = {"micro_f1": [], "macro_f1": [], "subset_accuracy": []}
        for b in range(B):
            rows = np.repeat(np.arange(8), counts[b][gidx])
            mx, my = e2e_metrics(GOLD8[rows], PX[rows]), e2e_metrics(GOLD8[rows], PY[rows])
            for i, k in enumerate(diffs):
                diffs[k].append(mx[i] - my[i])
        point = [a - b for a, b in zip(e2e_metrics(GOLD8, PX), e2e_metrics(GOLD8, PY))]
        for i, k in enumerate(diffs):
            lo, hi = np.percentile(diffs[k], [2.5, 97.5])
            self.assertAlmostEqual(out[k]["ci"][0], lo)
            self.assertAlmostEqual(out[k]["ci"][1], hi)
            self.assertAlmostEqual(out[k]["value"], point[i])
            self.assertEqual(out[k]["n_valid"], B)
        self.assertNotEqual(out["micro_f1"]["value"], 0.0)
        self.assertGreater(np.ptp(diffs["micro_f1"]), 0)

    def test_zero_division_f1_never_invalidates_paired_replicates(self):
        z = make(np.tile([0, 0, 0, 1], (8, 1)), "z")      # predicts nothing mapped: many zero-division labels
        out = esys({"x": systems8()["x"], "z": z}, GOLD8, REL8, GROUPS8, B=40, seed=1, compare=[("x", "z")])
        for k, v in out["paired"]["x_minus_z"].items():
            self.assertEqual(v["n_valid"], 40, k)

    def test_macro_auc_fixed_class_set_and_counting(self):
        B, seed = 300, 2
        out = esys({"x": systems8()["x"]}, GOLD8, REL8, GROUPS8, B=B, seed=seed)["systems"]["x"]["stage_auc"]["top"]
        counts, gidx = self.draws(B, seed)
        ran = np.array([0, 1, 2, 5, 6, 7])
        ts_ = np.array(systems8()["x"].top_scores)
        cols = {"a": 0, "b": 1}                     # OTHER has a single gold class on the population: not in the set
        self.assertIsNone(out["per_class"][OTHER]["value"])
        self.assertEqual(out["macro"]["n_undefined"], 1)
        vals, ok = [], {"a": 0, "b": 0}
        for b in range(B):
            w = counts[b][gidx]
            rows = np.repeat(np.arange(8), w)
            rows = rows[np.isin(rows, ran)]
            aucs = []
            for c, j in cols.items():
                y = GOLD8[rows, j]
                if y.min() == y.max():
                    aucs.append(None)
                else:
                    ok[c] += 1
                    aucs.append(roc_auc_score(y, ts_[rows, j]))
            if None not in aucs:
                vals.append(np.mean(aucs))
        self.assertLess(len(vals), min(ok.values()) + 1)
        self.assertTrue(len(vals) < B)
        self.assertEqual(out["macro"]["n_valid"], len(vals))
        for c in cols:
            self.assertEqual(out["per_class"][c]["n_valid"], ok[c])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        self.assertAlmostEqual(out["macro"]["ci"][0], lo)
        self.assertAlmostEqual(out["macro"]["ci"][1], hi)

    def test_auc_blocks_carry_population_and_support(self):
        out = esys(systems8(), GOLD8, REL8, GROUPS8, oracle=ORACLE8, B=10, seed=0)
        a = out["systems"]["x"]["stage_auc"]
        self.assertEqual(a["filter"]["population"], "x filter: all rows")
        self.assertEqual(a["filter"]["positive_support"]["value"], 6.0)
        self.assertEqual(a["filter"]["negative_support"]["value"], 2.0)
        self.assertEqual(a["top"]["per_class"]["a"]["population"], "x top: filter-accepted rows")
        self.assertEqual(a["top"]["macro"]["population"], "x top: filter-accepted rows")
        self.assertEqual(a["top"]["per_class"]["a"]["positive_support"]["value"], 4.0)
        self.assertEqual(a["top"]["per_class"]["a"]["negative_support"]["value"], 2.0)
        s1 = a["sub"]["a"]["per_class"]["s1"]
        self.assertEqual(s1["population"], "x sub a: rows with predicted parent")
        self.assertEqual((s1["positive_support"]["value"], s1["negative_support"]["value"]), (2.0, 2.0))
        o = out["oracle"]["top"]
        self.assertEqual(o["population"], "oracle top: gold-relevant rows")
        self.assertEqual(o["per_class"]["a"]["auc"]["population"], "oracle top: gold-relevant rows")
        self.assertEqual(out["oracle"]["sub:a"]["population"], "oracle sub:a: gold-parent rows")
        self.assertEqual(o["per_class"]["a"]["auc"]["positive_support"]["value"], 3.0)
        import json
        json.dumps(out)

    def test_empty_oracle_subset_replicates_handled_and_counted(self):
        B, seed = 60, 4
        o = {"top": (np.array([[1, 0]]), np.array([[.9, .1]]), np.array([[1, 1]]), ["g1"], ["a", "b"])}
        out = esys({"x": systems8()["x"]}, GOLD8, REL8, GROUPS8, oracle=o, B=B, seed=seed)["oracle"]["top"]
        counts, _ = self.draws(B, seed)
        empty = int((counts[:, 1] == 0).sum())              # g1 is the 2nd sorted group
        self.assertGreater(empty, 0)
        self.assertEqual(out["n_empty_replicates"], empty)
        for k in ("micro_f1", "macro_f1"):
            self.assertEqual(out[k]["n_valid"], B - empty)
        self.assertAlmostEqual(out["micro_f1"]["ci"][0], 2 / 3)
        self.assertAlmostEqual(out["micro_f1"]["ci"][1], 2 / 3)

    def test_oracle_subset_shares_group_weights_with_rows_outside_it(self):
        B, seed = 80, 6
        # oracle rows: r1 (group g0, whose rows r0 and r6 lie outside the subset) and r2 (group g1)
        o = {"top": (np.array([[1, 0], [0, 0]]), np.array([[.9, .1], [.2, .3]]), np.array([[1, 0], [1, 0]]),
                     ["g0", "g1"], ["a", "b"])}
        out = esys({"x": systems8()["x"]}, GOLD8, REL8, GROUPS8, oracle=o, B=B, seed=seed)["oracle"]["top"]
        counts, _ = self.draws(B, seed)
        vals = []
        for b in range(B):
            c0, c1 = counts[b][0], counts[b][1]              # g0 and g1 multiplicities, as for the system rows
            if c0 + c1:
                vals.append(2 * c0 / (2 * c0 + c1))          # tp = c0, fn = c1, fp = 0
        lo, hi = np.percentile(vals, [2.5, 97.5])
        self.assertAlmostEqual(out["micro_f1"]["ci"][0], lo)
        self.assertAlmostEqual(out["micro_f1"]["ci"][1], hi)
        self.assertEqual(out["micro_f1"]["n_valid"], len(vals))

    def test_public_label_invariants_are_enforced(self):
        def run(gold=GOLD8, P=PX):
            return esys({"x": make(P, "x")}, gold, REL8, GROUPS8, B=2, seed=0)
        run()
        bad = GOLD8.copy()
        bad[0] = [0, 0, 1, 0]                                  # sub without parent (gold)
        with self.assertRaisesRegex(se.SystemEvalError, "without parent"):
            run(gold=bad)
        bad = PX.copy()
        bad[2] = [0, 1, 1, 0]                                  # sub without parent (system)
        with self.assertRaisesRegex(se.SystemEvalError, "without parent"):
            run(P=bad)
        bad = PX.copy()
        bad[1] = [1, 0, 0, 1]                                  # other together with a mapped label
        with self.assertRaisesRegex(se.SystemEvalError, "iff"):
            run(P=bad)
        bad = GOLD8.copy()
        bad[3] = [0, 0, 0, 0]                                  # empty row
        with self.assertRaisesRegex(se.SystemEvalError, "empty"):
            run(gold=bad)
        bad = PX.copy()
        bad[5] = [0, 0, 0, 0]
        with self.assertRaisesRegex(se.SystemEvalError, "empty"):
            run(P=bad)
        bad = PX.copy()
        bad[5] = [0, 0, 0, 2]
        with self.assertRaisesRegex(se.SystemEvalError, "binary"):
            run(P=bad)
        se.check_public(GOLD8, LABELS, TAXO)
        with self.assertRaises(se.SystemEvalError):
            se.check_public(GOLD8[:, :3], LABELS, TAXO)

    def test_mapped_gold_on_irrelevant_row_is_rejected(self):
        rel = REL8.copy()
        rel[0] = False                                         # row 0 has mapped gold labels
        with self.assertRaisesRegex(se.SystemEvalError, "relevant=False"):
            esys({"x": make(PX, "x")}, GOLD8, rel, GROUPS8, B=2, seed=0)

    def test_oracle_inputs_are_validated(self):
        pred, sc, gold, g, cl = ORACLE8["top"]

        def run(**kw):
            a = dict(pred=pred, sc=sc, gold=gold, g=g, cl=cl)
            a.update(kw)
            o = {"top": (a["pred"], a["sc"], a["gold"], a["g"], a["cl"])}
            return esys({"x": make(PX, "x")}, GOLD8, REL8, GROUPS8, oracle=o, B=2, seed=0)
        run()
        with self.assertRaisesRegex(se.SystemEvalError, "binary"):
            run(pred=np.array([[1, 0], [0, 2], [1, 0]]))
        with self.assertRaisesRegex(se.SystemEvalError, "binary"):
            run(gold=np.array([[1, 0], [1, 0], [1, 3]]))
        bad = sc.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(se.SystemEvalError, "finite"):
            run(sc=bad)
        with self.assertRaisesRegex(se.SystemEvalError, "shapes"):
            run(cl=["a", "b", "c"])
        with self.assertRaises(se.SystemEvalError):
            run(g=["g0", "g0"])


if __name__ == "__main__":
    unittest.main()
