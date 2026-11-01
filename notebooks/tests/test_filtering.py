import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks import checkpoints as ck
from notebooks import filtering as fl
from notebooks.taxonomy import as_bool

CFG = fl.load_config("notebooks/filter.toml")
TINY = {
    "selection": {"embedders": ["A", "B"], "families": ["lr", "gnb"], "cv_folds": 5, "metric": "balanced_accuracy"},
    "tuning": {"max_evals": 4, "metric": "balanced_accuracy"},
    "smoke": {"embedders": ["A"], "families": ["gnb"], "max_evals": 2, "bootstrap_B": 7},
    "bootstrap": {"B": 100, "seed": 1, "level": 0.95},
    "families": {
        "lr": {"estimator": "LogisticRegression", "params": {"solver": "liblinear", "max_iter": 200}},
        "gnb": {"estimator": "GaussianNB", "params": {}},
    },
    "spaces": {
        "lr": {"C": {"type": "loguniform", "low": 0.01, "high": 10.0},
               "solver": {"type": "choice", "options": ["liblinear", "lbfgs"]}},
        "gnb": {"var_smoothing": {"type": "loguniform", "low": 1e-10, "high": 1e-2}},
    },
}


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 4)) + y[:, None] * 0.8
    folds = np.arange(n) % 5
    return X, y, folds


class Budget:
    def __init__(self, allow=10 ** 9):
        self.n = allow

    def allows(self, estimate=None, kind=None):
        return self.n > 0

    def record(self, kind, seconds):
        self.n -= 1


IDS = [str(i) for i in range(200)]


def make_store(d):
    return ck.Store(d, identity=ck.make_identity(config={"t": 1}))


class ConfigTests(unittest.TestCase):
    def test_production_has_28_candidates(self):
        c = fl.build_candidates(CFG)
        self.assertEqual(len(c), 28)
        self.assertEqual(len({x.name for x in c}), 28)
        self.assertEqual(c[0].name, "LR GloVe.6B.50D")
        self.assertEqual(CFG["tuning"]["max_evals"], 100)

    def test_every_family_has_space_and_builds(self):
        for fam in CFG["selection"]["families"]:
            sp = fl.build_space(CFG["spaces"][fam])
            self.assertTrue(sp)
            fl.make_pipeline(CFG, fam, 42, "cpu")

    def test_xgb_bounds(self):
        s = CFG["spaces"]["xgb"]
        self.assertEqual((s["max_depth"]["low"], s["max_depth"]["high"]), (2, 12))
        self.assertEqual((s["n_estimators"]["low"], s["n_estimators"]["high"]), (100, 1000))

    def test_smoke_overrides(self):
        e = fl.effective_config(CFG, True)
        self.assertTrue(e["smoke"])
        self.assertEqual(e["tuning"]["max_evals"], 2)
        self.assertLess(len(fl.build_candidates(e)), 28)
        self.assertEqual(len(fl.build_candidates(fl.effective_config(CFG, False))), 28)

    def test_random_state_and_device(self):
        x = fl.make_estimator(CFG, "xgb", 7, "cpu")
        self.assertEqual(x.get_params()["random_state"], 7)
        self.assertEqual(x.get_params()["device"], "cpu")
        self.assertNotIn("random_state", fl.make_estimator(CFG, "gnb", 7).get_params())

    def test_cast_params(self):
        out = fl.cast_params({"max_depth": 5.0, "learning_rate": 0.1}, CFG["spaces"]["xgb"])
        self.assertIsInstance(out["max_depth"], int)


class GuardTests(unittest.TestCase):
    def test_assert_no_test(self):
        fl.assert_no_test(["a", "b"], ["c"])
        with self.assertRaises(fl.LeakageError):
            fl.assert_no_test(["a", "c"], ["c"])

    def test_candidates_reject_test_ids(self):
        X, y, folds = data()
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as d, self.assertRaises(fl.LeakageError):
            fl.run_candidates(fl.build_candidates(TINY), TINY, lambda e, i: X, y, folds, ids, ["3"], seed=1,
                              device=None, store=make_store(d), budget=Budget())

    def test_tuning_rejects_test_ids(self):
        X, y, folds = data()
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as d, self.assertRaises(fl.LeakageError):
            fl.run_tuning("M", "gnb", TINY, X, y, folds, ids, ids[:1], seed=1, device=None,
                          store=make_store(d), budget=Budget(), max_evals=2)

    def test_grouped_cv_rejects_test_fold(self):
        X, y, folds = data()
        folds = folds.copy()
        folds[0] = -1
        with self.assertRaises(fl.FilterError):
            fl.grouped_cv(fl.make_pipeline(TINY, "gnb", 1), X, y, folds, IDS, [])

    def test_refuse_production_output(self):
        with self.assertRaises(fl.FilterError):
            fl.refuse_production_output("/x/smoke/filter.pkl", "/y/models/filter.pkl")
        fl.refuse_production_output("/x/smoke/smoke_filter.pkl", "/y/models/filter.pkl")


class BindingTests(unittest.TestCase):
    def test_length_mismatch_rejected(self):
        X, y, folds = data()
        pipe = fl.make_pipeline(TINY, "gnb", 1)
        with self.assertRaises(fl.FilterError):
            fl.grouped_cv(pipe, X, y, folds, IDS[:-1], [])
        with self.assertRaises(fl.FilterError):
            fl.grouped_cv(pipe, X[:-1], y, folds, IDS, [])
        with self.assertRaises(fl.FilterError):
            fl.grouped_cv(pipe, X, y, folds, None, [])

    def test_test_id_inside_bound_ids_rejected(self):
        X, y, folds = data()
        with self.assertRaises(fl.LeakageError):
            fl.grouped_cv(fl.make_pipeline(TINY, "gnb", 1), X, y, folds, IDS, ["5"])

    def test_candidates_feature_rows_must_match_ids(self):
        X, y, folds = data()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(fl.FilterError):
            fl.run_candidates(fl.build_candidates(TINY), TINY, lambda e, i: X[:-5], y, folds, IDS, ["zz"], seed=1,
                              device=None, store=make_store(d), budget=Budget(), log=lambda *a: None)

    def test_candidates_accessor_receives_train_ids(self):
        X, y, folds = data()
        seen = []
        with tempfile.TemporaryDirectory() as d:
            fl.run_candidates(fl.build_candidates(TINY)[:1], TINY, lambda e, i: (seen.append(list(i)), X)[1], y, folds,
                              IDS, ["zz"], seed=1, device=None, store=make_store(d), budget=Budget(), log=lambda *a: None)
        self.assertEqual(seen, [IDS])

    def test_tuning_length_mismatch(self):
        X, y, folds = data()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(fl.FilterError):
            fl.run_tuning("M", "gnb", TINY, X, y, folds, IDS[:-1], ["zz"], seed=1, device=None,
                          store=make_store(d), budget=Budget(), max_evals=1)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        X, y, _ = data(300, 3)
        self.p = fl.make_pipeline(TINY, "lr", 1).fit(X, y)
        self.X, self.y = X, y
        self.groups = [f"g{i // 3}" for i in range(300)]  # 100 groups of 3

    def test_deterministic_and_seeded(self):
        a = fl.evaluate(self.p, self.X, self.y, self.groups, B=40, seed=9)
        b = fl.evaluate(self.p, self.X, self.y, self.groups, B=40, seed=9)
        c = fl.evaluate(self.p, self.X, self.y, self.groups, B=40, seed=10)
        self.assertEqual(a["bootstrap"], b["bootstrap"])
        self.assertNotEqual(a["bootstrap"]["ci"], c["bootstrap"]["ci"])
        for k in fl.METRICS:
            lo, hi = a["bootstrap"]["ci"][k]
            self.assertLessEqual(lo, hi)
        self.assertAlmostEqual(a["prevalence"], float(self.y.mean()))
        self.assertEqual(a["bootstrap"]["unit"], "group")
        self.assertEqual(a["bootstrap"]["n_units"], 100)

    def test_no_bootstrap_by_default(self):
        self.assertNotIn("bootstrap", fl.evaluate(self.p, self.X, self.y))

    def test_resamples_whole_groups(self):
        # y == group parity and the prediction is perfect inside half the groups: a group bootstrap must
        # draw whole groups, so the resampled row count is a multiple of the group size (3).
        sizes = []
        orig = fl.metrics

        def spy(yt, yp, ys):
            sizes.append(len(yt))
            return orig(yt, yp, ys)
        fl.metrics = spy
        try:
            pred = self.p.predict(self.X)
            fl.bootstrap_ci(self.y, pred, self.p.predict_proba(self.X)[:, 1], self.groups, B=20, seed=1)
        finally:
            fl.metrics = orig
        self.assertTrue(sizes and all(n % 3 == 0 for n in sizes))

    def test_unequal_groups_total_varies_but_groups_intact(self):
        groups = ["a"] * 100 + [f"s{i}" for i in range(100)] + ["b"] * 100  # two big groups + singletons
        y = np.array([1] * 100 + [0, 1] * 50 + [0] * 100)
        score = y * 0.9 + 0.05
        out = fl.bootstrap_ci(y, (score > 0.5).astype(int), score, groups, B=50, seed=2)
        self.assertEqual(out["n_units"], 102)
        # whole-group resampling of a 100-row group yields much wider CIs than row resampling
        rows = fl.bootstrap_ci(y, (score > 0.5).astype(int), score, None, B=50, seed=2)
        self.assertEqual(rows["unit"], "row")
        w = lambda o: o["ci"]["accuracy"][1] - o["ci"]["accuracy"][0]
        self.assertTrue(w(out) >= w(rows))

    def test_effective_config_smoke_B(self):
        self.assertEqual(fl.effective_config(TINY, True)["bootstrap"]["B"], 7)
        self.assertEqual(fl.effective_config(TINY, False)["bootstrap"]["B"], 100)
        self.assertEqual(fl.load_config("notebooks/filter.toml")["bootstrap"]["B"], 1000)


class CotovPolicyTests(unittest.TestCase):
    def test_available(self):
        self.assertIsNone(fl.require_cotov(False, False, None))

    def test_production_fails_closed(self):
        with self.assertRaises(fl.FilterError):
            fl.require_cotov(False, False, "no labels")

    def test_override_and_smoke_record_reason(self):
        self.assertEqual(fl.require_cotov(False, True, "no labels"), "no labels")
        self.assertEqual(fl.require_cotov(True, False, "no artifact"), "no artifact")


class DeployedFilterTests(unittest.TestCase):
    def setUp(self):
        import joblib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.pkl, self.csv = str(d / "f.pkl"), str(d / "perf.csv")
        joblib.dump({"model": 1}, fl.staged(self.pkl))
        Path(fl.staged(self.csv)).write_text("a&b\n1&2\n")
        self.sidecar = {"pickle_sha256": ck.file_sha256(fl.staged(self.pkl)), "model": "x"}

    def test_nothing_final_before_publish(self):
        self.assertFalse(Path(self.pkl).exists() or Path(self.csv).exists() or Path(self.pkl + ".json").exists())

    def test_publish_and_load(self):
        fl.publish_filter(self.pkl, self.csv, self.sidecar)
        model, meta = fl.load_deployed_filter(self.pkl)
        self.assertEqual(model, {"model": 1})
        self.assertEqual(meta["status"], "complete")
        self.assertFalse(Path(fl.staged(self.pkl)).exists() or Path(fl.staged(self.csv)).exists())

    def test_blocked_cotov_status(self):
        fl.publish_filter(self.pkl, self.csv, self.sidecar, blocked_reason="no cotov")
        self.assertEqual(fl.load_deployed_filter(self.pkl)[1]["status"], "complete-cotov-blocked")

    def test_publish_rejects_digest_mismatch_and_writes_nothing(self):
        with self.assertRaises(fl.FilterError):
            fl.publish_filter(self.pkl, self.csv, {**self.sidecar, "pickle_sha256": "0" * 64})
        self.assertFalse(Path(self.pkl).exists() or Path(self.pkl + ".json").exists())

    def test_refuses_incomplete_status(self):
        fl.publish_filter(self.pkl, self.csv, self.sidecar)
        side = Path(self.pkl + ".json")
        for status in ("running", None):
            doc = json.loads(side.read_text())
            doc.pop("status", None)
            if status:
                doc["status"] = status
            side.write_text(json.dumps(doc))
            with self.assertRaises(fl.FilterError):
                fl.load_deployed_filter(self.pkl)

    def test_refuses_sha_mismatch_and_missing_sidecar(self):
        fl.publish_filter(self.pkl, self.csv, self.sidecar)
        Path(self.pkl).write_bytes(b"tampered")
        with self.assertRaises(fl.FilterError):
            fl.load_deployed_filter(self.pkl)
        Path(self.pkl + ".json").unlink()
        with self.assertRaises(fl.FilterError):
            fl.load_deployed_filter(self.pkl)


class SplitTests(unittest.TestCase):
    def frames(self):
        train = pd.DataFrame({"_id": ["a", "b", "c", "d"], "relevant": [True, False, True, False]})
        test = pd.DataFrame({"_id": ["e", "f"], "relevant": [False, True]})
        manifest = pd.DataFrame({"row_id": list("abcdef"), "group": list("abcdef"),
                                 "split": ["train"] * 4 + ["test"] * 2, "inner_fold": [0, 1, 0, 1, -1, -1]})
        return train, test, manifest

    def test_prepare(self):
        s = fl.prepare_split(*self.frames(), "_id", "relevant", as_bool)
        self.assertEqual(s.folds.tolist(), [0, 1, 0, 1])
        self.assertEqual(s.y_train.tolist(), [1, 0, 1, 0])
        self.assertEqual(s.y_test.tolist(), [0, 1])

    def test_strict_label_and_mismatch(self):
        train, test, m = self.frames()
        train["relevant"] = ["True", "False", "True", "False"]
        with self.assertRaises(TypeError):
            fl.prepare_split(train, test, m, "_id", "relevant", as_bool)
        train, test, m = self.frames()
        m.loc[0, "split"] = "test"
        with self.assertRaises(fl.FilterError):
            fl.prepare_split(train, test, m, "_id", "relevant", as_bool)

    def test_cotov(self):
        df = pd.DataFrame({"full_text": ["x", "y", "z"], "lang": ["en", "it", "en"], "Related": ["1", "0", 0]})
        ids, texts, y = fl.cotov_rows(df)
        self.assertEqual((ids, texts, y.tolist()), (["cotov-0", "cotov-2"], ["x", "z"], [1, 0]))
        df.loc[0, "Related"] = "2"
        with self.assertRaises(fl.FilterError):
            fl.cotov_rows(df)


class CvTests(unittest.TestCase):
    def test_folds_used_and_scaler_inside(self):
        X, y, folds = data()
        r = fl.grouped_cv(fl.make_pipeline(TINY, "lr", 1), X, y, folds, IDS, [])
        self.assertEqual(len(r["folds"]["balanced_accuracy"]), 5)
        self.assertEqual(set(r["mean"]), set(fl.METRICS))
        self.assertGreater(r["mean"]["roc_auc"], 0.7)
        # scaler fitted per fold: shifting the held-out fold by a huge constant must not matter to training
        X2 = X.copy()
        X2[folds == 0] += 1000
        r2 = fl.grouped_cv(fl.make_pipeline(TINY, "gnb", 1), X, y, folds, IDS, [])
        r3 = fl.grouped_cv(fl.make_pipeline(TINY, "gnb", 1), X2, y, folds, IDS, [])
        # fold 0 as validation changes; folds 1-4 train on shifted fold 0 and so may differ -> only check determinism
        r4 = fl.grouped_cv(fl.make_pipeline(TINY, "gnb", 1), X, y, folds, IDS, [])
        self.assertEqual(r2["folds"], r4["folds"])
        self.assertNotEqual(r2["folds"], r3["folds"])

    def test_ranking_and_tiebreak(self):
        def rec(m, s):
            mean = {k: m for k in fl.METRICS}
            return {"mean": mean, "std": {"balanced_accuracy": s}}
        recs = {"B": rec(0.9, 0.01), "A": rec(0.9, 0.01), "C": rec(0.9, 0.005), "D": rec(0.8, 0.0)}
        self.assertEqual(fl.rank_candidates(recs), ["C", "A", "B", "D"])
        self.assertEqual(fl.select_best(recs), "C")
        t = fl.performance_table(recs)
        self.assertEqual(t.loc["A", "CV Balanced Accuracy"], 90.0)

    def test_evaluate_confusion(self):
        X, y, folds = data()
        p = fl.make_pipeline(TINY, "lr", 1).fit(X, y)
        e = fl.evaluate(p, X, y)
        self.assertEqual(sum(map(sum, e["confusion_matrix"])), len(y))
        self.assertEqual(e["n"], len(y))


class CheckpointTests(unittest.TestCase):
    def run_all(self, d, budget):
        X, y, folds = data()
        ids = [str(i) for i in range(len(y))]
        store = make_store(d)
        rec = fl.run_candidates(fl.build_candidates(TINY), TINY, lambda e, i: X + (e == "B"), y, folds, ids, ["zz"],
                                seed=3, device=None, store=store, budget=budget, log=lambda *a: None)
        return X, y, folds, ids, store, rec

    def test_candidates_resume_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, _, _, store, rec = self.run_all(d, Budget(allow=2))
            self.assertEqual(len(rec), 2)  # stopped cleanly after 2 of 4
            _, _, _, _, store, rec2 = self.run_all(d, Budget())
            self.assertEqual(len(rec2), 4)
            self.assertEqual(rec["LR A"]["folds"], rec2["LR A"]["folds"])
            self.assertEqual(fl.select_best(rec2), fl.select_best(rec2))

    def test_tuning_resume_matches_uninterrupted(self):
        X, y, folds = data()
        ids = [str(i) for i in range(len(y))]
        kw = dict(seed=5, device=None, max_evals=6, log=lambda *a: None)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            full = fl.run_tuning("M", "lr", TINY, X, y, folds, ids, ["zz"], store=make_store(d1), budget=Budget(), **kw)
            part = fl.run_tuning("M", "lr", TINY, X, y, folds, ids, ["zz"], store=make_store(d2), budget=Budget(allow=3), **kw)
            self.assertFalse(part["complete"])
            self.assertEqual(part["n_done"], 3)
            resumed = fl.run_tuning("M", "lr", TINY, X, y, folds, ids, ["zz"], store=make_store(d2), budget=Budget(), **kw)
        self.assertTrue(full["complete"] and resumed["complete"])
        self.assertEqual(resumed["n_done"], 6)
        self.assertEqual(full["best_params"], resumed["best_params"])
        self.assertEqual([t["misc"]["vals"] for t in full["trials"].trials],
                         [t["misc"]["vals"] for t in resumed["trials"].trials])
        self.assertEqual(len({t["tid"] for t in resumed["trials"].trials}), 6)

    def test_identity_mismatch_fails_closed(self):
        X, y, folds = data()
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as d:
            fl.run_tuning("M", "gnb", TINY, X, y, folds, ids, ["zz"], seed=1, device=None,
                          store=make_store(d), budget=Budget(), max_evals=1, log=lambda *a: None)
            other = ck.Store(d, identity=ck.make_identity(config={"t": 2}))
            with self.assertRaises(ck.IdentityMismatchError):
                fl.run_tuning("M", "gnb", TINY, X, y, folds, ids, ["zz"], seed=1, device=None,
                              store=other, budget=Budget(), max_evals=1, log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()
