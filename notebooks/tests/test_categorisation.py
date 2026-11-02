import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks import categorisation as ct
from notebooks import checkpoints as ck
from notebooks import filtering as fl

CFG = ct.load_config("notebooks/categorisation.toml")
SEED = 7
TINY = {
    "levels": {
        "general": {"kind": "top", "scheme": "two_step"},
        "computer security": {"kind": "sub", "parent": "computer security", "scheme": "two_step"},
    },
    "selection": {"levels": ["general", "computer security"], "embedders": ["A", "B"],
                  "methods": ["BR", "PSt"], "classifiers": ["lr"], "cv_folds": 5, "min_labelset_count": 4},
    "bootstrap": {"B": 20, "seed": 1, "level": 0.95},
    "smoke": {"embedders": ["A"], "bootstrap_B": 5},
    "methods": {
        "BR": {"class": "MLC.BRClassifier", "params": {}},
        "PSt": {"class": "MLC.PStClassifier", "params": {}, "grid": {"pruning_value": [4, 10000], "max_reintroduced": [1]},
                "feasible_if_labelset_count_reaches": "pruning_value"},
    },
    "classifiers": {"lr": {"estimator": "sklearn.linear_model.LogisticRegression",
                           "params": {"solver": "liblinear", "max_iter": 200}, "one_vs_rest": True}},
}

CS = "/technology and computing/computer security"
TOPICS = [  # (watson category labels, relevant)
    ([CS + "/antivirus and malware"], True),
    ([CS + "/network security"], True),
    (["/technology and computing/software"], True),
    (["/technology and computing/hardware"], True),
    ([CS], True),
    (["/sports/football"], True),          # relevant, no mapped top category -> ["other"]
    ([CS + "/antivirus and malware"], False),  # irrelevant: never in any population
]


def watson(paths):
    return str({"categories": [{"label": p, "score": 0.9} for p in paths]})


def corpus(n_train=150, n_test=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_train + n_test):
        t = int(rng.integers(0, len(TOPICS)))
        paths, rel = TOPICS[t]
        rows.append({"_id": f"r{i:04d}", "relevant": rel, "watson": watson(paths), "topic": t,
                     "split": "train" if i < n_train else "test", "group": f"g{i // 2}"})
    df = pd.DataFrame(rows)
    tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    # groups never cross the split (n_train is even)
    folds = {r: k % 5 for k, r in enumerate(sorted(set(tr.group)))}
    manifest = pd.DataFrame({"row_id": df["_id"], "group": df["group"], "split": df["split"],
                             "inner_fold": [folds[g] if s == "train" else -1 for g, s in zip(df.group, df.split)]})
    return tr, te, manifest


def make_features(train_df, test_df, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    allr = pd.concat([train_df, test_df])
    mat = {r: rng.normal(size=dim) + np.eye(len(TOPICS), dim, k=0)[t % len(TOPICS)] * 3
           for r, t in zip(allr["_id"], allr["topic"])}
    calls = []

    def features(embedder, ids):
        calls.append((embedder, list(ids)))
        off = 0.0 if embedder == "A" else 0.5
        return np.array([mat[i] + off for i in ids])

    features.calls = calls
    return features


class Budget:
    def __init__(self, allow=10 ** 9):
        self.n = allow
        self.fits = 0

    def allows(self, estimate=None, kind=None):
        return self.n > 0

    def record(self, kind, seconds):
        self.n -= 1
        self.fits += 1


def make_store(d):
    return ck.Store(d, identity=ck.make_identity(config={"t": 1}))


def quiet(*a, **k):
    pass


class ConfigTests(unittest.TestCase):
    def test_production_config(self):
        sel = CFG["selection"]
        self.assertEqual(sel["levels"], ["general", "computer security", "hardware", "operating systems", "software"])
        self.assertEqual(len(sel["embedders"]), 7)
        self.assertEqual(sel["embedders"], fl.load_config("notebooks/filter.toml")["selection"]["embedders"])
        self.assertEqual((sel["min_labelset_count"], sel["cv_folds"]), (20, 5))
        cands = ct.build_candidates(CFG)
        self.assertEqual(len(cands), 13 * 3)  # 7 methods + 6 PSt grid points, x lr/dt/rf
        self.assertEqual(cands[0].name, "BR[LR]")
        self.assertIn("PSt(pruning_value=25,max_reintroduced=1)[LR]", [c.name for c in cands])
        for lvl in sel["levels"]:
            self.assertIn(lvl, CFG["levels"])
        self.assertEqual(CFG["bootstrap"]["B"], 1000)

    def test_every_candidate_builds_with_fixed_params(self):
        for c in ct.build_candidates(CFG):
            p = ct.make_pipeline(CFG, c, 42)
            self.assertEqual(p.steps[0][0], "scaler")
        lr = ct.make_classifier(CFG, "lr", 42)
        self.assertEqual(type(lr).__name__, "OneVsRestClassifier")
        self.assertEqual((lr.estimator.solver, lr.estimator.max_iter, lr.estimator.random_state), ("liblinear", 10000, 42))
        self.assertEqual(ct.make_classifier(CFG, "rf", 42).random_state, 42)
        mbr = [c for c in ct.build_candidates(CFG) if c.method == "MBR"][0]
        m = ct.make_pipeline(CFG, mbr, 42).named_steps["mlc"]
        self.assertEqual((m.cv, m.random_state), (5, 42))

    def test_smoke_overrides(self):
        e = ct.effective_config(CFG, True)
        self.assertTrue(e["smoke"])
        self.assertLess(len(ct.build_candidates(e)), len(ct.build_candidates(CFG)))
        self.assertEqual(e["bootstrap"]["B"], 50)
        self.assertFalse(ct.effective_config(CFG)["smoke"])


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.te, self.man = corpus()

    def test_general_is_relevant_only(self):
        d = ct.prepare_level(self.tr, self.te, self.man, "general", TINY)
        exp = self.tr.loc[self.tr.relevant, "_id"].tolist()
        self.assertEqual(d.train_ids, exp)
        self.assertEqual(d.Y_train.shape[1], 6)
        self.assertEqual(d.classes[4], "other")
        i = d.train_ids.index(self.tr.loc[self.tr.topic == 5, "_id"].iloc[0])
        self.assertEqual(d.labelsets_train[i], ("other",))

    def test_parent_level_uses_gold_parent(self):
        d = ct.prepare_level(self.tr, self.te, self.man, "computer security", TINY)
        exp = self.tr.loc[self.tr.relevant & self.tr.topic.isin([0, 1, 4]), "_id"].tolist()
        self.assertEqual(d.train_ids, exp)
        self.assertEqual(d.classes, ("antivirus and malware", "network security", "other"))
        self.assertEqual(set(d.labelsets_train), {("antivirus and malware",), ("network security",), ("other",)})

    def test_relevance_is_strict(self):
        tr = self.tr.copy()
        tr["relevant"] = tr["relevant"].astype(str)
        with self.assertRaises(TypeError):
            ct.prepare_level(tr, self.te, self.man, "general", TINY)

    def test_manifest_binding(self):
        bad = pd.concat([self.tr, self.te.iloc[:1]])
        with self.assertRaises(ct.LeakageError):
            ct.prepare_level(bad, self.te, self.man, "general", TINY)
        man = self.man.copy()
        man.loc[man.row_id == self.tr["_id"].iloc[0], "split"] = "test"
        with self.assertRaises(ct.CategorisationError):
            ct.prepare_level(self.tr, self.te, man, "general", TINY)
        cfg = json.loads(json.dumps(TINY))
        cfg["selection"]["cv_folds"] = 4
        with self.assertRaises(ct.CategorisationError):
            ct.prepare_level(self.tr, self.te, self.man, "general", cfg)


def handmade(labelsets, folds, classes=("a", "b", "c")):
    n = len(labelsets)
    ids = [f"t{i}" for i in range(n)]
    return ct.LevelData("lvl", tuple(classes), ids, ct._labels_matrix(labelsets, classes), list(labelsets),
                        np.asarray(folds), np.array(ids), ["x0"], ct._labels_matrix([("a",)], classes), [("a",)],
                        np.array(["gx"]))


class PartitionTests(unittest.TestCase):
    def test_rare_labelsets_from_fitting_rows_only(self):
        # ("b",) has 4 rows in total (frequent on all train) but 3 of them are in fold 0, so in fold 0's
        # fitting partition it has 1 row (rare there) and the folds 1-4 partitions hold >= 3.
        ls = [("a",)] * 20 + [("b",)] * 4
        folds = [i % 5 for i in range(20)] + [0, 0, 0, 1]
        d = handmade(ls, folds)
        plans = {p.fold: p for p in ct.plan_folds(d, 3)}
        fit0 = {d.labelsets_train[i] for i in plans[0].fit_idx}
        self.assertEqual(fit0, {("a",)})
        self.assertIn(("b",), {d.labelsets_train[i] for i in plans[1].fit_idx})
        # validation rows: all eligible rows of the held-out fold, incl. the rare labelsets
        self.assertEqual(sorted(plans[0].val_idx.tolist()), [i for i, f in enumerate(folds) if f == 0])
        self.assertEqual(plans[0].n_partition, folds.count(1) + folds.count(2) + folds.count(3) + folds.count(4))
        # the whole-train filter (refit) keeps ("b",)
        self.assertEqual(len(ct.frequent_rows(d.labelsets_train, np.arange(24), 3)), 24)

    def test_fixed_columns_with_absent_class(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 3))
        Y = np.zeros((30, 3), dtype=int)
        Y[:15, 0] = 1
        Y[15:, 1] = 1  # class c never positive
        cand = ct.build_candidates(TINY)[0]
        m = ct.fit_candidate(TINY, cand, X, Y, np.arange(30), 0)
        self.assertEqual(m.active.tolist(), [0, 1])
        self.assertEqual(m.predict(X).shape, (30, 3))
        self.assertTrue((m.predict(X)[:, 2] == 0).all() and (m.predict_proba(X)[:, 2] == 0).all())


class OOFTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.te, self.man = corpus()
        self.data = ct.prepare_level(self.tr, self.te, self.man, "general", TINY)
        self.feat = make_features(self.tr, self.te)
        self.cands = ct.build_candidates(TINY)

    def run_oof(self, store, budget, emb="A"):
        return ct.run_oof(self.data, emb, self.cands, TINY, self.feat, self.data.test_ids, seed=SEED, store=store,
                          budget=budget, log=quiet)

    def test_exact_coverage_and_infeasible(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(d)
            run = self.run_oof(store, Budget())
            self.assertTrue(run.complete)
            ok = run.records["BR[LR]"]
            self.assertEqual(ok["status"], ct.OK)
            self.assertEqual(ok["pred"].shape, self.data.Y_train.shape)
            seen = []
            for f in range(5):
                rec = store.load(ct.oof_key("general", "A", "BR[LR]", f))
                seen += rec["val_ids"]
            self.assertEqual(sorted(seen), sorted(self.data.train_ids))
            self.assertEqual(len(seen), len(set(seen)))
            bad = "PSt(pruning_value=10000,max_reintroduced=1)[LR]"
            self.assertEqual(run.records[bad]["status"], ct.INFEASIBLE)
            self.assertEqual(run.records["PSt(pruning_value=4,max_reintroduced=1)[LR]"]["status"], ct.OK)
            # every fold of the infeasible candidate is checkpointed; test ids never reached the features
            ck.verify_complete(store, ct.expected_keys(self.data, ["A"], self.cands))
            for _, ids in self.feat.calls:
                self.assertFalse(set(ids) & set(self.data.test_ids))
            run_b = self.run_oof(store, Budget(), "B")
            sel = ct.select(self.data, {"A": run.records, "B": run_b.records}, [c.name for c in self.cands], ["A", "B"])
            self.assertNotIn(bad, sel.label_candidates)
            self.assertEqual(sel.excluded["A"], {bad: ct.INFEASIBLE})

    def test_leakage_error_on_test_ids(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ct.LeakageError):
                ct.run_oof(self.data, "A", self.cands, TINY, self.feat, self.data.train_ids[:1], seed=SEED,
                           store=make_store(d), budget=Budget(), log=quiet)
            leaky = ct.LevelData(**{**self.data.__dict__, "train_ids": self.data.train_ids[:-1] + [self.data.test_ids[0]]})
            with self.assertRaises(ct.LeakageError):
                ct.run_oof(leaky, "A", self.cands, TINY, self.feat, self.data.test_ids, seed=SEED,
                           store=make_store(d), budget=Budget(), log=quiet)
        self.assertTrue(issubclass(ct.LeakageError, fl.LeakageError))

    def test_checkpoint_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            full = self.run_oof(make_store(d1), Budget())
            s2 = make_store(d2)
            b = Budget(allow=3)
            part = self.run_oof(s2, b)
            self.assertFalse(part.complete)
            self.assertTrue(part.missing)
            b2 = Budget()
            res = self.run_oof(s2, b2)
            self.assertTrue(res.complete)
            self.assertEqual(b.fits + b2.fits, 2 * 5)  # BR + feasible PSt, 5 folds; infeasible costs no fit
            for name, rec in full.records.items():
                self.assertEqual(res.records[name]["status"], rec["status"])
                if rec["status"] == ct.OK:
                    np.testing.assert_array_equal(res.records[name]["pred"], rec["pred"])
                    np.testing.assert_allclose(res.records[name]["score"], rec["score"])

    def _failing_fit(self, fail_name, fold_calls):
        real = ct.fit_candidate

        def fit(cfg, cand, X, Y, fit_idx, seed):
            if cand.name == fail_name:
                fold_calls.append(len(fit_idx))
                raise ValueError("boom")
            return real(cfg, cand, X, Y, fit_idx, seed)
        return fit

    def test_failure_skips_remaining_folds_and_is_recorded_once(self):
        calls = []
        real = ct.fit_candidate
        ct.fit_candidate = self._failing_fit("BR[LR]", calls)
        try:
            with tempfile.TemporaryDirectory() as d:
                store = make_store(d)
                run = self.run_oof(store, Budget())
                self.assertEqual(len(calls), 1)  # only the first fold was fitted
                self.assertTrue(run.complete)
                rec0 = run.records["BR[LR]"]
                self.assertEqual(rec0["status"], ct.FAILED)
                self.assertEqual(list(rec0["reason"]), [0])  # failure reason recorded once
                ck.verify_complete(store, ct.expected_keys(self.data, ["A"], self.cands))
                self.assertTrue(store.load(ct.oof_key("general", "A", "BR[LR]", 3))["skipped"])
                # resume: no further fit attempts
                run2 = self.run_oof(store, Budget())
                self.assertEqual(len(calls), 1)
                self.assertEqual(run2.records["BR[LR]"]["status"], ct.FAILED)
        finally:
            ct.fit_candidate = real

    def test_resume_validates_val_ids_of_failed_and_infeasible_records(self):
        plan0 = ct.plan_folds(self.data, int(TINY["selection"]["min_labelset_count"]))[0]
        wrong = [self.data.train_ids[i] for i in plan0.val_idx][::-1]
        for name, status in (("BR[LR]", ct.FAILED), ("PSt(pruning_value=10000,max_reintroduced=1)[LR]", ct.INFEASIBLE)):
            with tempfile.TemporaryDirectory() as d:
                store = make_store(d)
                store.save(ct.oof_key("general", "A", name, 0),
                           {"status": status, "reason": "x", "classes": list(self.data.classes), "val_ids": wrong})
                with self.assertRaises(ct.CategorisationError):
                    self.run_oof(store, Budget())


def rec(pred):
    pred = np.asarray(pred)
    return {"status": ct.OK, "pred": pred, "score": pred.astype(float)}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.d = handmade([("a",), ("b",), ("a", "b"), ("c",)], [0, 1, 2, 3])
        Y = self.d.Y_train
        good_a = Y.copy(); good_a[:, 1:] = 0
        good_b = Y.copy(); good_b[:, [0, 2]] = 0
        self.recs = {"X": rec(good_a), "Y": rec(good_b), "Z": rec(good_a)}

    def test_per_label_pick_and_tie_break(self):
        s = ct.select(self.d, {"E1": self.recs}, ["X", "Y", "Z"], ["E1"])
        # a: X and Z tie -> X (earlier); b: Y; c: all 0 -> earliest (X)
        self.assertEqual(s.label_candidates, ("X", "Y", "X"))
        s2 = ct.select(self.d, {"E1": self.recs}, ["Z", "Y", "X"], ["E1"])
        self.assertEqual(s2.label_candidates, ("Z", "Y", "Z"))
        self.assertAlmostEqual(s.embedder_scores["E1"], 2 / 3)
        self.assertEqual(len(s.table()), 3)
        self.assertIn("selection", s.table().attrs["note"])

    def test_embedder_pick_and_tie(self):
        worse = {k: rec(np.zeros_like(self.d.Y_train)) for k in self.recs}
        self.assertEqual(ct.select(self.d, {"E1": worse, "E2": self.recs}, list(self.recs), ["E1", "E2"]).embedder, "E2")
        self.assertEqual(ct.select(self.d, {"E1": self.recs, "E2": self.recs}, list(self.recs), ["E2", "E1"]).embedder, "E2")

    def test_incomplete_and_failed(self):
        with self.assertRaises(ct.CategorisationError):
            ct.select(self.d, {"E1": {"X": self.recs["X"]}}, ["X", "Y"], ["E1"])
        recs = {**self.recs, "Y": {"status": ct.FAILED, "reason": "boom"}}
        with self.assertRaises(ct.CategorisationError):
            ct.select(self.d, {"E1": recs}, ["X", "Y", "Z"], ["E1"])
        s = ct.select(self.d, {"E1": recs}, ["X", "Y", "Z"], ["E1"], allow_failed=True)
        self.assertNotIn("Y", s.label_candidates)
        with self.assertRaises(ct.LeakageError):
            ct.select(self.d, {"E1": self.recs}, ["X", "Y", "Z"], ["E1"], forbidden_ids=["t0"])


class MetricsTests(unittest.TestCase):
    def test_matches_sklearn(self):
        from sklearn.metrics import f1_score, hamming_loss, accuracy_score, roc_auc_score
        rng = np.random.default_rng(3)
        Y = rng.integers(0, 2, (60, 4)); Y[:, 3] = 0
        P = rng.integers(0, 2, (60, 4))
        S = rng.random((60, 4))
        m = ct.multilabel_metrics(Y, P, S, list("abcd"))
        for avg in ("micro", "macro", "weighted"):
            self.assertAlmostEqual(m[f"{avg}_f1"], f1_score(Y, P, average=avg, zero_division=0))
        self.assertAlmostEqual(m["hamming_loss"], hamming_loss(Y, P))
        self.assertAlmostEqual(m["subset_accuracy"], accuracy_score(Y, P))
        self.assertTrue(np.isnan(m["labels"]["d"]["roc_auc"]))
        self.assertAlmostEqual(m["macro_roc_auc"], np.mean([roc_auc_score(Y[:, j], S[:, j]) for j in range(3)]))
        np.testing.assert_allclose(ct.label_f1(Y, P), f1_score(Y, P, average=None, zero_division=0))

    def test_group_bootstrap_resamples_whole_groups(self):
        Y = np.array([[1, 0]] * 4)
        P = np.array([[1, 0]] * 3 + [[0, 1]])  # group A (3 rows) right, group B (1 row) wrong
        groups = ["A", "A", "A", "B"]
        b = ct.group_bootstrap(Y, P, P.astype(float), groups, ["x", "y"], B=200, seed=5)
        self.assertEqual((b["unit"], b["n_units"]), ("group", 2))
        lo, hi = b["ci"]["subset_accuracy"]
        self.assertTrue(lo in (0.0, 0.75, 1.0) and hi in (0.0, 0.75, 1.0))
        self.assertEqual(b["n_replicates"]["macro_roc_auc"], 0)  # AUC undefined: skipped, not failed
        again = ct.group_bootstrap(Y, P, P.astype(float), groups, ["x", "y"], B=200, seed=5)
        self.assertEqual(json.dumps(b, sort_keys=True), json.dumps(again, sort_keys=True))  # seeded (NaN-safe)


class RefitPublishTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.te, self.man = corpus()
        self.data = ct.prepare_level(self.tr, self.te, self.man, "general", TINY)
        self.feat = make_features(self.tr, self.te)
        cands = ct.build_candidates(TINY)
        names = [c.name for c in cands]
        self.tmp = tempfile.TemporaryDirectory()
        store = make_store(self.tmp.name)
        oof = {e: ct.run_oof(self.data, e, cands, TINY, self.feat, self.data.test_ids, seed=SEED, store=store,
                             budget=Budget(), log=quiet).records for e in ("A", "B")}
        self.sel = ct.select(self.data, oof, names, ["A", "B"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_refit_on_all_eligible_train_and_evaluate(self):
        self.feat.calls.clear()
        cat = ct.refit(self.data, self.sel, TINY, self.feat, self.data.test_ids, seed=SEED)
        self.assertEqual(self.feat.calls, [(self.sel.embedder, self.data.train_ids)])
        exp = len(ct.frequent_rows(self.data.labelsets_train, np.arange(len(self.data.train_ids)), 4))
        self.assertEqual((cat.info["n_train"], cat.info["n_fit"]), (len(self.data.train_ids), exp))
        self.assertEqual(set(cat.models), set(self.sel.label_candidates))
        res = ct.evaluate_test(cat, self.data, TINY, self.feat, B=10, seed=1)
        self.assertEqual(res["all"]["n"], len(self.data.test_ids))
        self.assertLessEqual(res["thesis_comparable"]["n"], res["all"]["n"])
        self.assertEqual(list(res["all"]["labels"]), list(self.data.classes))
        self.assertIn("bootstrap", res["all"])
        with self.assertRaises(ct.LeakageError):
            ct.refit(self.data, self.sel, TINY, self.feat, self.data.train_ids[:2], seed=SEED)

    def test_publish_and_load_fail_closed(self):
        cat = ct.refit(self.data, self.sel, TINY, self.feat, self.data.test_ids, seed=SEED)
        with tempfile.TemporaryDirectory() as d:
            pk, csv = Path(d) / "general.pkl", Path(d) / "general.csv"
            with self.assertRaises(ct.CategorisationError):
                ct.load_deployed_categoriser(pk, "general")  # nothing published
            sha = ct.stage_categoriser(cat, pk)
            Path(ct.staged(csv)).write_text("x")
            self.assertFalse(pk.exists())
            doc = ct.publish_categoriser(pk, {"level": "general", "pickle_sha256": sha}, [csv])
            self.assertEqual(doc["status"], "complete")
            obj, meta = ct.load_deployed_categoriser(pk, "general")
            X = self.feat("A", self.data.test_ids)
            np.testing.assert_array_equal(obj.predict(X), cat.predict(X))
            self.assertEqual(obj.predict_proba(X).shape, (len(X), 6))
            with self.assertRaises(ct.CategorisationError):
                ct.load_deployed_categoriser(pk, "computer security")
            side = Path(str(pk) + ".json")
            good = side.read_text()
            side.write_text(json.dumps({**json.loads(good), "status": "incomplete"}))
            with self.assertRaises(ct.CategorisationError):
                ct.load_deployed_categoriser(pk, "general")
            side.write_text(good)
            pk.write_bytes(pk.read_bytes() + b"x")
            with self.assertRaises(ct.CategorisationError):
                ct.load_deployed_categoriser(pk, "general")
            # smoke: refuses production names, and is not loadable as production
            sp = Path(d) / "smoke" / "general.pkl"
            sp.parent.mkdir()
            sha = ct.stage_categoriser(cat, sp)
            with self.assertRaises(ct.CategorisationError):
                ct.publish_categoriser(sp, {"level": "general", "pickle_sha256": sha}, smoke=True, production_paths=[pk])
            ok = Path(d) / "smoke" / "smoke-general.pkl"
            sha = ct.stage_categoriser(cat, ok)
            ct.publish_categoriser(ok, {"level": "general", "pickle_sha256": sha}, smoke=True, production_paths=[pk])
            with self.assertRaises(ct.CategorisationError):
                ct.load_deployed_categoriser(ok, "general")
            ct.load_deployed_categoriser(ok, "general", allow_non_production=True)
            # wrong sha in the sidecar is refused before anything is published
            bad = Path(d) / "bad.pkl"
            ct.stage_categoriser(cat, bad)
            with self.assertRaises(ct.CategorisationError):
                ct.publish_categoriser(bad, {"level": "general", "pickle_sha256": "0" * 64})
            self.assertFalse(bad.exists())


if __name__ == "__main__":
    unittest.main()
