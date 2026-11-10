import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from notebooks import categorisation as ct
from notebooks import filtering as fl
from notebooks import system_pipeline as sp
from notebooks import taxonomy as tx
from notebooks.tests.test_system import NF, SUBCOL, TOPCOL, StubCat, StubFilter, feats_matrix

TAX = tx.default_taxonomy()
TC = "/technology and computing"
CS = "computer security"
SHA = "a" * 64


def watson(*paths):
    return repr({"categories": [{"label": p, "score": 0.5} for p in paths], "usage": {}})


# (relevant, predicted top, predicted sub, watson paths); group = index // 2
ROWS = [
    (True, [CS], [(CS, "network security")], [f"{TC}/{CS}/network security"]),
    (True, ["hardware"], [("hardware", "computer")], [f"{TC}/hardware/computer"]),
    (True, ["software"], [("software", "databases")], [f"{TC}/software/databases"]),
    (True, ["networking"], [], [f"{TC}/networking"]),
    (False, [], [], []),
    (False, ["software"], [], []),
    (True, [], [], []),                       # relevant but unmapped; the top categoriser predicts nothing
    (True, ["operating systems"], [("operating systems", "linux")], [f"{TC}/operating systems/linux"]),
]
COTOV = [("en", 1, [CS]), ("en", 0, []), ("fr", 1, [CS]), ("en", 1, ["hardware"]), ("en", 0, ["software"]), ("en", 1, [])]


def stub_features(rows):
    return feats_matrix([{"rel": 1 if r[0] else 0, "top": r[1], "sub": r[2]} for r in rows])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        ids = [f"r{i}" for i in range(len(ROWS))]
        df = pd.DataFrame({"_id": ids, "text": ["t"] * len(ROWS), "relevant": [r[0] for r in ROWS],
                           "watson": [watson(*r[3]) for r in ROWS]})
        df.iloc[:4].to_csv(self.tmp / "train.csv", index=False)   # contents irrelevant: verify_manifest is patched
        df.to_csv(self.tmp / "test.csv", index=False)
        manifest = pd.DataFrame({"row_id": ids, "group": [f"g{i // 2}" for i in range(len(ROWS))],
                                 "split": "test", "inner_fold": -1})
        manifest.attrs["metadata"] = {"columns": {"id": "_id", "text": "text", "relevant": "relevant", "watson": "watson"}}
        self.X = {i: x for i, x in zip(ids, stub_features(ROWS))}
        pd.DataFrame({"lang": [c[0] for c in COTOV], "full_text": ["x"] * len(COTOV),
                      "Related": [c[1] for c in COTOV]}).to_csv(self.tmp / "cotov.csv", index=False)
        crow = [{"rel": c[1], "top": c[2], "sub": []} for c in COTOV]
        cx = feats_matrix(crow)
        self.X.update({f"cotov-{i}": cx[i] for i in range(len(COTOV))})
        (self.tmp / "emb").mkdir()
        (self.tmp / "emb" / "cotov__E.json").write_text("{}")
        self.meta_sha = SHA
        self.verify = mock.Mock()
        for p in (mock.patch.object(sp.splits, "read_manifest", return_value=manifest),
                  mock.patch.object(sp.splits, "verify_manifest", self.verify),
                  mock.patch.object(ct, "load_deployed_categoriser", self._load_cat),
                  mock.patch.object(fl, "load_deployed_filter", self._load_filter)):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = sp.load_config("notebooks/system.toml")
        self.cfg["bootstrap"]["B"] = 20

    def _load_cat(self, path, level, allow_non_production=False):
        self.allow_np = allow_non_production
        if level in ("general", "one step"):
            cls, colmap = TAX.classes("top"), TOPCOL
        else:
            cls, colmap = TAX.classes(level), {c: SUBCOL[(level, c)] for c in TAX.classes(level)}
        return (StubCat(level, cls, colmap, embedder="E"),
                {"pickle_sha256": f"sha-{level}", "manifest_sha256": self.meta_sha, "embedder": {"name": "E", "fingerprint": {"sha": "e"}}})

    def _load_filter(self, path):
        return StubFilter(), {"pickle_sha256": "sha-filter", "manifest_sha256": self.meta_sha, "embedder": {"name": "E", "fingerprint": {"sha": "e"}}}

    def env(self, **kw):
        e = {"SPLIT_MANIFEST_CSV": "m.csv", "SPLIT_TRAIN_CSV": str(self.tmp / "train.csv"),
             "SPLIT_TEST_CSV": str(self.tmp / "test.csv"), "SPLIT_MANIFEST_SHA256": SHA,
             "EMBEDDINGS_DIR": str(self.tmp / "emb"), "FILTER_PICKLE": "f.pkl", "COTOV_CSV": str(self.tmp / "cotov.csv"),
             "SYSTEM_OUT_DIR": str(self.tmp / "out"), "OVERWRITE": "true"}
        e.update({v: f"{k}.pkl" for k, v in self.cfg["levels"].items()})
        e.update(kw)
        return {k: v for k, v in e.items() if v is not None}

    def ff(self, system, s, corpus, row_ids, **kw):
        self.kw = kw

        def f(embedder, ids):
            return np.stack([self.X[i] for i in ids])
        return f

    def run_(self, cfg=None, **kw):
        cfg = cfg or self.cfg
        s = sp.settings_from_env(self.env(**kw), cfg)
        return sp.run(s, sp.effective_config(cfg, s.smoke), features_factory=self.ff, log=lambda m: None), s

    def out(self):
        return self.tmp / "out"


class EndToEnd(Base):
    def test_outputs_and_contents(self):
        doc, s = self.run_()
        files = sorted(p.name for p in self.out().iterdir())
        self.assertEqual(files, ["system_performance.json", "system_predictions.csv"])   # no .tmp left
        text = (self.out() / "system_performance.json").read_text()
        d = json.loads(text, parse_constant=lambda c: self.fail(f"non-finite {c}"))
        self.assertEqual(d["status"], "complete")
        self.assertTrue(d["production"])
        self.assertEqual((d["config"]["B"], d["config"]["seed"]), (20, 20241102))
        self.assertEqual(d["manifest_sha256"], SHA)
        self.assertEqual(d["taxonomy_sha256"], TAX.sha256)
        self.assertIn("textprep_version", d)
        self.assertEqual(d["components"]["filter"]["pickle_sha256"], "sha-filter")
        self.assertEqual(d["components"]["general"]["sidecar"], "general.pkl.json")
        self.assertEqual(d["components"]["one step"]["embedder"], "E")
        self.assertEqual(d["components"]["hardware"]["fingerprint"], {"sha": "e"})
        self.assertEqual(d["cotov"]["status"], "complete")
        self.assertEqual(d["cotov"]["n"], 5)          # English rows only
        self.assertIn("filter_decision", d["cotov"]["systems"]["two_step"])
        self.assertNotIn("filter_decision", d["cotov"]["systems"]["one_step"])
        self.assertIn("two_step_minus_one_step", d["evaluation"]["paired"])
        self.assertIn("top:two_step", d["evaluation"]["oracle"])
        self.assertIn("top:one_step", d["evaluation"]["oracle"])
        pred = pd.read_csv(self.out() / "system_predictions.csv", keep_default_na=False)
        self.assertEqual(len(pred), len(ROWS))
        self.assertEqual(pred.loc[0, "two_step"], f"{CS}|network security")
        self.assertEqual(pred.loc[0, "gold"], f"{CS}|network security")
        self.assertEqual(pred.loc[4, "two_step"], "other")      # filter reject
        self.assertEqual(pred.loc[4, "two_step_filter_reject"], 1)
        self.assertEqual(pred.loc[6, "two_step_empty_top_fallback"], 1)
        self.assertEqual(pred.loc[5, "one_step"], "software")    # One-Step has no filter
        self.assertEqual(list(pred["group"][:3]), ["g0", "g0", "g1"])
        self.assertIn("predictions_sha256", d["outputs"])
        self.assertTrue(any("micro P" in ln for ln in sp.report_lines(doc)))
        self.assertFalse(self.allow_np)

    def test_split_verified_with_pins(self):
        self.run_()
        _, kw = self.verify.call_args
        self.assertEqual((kw["expected_seed"], kw["expected_manifest_sha256"]), (42, SHA))

    def test_failure_leaves_no_final_outputs(self):
        with mock.patch.object(sp.se, "evaluate_systems", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.run_()
        self.assertFalse(self.out().exists() and any(self.out().iterdir()))

    def test_failure_while_staging_leaves_nothing(self):
        with mock.patch.object(sp, "json_safe", side_effect=ValueError("nan")):
            with self.assertRaises(ValueError):
                self.run_()
        self.assertEqual(list(self.out().iterdir()), [])

    def test_overwrite_policy(self):
        self.run_()
        with self.assertRaises(FileExistsError):
            self.run_(OVERWRITE="false")
        self.run_(OVERWRITE="true")

    def test_json_safe_nan(self):
        self.assertEqual(sp.json_safe({"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3), "d": [np.nan]}),
                         {"a": None, "b": None, "c": 3, "d": [None]})


class ManifestPin(Base):
    def test_every_component_must_match(self):
        for comp in ("filter", "general", "one step", "hardware"):
            real_cat, real_f = self._load_cat, self._load_filter

            def cat(path, level, allow_non_production=False, comp=comp):
                c, m = real_cat(path, level, allow_non_production)
                if level == comp:
                    m["manifest_sha256"] = "b" * 64
                return c, m

            def flt(path, comp=comp):
                c, m = real_f(path)
                if comp == "filter":
                    m.pop("manifest_sha256")      # missing key
                return c, m
            with mock.patch.object(ct, "load_deployed_categoriser", cat), mock.patch.object(fl, "load_deployed_filter", flt):
                with self.assertRaises(sp.SystemPipelineError) as c:
                    self.run_()
            self.assertIn(comp, str(c.exception))
            self.assertFalse(self.out().exists() and any(self.out().iterdir()))

    def test_match_recorded(self):
        doc, _ = self.run_()
        self.assertTrue(all(c["manifest_sha256"] == SHA for c in doc["components"].values()))
        self.assertEqual(doc["components"]["general"]["level"], "general")
        self.assertEqual(sorted(doc["components"]), sorted(["filter", "general", "one step", *TAX.parents()]))


class Publish(Base):
    def test_failed_summary_rename_leaves_nothing_and_clears_old(self):
        self.run_()                                     # an old complete set exists
        real, calls = sp.os.replace, []

        def replace(a, b):
            calls.append(b)
            if len(calls) == 2:
                raise OSError("disk")
            return real(a, b)
        with mock.patch.object(sp.os, "replace", replace):
            with self.assertRaises(OSError):
                self.run_()
        self.assertEqual(list(self.out().iterdir()), [])

    def test_old_summary_removed_before_rename(self):
        self.run_()
        order = []
        real = sp.os.replace
        with mock.patch.object(sp.os, "replace", lambda a, b: (order.append((Path(b).name, (self.out() / "system_performance.json").exists())), real(a, b))[1]):
            self.run_()
        self.assertEqual(order[0], ("system_predictions.csv", False))     # no stale summary beside new predictions


class Values(Base):
    """Hand-computed cases on the stub rows."""
    def setUp(self):
        super().setUp()
        self.doc, _ = self.run_()

    def test_cotov_implicit_relevance_values(self):
        # English cotov rows: Related 1,0,1,0,1 ; tops [CS],[],[hw],[sw],[] ; filter accepts by feature col0 (= Related)
        cv = self.doc["cotov"]
        self.assertEqual(cv["n"], 5)
        two, one = cv["systems"]["two_step"], cv["systems"]["one_step"]
        # filter accept == Related: perfect
        self.assertEqual(two["filter_decision"]["confusion_matrix"], [[2, 0], [0, 3]])
        self.assertIn("roc_auc", two["filter_decision"])
        # Two-Step: accept AND mapped: rows (Related, top) -> (1,CS) yes, (0,-) no, (1,hw) yes, (0,sw) no, (1,-) no
        self.assertEqual(two["implicit_relevance"]["confusion_matrix"], [[2, 0], [1, 2]])
        self.assertNotIn("roc_auc", two["implicit_relevance"])
        self.assertNotIn("roc_auc", two["implicit_relevance"]["bootstrap"]["ci"])
        self.assertEqual(two["implicit_relevance"]["score"], "none (decision metrics only)")
        # One-Step: mapped only: 1,0,1,1,0
        self.assertEqual(one["implicit_relevance"]["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertIn("roc_auc", one["implicit_relevance"])
        self.assertEqual(one["implicit_relevance"]["score"], "maximum mapped top-class score")
        self.assertNotIn("filter_decision", one)
        self.assertEqual(two["filter_decision"]["bootstrap"]["unit"], "row")
        self.assertEqual(one["implicit_relevance"]["bootstrap"]["n_units"], 5)

    def test_oracle_keys_and_sizes(self):
        o = self.doc["evaluation"]["oracle"]
        subs = [k for k in o if k.startswith("sub:")]
        self.assertEqual(sorted(subs), sorted(f"sub:{p}" for p in [CS, "hardware", "operating systems", "software"]))
        self.assertFalse([k for k in o if k.startswith("sub:") and k.endswith(("two_step", "one_step"))])
        self.assertEqual(sum(k.startswith("top:") for k in o), 2)

    def test_routing_values(self):
        r = self.doc["routing"]
        self.assertEqual(r["two_step"]["filter_reject"], 2)       # rows 4 and 5 are irrelevant by feature col0

    def test_rejects_non_test_row_and_duplicates(self):
        manifest = sp.splits.read_manifest()
        m = manifest.copy()
        m.attrs = manifest.attrs
        m.loc[2, "split"] = "train"
        s = sp.settings_from_env(self.env(), self.cfg)
        with mock.patch.object(sp.splits, "read_manifest", return_value=m):
            with self.assertRaises(sp.SystemPipelineError):
                sp.load_test_rows(s)
        df = pd.read_csv(self.tmp / "test.csv")
        pd.concat([df, df.iloc[:1]]).to_csv(self.tmp / "test.csv", index=False)
        with self.assertRaises(sp.SystemPipelineError):
            sp.load_test_rows(s)


class Smoke(Base):
    def test_smoke_writes_only_smoke_names(self):
        doc, s = self.run_(SYSTEM_SMOKE="1", SYSTEM_SMOKE_OUT_DIR=str(self.tmp / "smk"))
        self.assertEqual(sorted(p.name for p in (self.tmp / "smk").iterdir()),
                         ["smoke_system_performance.json", "smoke_system_predictions.csv"])
        self.assertFalse(self.out().exists())
        self.assertTrue(doc["non_production"] and not doc["production"])
        self.assertEqual(doc["config"]["B"], 50)
        self.assertTrue(self.allow_np)

    def test_smoke_refuses_production_dir(self):
        s = sp.settings_from_env(self.env(SYSTEM_SMOKE="1", SYSTEM_SMOKE_OUT_DIR=str(self.out())), self.cfg)
        with self.assertRaises(sp.SystemPipelineError):
            sp.output_paths(s)

    def test_refuse_production_output(self):
        with self.assertRaises(sp.SystemPipelineError):
            sp.refuse_production_output(self.tmp / "x" / "system_performance.json", self.out() / "system_performance.json")
        sp.refuse_production_output(self.tmp / "x" / "smoke_system_performance.json", self.out() / "system_performance.json")

    def test_smoke_may_skip_cotov(self):
        doc, _ = self.run_(SYSTEM_SMOKE="1", SYSTEM_SMOKE_OUT_DIR=str(self.tmp / "smk"), COTOV_CSV=None)
        self.assertEqual(doc["status"], "complete-cotov-blocked")


class CotovPolicy(Base):
    def test_missing_cotov_fails_production(self):
        with self.assertRaises(fl.FilterError):
            self.run_(COTOV_CSV=None)
        self.assertFalse(self.out().exists() and any(self.out().iterdir()))

    def test_missing_artifact_fails_production(self):
        (self.tmp / "emb" / "cotov__E.json").unlink()
        with self.assertRaises(fl.FilterError):
            self.run_()

    def test_override_records_blocked(self):
        doc, _ = self.run_(COTOV_CSV=None, SYSTEM_ALLOW_NO_COTOV="1")
        d = json.loads((self.out() / "system_performance.json").read_text())
        self.assertEqual(d["status"], "complete-cotov-blocked")
        self.assertEqual(d["cotov"]["status"], "blocked")
        self.assertTrue(d["production"])
        self.assertTrue(any(ln.startswith("BLOCKED:") for ln in sp.report_lines(doc)))


class Settings(Base):
    def test_missing_env_names_the_variable(self):
        with self.assertRaises(sp.SystemPipelineError) as c:
            sp.settings_from_env(self.env(SPLIT_MANIFEST_SHA256=None), self.cfg)
        self.assertIn("SPLIT_MANIFEST_SHA256", str(c.exception))

    def test_config_is_precommitted(self):
        cfg = sp.load_config("notebooks/system.toml")
        self.assertEqual((cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"], cfg["bootstrap"]["level"]), (10000, 20241102, 0.95))
        self.assertEqual(cfg["bootstrap"]["compare"], [["two_step", "one_step"]])
        self.assertEqual(cfg["smoke"]["B"], 50)
        sp.validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
