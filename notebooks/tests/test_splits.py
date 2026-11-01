import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks import splits as sp

TC = "/technology and computing"
PATHS = [
    [],
    [f"{TC}/computer security/antivirus and malware"],
    [f"{TC}/computer security/network security"],
    [f"{TC}/computer security"],
    [f"{TC}/hardware/computer/servers"],
    [f"{TC}/operating systems/windows", f"{TC}/software/databases"],
    [f"{TC}/networking/vpn and remote access"],
    ["/sports/football"],
]


def corpus(n_groups=300, seed=0):
    """Synthetic corpus: groups of 1-4 copies (same text up to URLs, mentions, case and
    punctuation), mixed relevance and Watson paths, some groups with conflicting labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        base = f"Post number {g} about topic {g % 7}"
        relevant = bool(rng.random() < 0.6)
        paths = PATHS[rng.integers(len(PATHS))]
        for c in range(int(rng.integers(1, 5))):
            text = [base, f"@u{c} {base.upper()}!", f"{base} https://t.co/{c}", f"#tag {base}."][c]
            rel = relevant if rng.random() > 0.1 else not relevant
            rows.append({
                "_id": f"id{len(rows):05d}",
                "text": text,
                "relevant": rel,
                "watson": repr({"categories": [{"label": p} for p in paths]}),
            })
    rows.append({"_id": "url-only-1", "text": "https://t.co/a", "relevant": False, "watson": "{}"})
    rows.append({"_id": "url-only-2", "text": "@x #y", "relevant": True, "watson": "{}"})
    return pd.DataFrame(rows)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = corpus()
        cls.manifest = sp.build_split(cls.df, seed=7)


class TestBuild(Base):
    def test_shape_and_columns(self):
        m = self.manifest
        self.assertEqual(list(m.columns), sp.MANIFEST_COLUMNS)
        self.assertEqual(list(m["row_id"]), list(self.df["_id"]))
        self.assertTrue(m["split"].isin(["train", "test"]).all())

    def test_no_group_crossing(self):
        m = self.manifest
        self.assertEqual(m.groupby("group")["split"].nunique().max(), 1)
        self.assertEqual(m.groupby("group")["inner_fold"].nunique().max(), 1)
        # Copies of the same post share a group; URL/mention-only posts share the empty group.
        self.assertEqual(m["group"].nunique(), 301)
        self.assertEqual(m.loc[m["row_id"].str.startswith("url-only"), "group"].nunique(), 1)

    def test_fractions(self):
        m = self.manifest
        self.assertAlmostEqual((m["split"] == "test").mean(), 0.2, delta=0.03)
        folds = m.loc[m["split"] == "train", "inner_fold"].value_counts()
        self.assertEqual(sorted(folds.index), [0, 1, 2, 3, 4])
        self.assertLessEqual(folds.max() - folds.min(), 0.05 * folds.sum())
        self.assertTrue((m.loc[m["split"] == "test", "inner_fold"] == -1).all())

    def test_stratified(self):
        labels = sp.stratification_labels(self.df["relevant"], [sp.tx.watson_list(w) for w in self.df["watson"]])
        test = (self.manifest["split"] == "test").to_numpy()
        for name in ("relevant", "irrelevant", "top:computer security", "top:other"):
            has = np.array([name in row for row in labels])
            self.assertAlmostEqual(has[test].sum() / has.sum(), 0.2, delta=0.06, msg=name)

    def test_deterministic_and_order_independent(self):
        again = sp.build_split(self.df, seed=7)
        pd.testing.assert_frame_equal(again, self.manifest)
        self.assertEqual(again.attrs["metadata"], self.manifest.attrs["metadata"])
        shuffled = self.df.sample(frac=1.0, random_state=1)
        m2 = sp.build_split(shuffled, seed=7).set_index("row_id").loc[self.manifest["row_id"]]
        pd.testing.assert_frame_equal(m2.reset_index(), self.manifest)
        other = sp.build_split(self.df, seed=8)
        self.assertFalse(other["split"].equals(self.manifest["split"]))

    def test_metadata(self):
        meta = self.manifest.attrs["metadata"]
        self.assertEqual(meta["seed"], 7)
        self.assertEqual(meta["algorithm"], sp.ALGORITHM)
        self.assertEqual(meta["splits_version"], sp.SPLITS_VERSION)
        for name in ("test_fraction", "inner_folds", "min_prefix", "max_group_rows", "row_order",
                     "textprep_version", "corpus_sha256", "input_sha256", "manifest_sha256", "grouping"):
            self.assertIn(name, meta)
        self.assertEqual((meta["min_prefix"], meta["max_group_rows"]), (40, 300))
        self.assertEqual(meta["manifest_sha256"], sp.manifest_sha256(self.manifest))
        self.assertEqual(meta["counts"]["rows"], len(self.df))
        self.assertEqual(sum(meta["counts"]["inner_fold_rows"]), meta["counts"]["train_rows"])
        self.assertEqual(meta["taxonomy_sha256"], sp.tx.taxonomy_sha256())

    def test_rejects_bad_input(self):
        dup = pd.concat([self.df, self.df.iloc[:1]])
        with self.assertRaises(ValueError):
            sp.build_split(dup, seed=1)
        with self.assertRaises(ValueError):
            sp.build_split(self.df, seed=1, test_fraction=0)
        with self.assertRaises(ValueError):
            sp.build_split(self.df, seed=1, inner_folds=1)
        with self.assertRaises(TypeError):
            sp.build_split(self.df.assign(text=[None] * len(self.df)), seed=1)
        with self.assertRaises(ValueError):
            sp.build_split(self.df, seed=1, min_prefix=0)


def tiny(n_groups):
    return pd.DataFrame({
        "_id": [f"t{i}" for i in range(n_groups)],
        "text": [f"distinct post {i}" for i in range(n_groups)],
        "relevant": [i % 2 == 0 for i in range(n_groups)],
        "watson": ["{}"] * n_groups,
    })


class TestEmptyPartitions(unittest.TestCase):
    def test_empty_test_split_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            sp.build_split(tiny(1), seed=0)

    def test_empty_inner_fold_raises(self):
        # 10 groups, at least one in test: 9 inner folds cannot all be filled.
        with self.assertRaisesRegex(ValueError, "inner fold"):
            sp.build_split(tiny(10), seed=0, inner_folds=9)
        manifest = sp.build_split(tiny(20), seed=0, inner_folds=3)
        counts = manifest.attrs["metadata"]["counts"]
        self.assertTrue(counts["test_rows"] > 0 and min(counts["inner_fold_rows"]) > 0)


LONG = "Critical vulnerability found in major manufacturers of android phones worldwide"


class TestGrouping(unittest.TestCase):
    def test_full_prefix_closure(self):
        # a links ab and ac although ab and ac are adjacent and not prefixes of each other.
        root = sp.prefix_components(["ac", "ab", "a", "b"], min_prefix=1)
        self.assertEqual(root, {"a": "a", "ab": "a", "ac": "a", "b": "b"})
        # Transitive chains and non-adjacent members.
        root = sp.prefix_components(["abc", "abcde", "abcd", "abcdz", "abd", "abcx"], min_prefix=3)
        self.assertEqual({k for k, r in root.items() if r == "abc"}, {"abc", "abcd", "abcde", "abcdz", "abcx"})
        self.assertEqual(root["abd"], "abd")

    def test_min_length_boundary(self):
        # A prefix of exactly min_prefix characters links; one character shorter does not.
        self.assertEqual(sp.prefix_components(["abcd", "abcdef"], min_prefix=4)["abcdef"], "abcd")
        self.assertEqual(sp.prefix_components(["abc", "abcdef"], min_prefix=4)["abcdef"], "abcdef")
        # A short key does not link, but a longer key above it still does.
        root = sp.prefix_components(["ab", "abcd", "abcdx"], min_prefix=3)
        self.assertEqual(root, {"ab": "ab", "abcd": "abcd", "abcdx": "abcd"})
        # The empty key never merges.
        self.assertEqual(sp.prefix_components(["", "x" * 50], min_prefix=1)["x" * 50], "x" * 50)
        with self.assertRaises(ValueError):
            sp.prefix_components(["a"], min_prefix=0)

    def test_truncated_copies_share_group(self):
        texts = [LONG, LONG[:45] + "\u2026", "RT @x: " + LONG[:60] + "... https://t.co/a", LONG[:39] + "\u2026",
                 "Something else entirely"]
        keys = sp.group_rows(texts, min_prefix=40)
        self.assertEqual(len(set(keys[:3])), 1)
        self.assertEqual(keys[0], min(keys[:3]))  # the smallest key names the component
        self.assertNotEqual(keys[3], keys[0])  # 38-character key: below min_prefix
        stats = sp.grouping_stats(texts, min_prefix=40)
        self.assertEqual((stats["merged_components"], stats["keys_absorbed"]), (1, 2))
        self.assertEqual(stats["merged_component_rows"], {"3": 1})

    def test_build_split_merges_and_caps(self):
        extra = pd.DataFrame({
            "_id": [f"trunc{i}" for i in range(12)],
            "text": [LONG[: 40 + i] + "\u2026" for i in range(12)],
            "relevant": [True] * 12,
            "watson": ["{}"] * 12,
        })
        df = pd.concat([corpus(), extra], ignore_index=True)
        manifest = sp.build_split(df, seed=3)
        self.assertEqual(manifest.loc[manifest["row_id"].str.startswith("trunc"), "group"].nunique(), 1)
        self.assertEqual(manifest.attrs["metadata"]["grouping"]["largest_merged_component_rows"], 12)
        sp.verify_manifest(df, manifest)
        with tempfile.TemporaryDirectory() as tmp:  # metadata with merges survives JSON
            sp.write_manifest(manifest, Path(tmp) / "m.csv")
            sp.verify_manifest(df, sp.read_manifest(Path(tmp) / "m.csv"), expected_seed=3)
        with self.assertRaisesRegex(ValueError, "max_group_rows"):
            sp.build_split(df, seed=3, max_group_rows=11)
        sp.build_split(df, seed=3, max_group_rows=None)


class TestVerifyAndIO(Base):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_round_trip_verifies(self):
        sp.verify_manifest(self.df, self.manifest)
        path = self.tmp / "split.csv"
        sidecar = sp.write_manifest(self.manifest, path)
        self.assertEqual(sidecar, self.tmp / "split.json")
        loaded = sp.read_manifest(path)
        sp.verify_manifest(self.df, loaded)
        sp.verify_manifest(self.df.sample(frac=1.0, random_state=3), loaded)

    def assert_rejected(self, manifest, df=None, metadata=None):
        with self.assertRaises(sp.ManifestError):
            sp.verify_manifest(self.df if df is None else df, manifest, metadata)

    def test_missing_metadata_key_is_manifest_error(self):
        meta = self.manifest.attrs["metadata"]
        for key in ("inner_folds", "seed", "test_fraction", "max_group_rows"):
            with self.subTest(key=key):
                self.assert_rejected(self.manifest, metadata={k: v for k, v in meta.items() if k != key})

    def test_rebuild_cap_failure_is_manifest_error(self):
        meta = self.manifest.attrs["metadata"]
        capped = {**meta, "max_group_rows": 1}
        capped["input_sha256"] = sp.input_sha256(meta["corpus_sha256"], capped)
        with self.assertRaisesRegex(sp.ManifestError, "max_group_rows"):
            sp.verify_manifest(self.df, self.manifest, capped)

    def test_catches_tampering(self):
        m = self.manifest
        meta = m.attrs["metadata"]
        test_group = m.loc[m["split"] == "test", "group"].iloc[0]
        train_rows = m.index[m["split"] == "train"]
        # Moving a whole group keeps all structural invariants; only the rebuild catches it.
        moved = m.copy()
        sel = moved["group"] == test_group
        moved.loc[sel, "split"] = "train"
        moved.loc[sel, "inner_fold"] = 0
        new_counts = sp._counts(moved, meta["inner_folds"])
        self.assert_rejected(moved, metadata={**meta, "counts": new_counts})
        # Splitting a group across train and test.
        crossed = m.copy()
        idx = crossed.index[sel][0]
        crossed.loc[idx, ["split", "inner_fold"]] = ["train", 0]
        self.assert_rejected(crossed, metadata=meta)
        # Missing and duplicated rows.
        self.assert_rejected(m.drop(index=train_rows[0]), metadata=meta)
        self.assert_rejected(pd.concat([m, m.iloc[:1]]), metadata=meta)
        # Inner fold of a train row changed.
        refold = m.copy()
        refold.loc[train_rows[0], "inner_fold"] = (refold.loc[train_rows[0], "inner_fold"] + 1) % 5
        self.assert_rejected(refold, metadata=meta)
        # Metadata mismatches.
        self.assert_rejected(m, metadata={**meta, "seed": 8})
        self.assert_rejected(m, metadata={**meta, "textprep_version": "0"})
        self.assert_rejected(m, metadata={**meta, "taxonomy_sha256": "x"})
        # Corpus changed (a label edit).
        edited = self.df.copy()
        edited.loc[0, "relevant"] = not edited.loc[0, "relevant"]
        self.assert_rejected(m, df=edited, metadata=meta)

    def test_catches_wrong_group(self):
        m = self.manifest
        meta = m.attrs["metadata"]
        # Rename one whole group: splits, folds and counts stay valid, the grouping does not.
        renamed = m.copy()
        renamed.loc[renamed["group"] == renamed["group"].iloc[0], "group"] = "g0000000000000000"
        self.assert_rejected(renamed, metadata=meta)
        # Move one row into another existing group in the same split and fold.
        moved = m.copy()
        first = moved.iloc[0]
        same = moved[(moved["split"] == first["split"]) & (moved["inner_fold"] == first["inner_fold"])
                     & (moved["group"] != first["group"])]
        moved.loc[moved.index[0], "group"] = same["group"].iloc[0]
        with self.assertRaisesRegex(sp.ManifestError, "recomputed grouping"):
            sp.verify_manifest(self.df, moved, meta)

    def test_expected_seed_and_sha(self):
        m = self.manifest
        digest = sp.manifest_sha256(m)
        sp.verify_manifest(self.df, m, expected_seed=7, expected_manifest_sha256=digest)
        with self.assertRaisesRegex(sp.ManifestError, "seed"):
            sp.verify_manifest(self.df, m, expected_seed=42)
        with self.assertRaisesRegex(sp.ManifestError, "sha256"):
            sp.verify_manifest(self.df, m, expected_manifest_sha256="0" * 64)
        # The digest is canonical: row order does not change it, a content edit does.
        self.assertEqual(sp.manifest_sha256(m.sample(frac=1.0, random_state=4)), digest)
        edited = m.copy()
        edited.loc[edited.index[0], "inner_fold"] = 9
        self.assertNotEqual(sp.manifest_sha256(edited), digest)

    def test_catches_invalid_values_and_input_digest(self):
        m = self.manifest
        meta = m.attrs["metadata"]
        bad = m.copy()
        bad.loc[bad.index[0], "split"] = "validation"
        self.assert_rejected(bad, metadata={**meta, "manifest_sha256": sp.manifest_sha256(bad)})
        test_row = m.index[m["split"] == "test"][0]
        bad = m.copy()
        bad.loc[test_row, "inner_fold"] = 0
        self.assert_rejected(bad, metadata=meta)
        self.assert_rejected(m, metadata={**meta, "input_sha256": "x"})
        self.assert_rejected(m, metadata={**meta, "min_prefix": 41})
        self.assert_rejected(m, metadata={**meta, "algorithm": "other"})

    def test_corpus_digest_is_canonical(self):
        digest = sp.corpus_sha256(self.df)
        self.assertEqual(sp.corpus_sha256(self.df.sample(frac=1.0, random_state=5)), digest)
        self.assertEqual(sp.corpus_sha256(self.df.assign(extra=1)), digest)
        edited = self.df.copy()
        edited.loc[0, "text"] += "!"
        self.assertNotEqual(sp.corpus_sha256(edited), digest)

    def test_read_detects_edited_csv(self):
        path = self.tmp / "split.csv"
        sp.write_manifest(self.manifest, path)
        text = path.read_text().replace(",test,-1", ",train,0", 1)
        path.write_text(text)
        with self.assertRaises(sp.ManifestError):
            sp.read_manifest(path)

    def test_assign_split(self):
        out = sp.assign_split(self.df.sample(frac=1.0, random_state=2), self.manifest)
        joined = out.set_index("_id").loc[self.manifest["row_id"]]
        self.assertEqual(list(joined["split"]), list(self.manifest["split"]))
        self.assertEqual(list(joined["inner_fold"]), list(self.manifest["inner_fold"]))


class TestBalanceReport(Base):
    def test_report(self):
        rep = sp.balance_report(self.df, self.manifest)
        counts = self.manifest.attrs["metadata"]["counts"]
        folds = [f"fold_{f}" for f in range(5)]
        self.assertEqual(set(rep["level"]), {"rows", "groups"})
        rows = rep[rep["level"] == "rows"].set_index("label")
        groups = rep[rep["level"] == "groups"].set_index("label")
        self.assertEqual(rows.loc["(all)", "total"], counts["rows"])
        self.assertEqual(rows.loc["(all)", "test"], counts["test_rows"])
        self.assertEqual(groups.loc["(all)", "total"], counts["groups"])
        self.assertEqual(list(groups.loc["(all)", folds]), counts["inner_fold_groups"])
        self.assertTrue((rows[folds].sum(axis=1) == rows["train"]).all())
        self.assertTrue((rows["train"] + rows["test"] == rows["total"]).all())
        self.assertTrue((groups["total"] <= rows["total"]).all())
        self.assertLess(groups.loc["(all)", "total"], rows.loc["(all)", "total"])
        self.assertAlmostEqual(rows.loc["relevant", "test_dev"], rows.loc["relevant", "test_share"] - 0.2)
        self.assertLess(abs(rows.loc["relevant", "test_dev"]), 0.06)
        self.assertGreaterEqual(rows["max_fold_dev"].min(), 0)
        # Hand check of one label at both levels.
        labels = sp.stratification_labels(self.df["relevant"], [sp.tx.watson_list(w) for w in self.df["watson"]])
        has = np.array(["relevant" in row for row in labels])
        test = (self.manifest["split"] == "test").to_numpy()
        self.assertEqual(rows.loc["relevant", "test"], int((has & test).sum()))
        self.assertEqual(groups.loc["relevant", "test"], self.manifest.loc[has & test, "group"].nunique())

    def test_custom_labels(self):
        labels = [["x"] if i % 2 else [] for i in range(len(self.df))]
        rep = sp.balance_report(self.df, self.manifest, labels)
        self.assertEqual(set(rep["label"]), {"(all)", "x"})
        with self.assertRaises(ValueError):
            sp.balance_report(self.df, self.manifest, labels[:-1])


class TestFrequentLabelsets(unittest.TestCase):
    def test_threshold_and_order(self):
        labels = [["b", "a"]] * 20 + [["a", "b"]] * 1 + [["c"]] * 19 + [["other"]] * 25
        self.assertEqual(sp.frequent_labelsets(labels), {("a", "b"), ("other",)})
        self.assertEqual(sp.frequent_labelsets(labels, min_count=19), {("a", "b"), ("c",), ("other",)})
        self.assertEqual(sp.frequent_labelsets([]), set())

    def test_group_counts(self):
        labels = [["a"]] * 20 + [["b"]] * 3
        groups = ["g1"] * 15 + ["g2"] * 5 + ["g3"] * 3
        self.assertEqual(sp.frequent_labelsets(labels, 20, groups), {("a",): {"rows": 20, "groups": 2}})
        with self.assertRaises(ValueError):
            sp.frequent_labelsets(labels, 20, groups[:-1])


if __name__ == "__main__":
    unittest.main()
