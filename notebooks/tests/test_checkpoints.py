import json
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from notebooks import checkpoints as cp


def ident(cfg=None, tmp=None):
    return cp.make_identity(config=cfg or {"a": 1}, modules=["numpy"])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.id = ident()

    def store(self, name="out", search=(), identity=None):
        return cp.Store(self.tmp / name, [self.tmp / s for s in search], identity or self.id)


class TestIdentity(Base):
    def test_stable_and_sensitive(self):
        f = self.tmp / "code.py"
        f.write_text("x=1")
        a = cp.make_identity([f], config={"b": 2, "a": 1}, modules=["numpy"])
        b = cp.make_identity([f], config={"a": 1, "b": 2}, modules=["numpy"])
        self.assertEqual(a["digest"], b["digest"])
        f.write_text("x=2")
        self.assertNotEqual(a["digest"], cp.make_identity([f], config={"a": 1, "b": 2}, modules=["numpy"])["digest"])
        self.assertNotEqual(a["digest"], cp.make_identity([f], config={"a": 2}, modules=["numpy"])["digest"])

    def test_config_must_be_json(self):
        with self.assertRaises(TypeError):
            cp.make_identity(config={"x": object()})


class TestStore(Base):
    def test_roundtrip_and_list(self):
        s = self.store()
        self.assertFalse(s.has(("m", "1")))
        self.assertTrue(s.save(("m", "1"), {"w": [1, 2]}))
        self.assertTrue(s.has(("m", "1")))
        self.assertEqual(s.load(("m", "1")), {"w": [1, 2]})
        s.save({"a": "x", "b": "y"}, np.arange(5), fmt="npy")
        np.testing.assert_array_equal(s.load({"b": "y", "a": "x"}), np.arange(5))
        self.assertEqual(len(s.list_keys()), 2)
        self.assertEqual([p for p in os.listdir(s.out_dir) if p.endswith(".tmp")], [])

    def test_filename_safe_deterministic(self):
        n = cp.key_basename(("a b", "x y"))
        self.assertEqual(n, cp.key_basename(("a b", "x y")))
        self.assertRegex(n, r"^[A-Za-z0-9._=-]+-[0-9a-f]{64}$")
        self.assertNotEqual(cp.key_basename(("a", "b")), cp.key_basename(("a-b",)))

    def test_data_without_sidecar_ignored(self):
        s = self.store()
        s.save(("k",), 1)
        base = cp.key_basename(("k",))
        (s.out_dir / (base + ".json")).unlink()  # crash before marker
        self.assertFalse(s.has(("k",)))
        self.assertEqual(s.list_keys(), [])
        with self.assertRaises(KeyError):
            s.load(("k",))
        self.assertTrue(s.save(("k",), 2))  # orphan data is overwritten
        self.assertEqual(s.load(("k",)), 2)

    def test_sidecar_written_last(self):
        s = self.store()
        orig = cp._atomic_write
        seen = []

        def spy(path, writer):
            seen.append(Path(path).suffix)
            if Path(path).suffix == ".json":
                raise RuntimeError("crash")
            orig(path, writer)

        cp._atomic_write = spy
        try:
            with self.assertRaises(RuntimeError):
                s.save(("k",), 1)
        finally:
            cp._atomic_write = orig
        self.assertEqual(seen, [".pkl", ".json"])
        self.assertFalse(s.has(("k",)))

    def test_corruption_detected(self):
        s = self.store()
        s.save(("k",), list(range(100)))
        p = s.out_dir / (cp.key_basename(("k",)) + ".pkl")
        raw = bytearray(p.read_bytes())
        raw[-3] ^= 0xFF
        p.write_bytes(bytes(raw))
        with self.assertRaises(cp.CorruptArtifactError):
            s.load(("k",))
        with self.assertRaises(cp.CorruptArtifactError):
            s.has(("k",))  # verified by default: same-size corruption caught
        p.write_bytes(bytes(raw[:-5]))  # truncated: caught even in cheap mode
        with self.assertRaises(cp.CorruptArtifactError):
            s.has(("k",), verify=False)

    def test_bad_sidecar(self):
        s = self.store()
        s.save(("k",), 1)
        (s.out_dir / (cp.key_basename(("k",)) + ".json")).write_text("{nope")
        with self.assertRaises(cp.CorruptArtifactError):
            s.has(("k",))
        self.assertEqual(s.list_keys(), [])

    def test_identity_mismatch(self):
        self.store().save(("k",), 1)
        other = self.store(identity=ident({"a": 2}))
        with self.assertRaises(cp.IdentityMismatchError):
            other.has(("k",))
        with self.assertRaises(cp.ConflictError):
            other.save(("k",), 1)
        self.assertEqual(other.list_keys(), [])

    def test_resave(self):
        s = self.store()
        self.assertTrue(s.save(("k",), {"a": 1}))
        self.assertFalse(s.save(("k",), {"a": 1}))
        with self.assertRaises(cp.ConflictError):
            s.save(("k",), {"a": 2})
        self.assertEqual(s.load(("k",)), {"a": 1})

    def test_search_dirs_readonly_and_duplicates(self):
        prev = self.store("prev")
        prev.save(("k",), "v")
        before = sorted(os.listdir(prev.out_dir))
        s = self.store("out", search=["prev", "prev", "out"])
        self.assertEqual(len(s.search_dirs), 1)
        self.assertEqual(s.load(("k",)), "v")
        self.assertEqual(s.list_keys(), [("k",)])
        self.assertEqual(sorted(os.listdir(prev.out_dir)), before)
        # identical duplicate accepted
        self.assertFalse(s.save(("k",), "v"))  # identical valid copy in a search dir: no-op
        self.assertEqual(cp.Store(self.tmp / "out", [], self.id).list_keys(), [])
        shutil.copytree(prev.out_dir, self.tmp / "copy")
        self.assertEqual(self.store("out", search=["prev", "copy"]).load(("k",)), "v")
        # different duplicate rejected
        d = self.store("out2", search=["prev"])
        d.out_dir.joinpath("x").touch()
        with self.assertRaises(cp.ConflictError):  # save checks search dirs too
            d.save(("k",), "other")
        d3 = self.store("out3")
        d3.save(("k",), "other")
        d = self.store("out3", search=["prev"])
        with self.assertRaises(cp.DuplicateError):
            d.load(("k",))
        with self.assertRaises(cp.DuplicateError):
            d.has(("k",))
        self.assertEqual(d.list_keys(), [])
        # save never writes into search dirs
        self.assertEqual(sorted(os.listdir(prev.out_dir)), before)


def sidecar(s, key):
    return s.out_dir / (cp.key_basename(key) + ".json")


def edit_sidecar(s, k0, ch):
    p = sidecar(s, k0)
    m = json.loads(p.read_text())
    m.update(ch)
    p.write_text(json.dumps(m))


class TestHardening(Base):
    def test_sidecar_validation(self):
        k = ("k",)
        for ch in ({"data_file": "other.pkl"}, {"format": "npy"}, {"format": "exe"},
                   {"schema": 99}, {"key": cp._normalize_key(("z",))[0]},
                   {"data_file": "../x.pkl"}, {"sha256": "abc"}):
            s = self.store(f"o{abs(hash(str(ch)))}")
            s.save(k, 1)
            edit_sidecar(s, k, ch)
            with self.assertRaises(cp.CorruptArtifactError, msg=str(ch)):
                s.has(k, verify=False)

    def test_npy_safety(self):
        s = self.store()
        with self.assertRaises(TypeError):
            s.save(("o",), np.array([{"a": 1}, None], dtype=object), fmt="npy")
        self.assertFalse(s.has(("o",)))
        with self.assertRaises(ValueError):
            s.save(("o",), 1, fmt="exe")
        s.save(("n",), np.arange(3), fmt="npy")
        calls = []
        orig = np.load
        np.load = lambda *a, **k: (calls.append(k), orig(*a, **k))[1]
        try:
            s.load(("n",))
        finally:
            np.load = orig
        self.assertEqual(calls, [{"allow_pickle": False}])

    def test_has_verified_by_default(self):
        s = self.store()
        s.save(("k",), list(range(50)))
        p = s.out_dir / (cp.key_basename(("k",)) + ".pkl")
        raw = bytearray(p.read_bytes())
        raw[-3] ^= 0xFF
        p.write_bytes(bytes(raw))
        self.assertTrue(s.has(("k",), verify=False))  # cheap mode misses it
        with self.assertRaises(cp.CorruptArtifactError):
            s.has(("k",))

    def test_priority_copy_fails_closed(self):
        prev = self.store("prev")
        prev.save(("k",), "v")
        shutil.copytree(prev.out_dir, self.tmp / "out")
        p = self.tmp / "out" / (cp.key_basename(("k",)) + ".pkl")
        raw = bytearray(p.read_bytes())
        raw[-3] ^= 0xFF
        p.write_bytes(bytes(raw))
        s = self.store("out", search=["prev"])  # lower-priority copy is good
        with self.assertRaises(cp.CorruptArtifactError):
            s.has(("k",))
        with self.assertRaises(cp.CorruptArtifactError):
            s.load(("k",))

    def test_save_repair_required(self):
        s = self.store()
        s.save(("k",), 1)
        (s.out_dir / (cp.key_basename(("k",)) + ".pkl")).unlink()
        with self.assertRaises(cp.RepairRequiredError) as cm:
            s.save(("k",), 1)
        self.assertIn("delete", str(cm.exception))
        self.assertFalse((s.out_dir / (cp.key_basename(("k",)) + ".lock")).exists())

    def test_lock(self):
        s = self.store()
        lock = s.out_dir / (cp.key_basename(("k",)) + ".lock")
        lock.write_text("123")  # stale lock
        with self.assertRaises(cp.LockError) as cm:
            s.save(("k",), 1)
        self.assertIn(".lock", str(cm.exception))
        self.assertTrue(lock.exists())  # not removed by a failed acquire
        lock.unlink()
        self.assertTrue(s.save(("k",), 1))
        self.assertFalse(lock.exists())

    def test_key_encoding(self):
        for bad in (("a/b",), ("..",), ("a\\b",), ("a\x00",), ("x..y",), {"k": "../v"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                cp.key_basename(bad)
        with self.assertRaises(TypeError):
            cp.key_basename((1,))
        a = cp._normalize_key(("a", "b"))[0]
        self.assertEqual(json.loads(a), ["t", ["a", "b"]])
        self.assertNotEqual(cp.key_basename(("a", "b")), cp.key_basename({"a": "b"}))
        self.assertNotEqual(cp.key_basename(("a__b",)), cp.key_basename(("a", "b")))
        self.assertEqual(len(cp.key_basename(("a",)).rsplit("-", 1)[1]), 64)

    def test_identity_names(self):
        f, g = self.tmp / "f.py", self.tmp / "g.py"
        f.write_text("1")
        g.write_text("2")
        i = cp.make_identity({"util": f}, {"train": g})
        self.assertEqual(set(i["code"]), {"util"})
        self.assertEqual(set(i["data"]), {"train"})
        self.assertEqual(i["identity_version"], cp.IDENTITY_VERSION)
        self.assertIn("python", i)
        self.assertEqual(cp.make_identity([f])["code"].keys(), {"f.py"})
        d = self.tmp / "sub"
        d.mkdir()
        (d / "f.py").write_text("3")
        with self.assertRaises(ValueError):
            cp.make_identity([f, d / "f.py"])
        with self.assertRaises(FileNotFoundError):
            cp.make_identity([self.tmp / "nope.py"])
        with self.assertRaises(FileNotFoundError):
            cp.make_identity({"x": self.tmp / "nope.py"})
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cp.make_identity(config={"x": bad})


class TestBudgetHardening(unittest.TestCase):
    def test_started_at_and_offset(self):
        b = cp.Budget(hours=1, clock=lambda: 100.0, started_at=900.0, wall=lambda: 1000.0)
        self.assertEqual(b.elapsed(), 100.0)
        b = cp.Budget(hours=1, clock=lambda: 5.0, spent_seconds=600)
        self.assertEqual(b.remaining(), 3000)
        with self.assertRaises(ValueError):  # start in the future
            cp.Budget(hours=1, started_at=2000.0, wall=lambda: 1000.0)

    def test_env_session_start(self):
        import time
        b = cp.Budget.from_env({"CHECKPOINT_BUDGET_HOURS": "1",
                                "CHECKPOINT_SESSION_START": str(time.time() - 1800),
                                "CHECKPOINT_FINALISE_MINUTES": "5"})
        self.assertGreaterEqual(b.elapsed(), 1800)
        self.assertEqual(b.finalise_seconds, 300.0)

    def test_validation(self):
        for kw in ({"hours": float("nan")}, {"hours": -1}, {"hours": float("inf")},
                   {"margin_minutes": -1}, {"margin_minutes": float("nan")},
                   {"finalise_minutes": -2}, {"spent_seconds": -1}):
            with self.assertRaises(ValueError, msg=str(kw)):
                cp.Budget(**kw)
        with self.assertRaises(ValueError):
            cp.Budget.from_env({"CHECKPOINT_BUDGET_HOURS": "nan"})
        with self.assertRaises(ValueError):
            cp.Budget(hours=1).allows(float("nan"))

    def test_finalise_reserved(self):
        b = cp.Budget(hours=1, margin_minutes=10, finalise_minutes=5, clock=lambda: 0.0)
        self.assertTrue(b.allows(2700))
        self.assertFalse(b.allows(2701))


class TestBudget(unittest.TestCase):
    def test_budget(self):
        t = [0.0]
        b = cp.Budget(hours=1, margin_minutes=10, clock=lambda: t[0])
        self.assertEqual(b.remaining(), 3000)
        self.assertTrue(b.allows(3000))
        self.assertFalse(b.allows(3001))
        self.assertTrue(b.allows(kind="fit"))  # nothing recorded yet
        b.record("fit", 1000)
        b.record("fit", 2000)
        self.assertEqual(b.stats("fit"), {"count": 2, "mean": 1500.0, "max": 2000.0})
        t[0] = 1500
        self.assertEqual(b.elapsed(), 1500)
        self.assertFalse(b.allows(kind="fit"))  # max 2000 > 1500 remaining
        self.assertTrue(b.allows(1500))

    def test_unlimited_and_env(self):
        self.assertTrue(cp.Budget().allows(1e12))
        b = cp.Budget.from_env({"CHECKPOINT_BUDGET_HOURS": "2", "CHECKPOINT_MARGIN_MINUTES": "30"})
        self.assertEqual((b.hours, b.margin_seconds), (2.0, 1800.0))
        self.assertIsNone(cp.Budget.from_env({}).hours)


class TestStatusAndVerify(Base):
    def test_status_roundtrip(self):
        d = self.tmp / "st"
        self.assertIsNone(cp.read_status(d))
        cp.write_status(d, "INCOMPLETE", done=[("a",)], missing=["b"], failed={"c": "boom"})
        st = cp.read_status(d)
        self.assertEqual(st["state"], "INCOMPLETE")
        self.assertEqual(st["missing"], ["b"])
        self.assertEqual(st["failed"], {"c": "boom"})
        self.assertEqual(len(st["done"]), 1)
        with self.assertRaises(ValueError):
            cp.write_status(d, "DONE")

    def test_verify_complete(self):
        s = self.store()
        s.save(("a",), 1)
        s.save(("b",), 2)
        s.save(("extra",), 3)
        self.assertEqual(cp.verify_complete(s, [("a",), ("b",)])["missing"], [])
        p = s.out_dir / (cp.key_basename(("b",)) + ".pkl")
        p.write_bytes(p.read_bytes()[:-1] + b"\x00")
        with self.assertRaises(cp.IncompleteError) as cm:
            cp.verify_complete(s, [("a",), ("b",), ("c",)])
        msg = str(cm.exception)
        self.assertIn("1 missing, 1 invalid, 0 duplicate of 3", msg)
        self.assertIn("missing: ", msg)
        self.assertEqual(cm.exception.missing, [("c",)])
        self.assertTrue(any("CorruptArtifactError" in v for v in cm.exception.invalid.values()))
        with self.assertRaises(cp.IncompleteError) as cm:
            cp.verify_complete(s, [("a",), ("a",)])
        self.assertEqual(cm.exception.duplicates, [("a",)])
        self.assertIn("duplicate:", str(cm.exception))
        with self.assertRaises(ValueError):
            cp.verify_complete(s, [])
        self.assertEqual(cp.verify_complete(s, [], allow_empty=True)["missing"], [])


class TestEnv(Base):
    def test_env(self):
        (self.tmp / "s1").mkdir()
        (self.tmp / "s2").mkdir()
        env = {"CHECKPOINT_INPUTS": os.pathsep.join([str(self.tmp / "s*"), str(self.tmp / "nope")])}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dirs = cp.input_dirs_from_env(env)
        self.assertEqual(dirs, [str(self.tmp / "s1"), str(self.tmp / "s2")])
        self.assertEqual(len(w), 1)
        self.assertEqual(cp.input_dirs_from_env({}), [])
        self.assertEqual(cp.checkpoint_dir_from_env({}), "./checkpoints")
        self.assertEqual(cp.checkpoint_dir_from_env({"CHECKPOINT_DIR": "x"}), "x")


if __name__ == "__main__":
    unittest.main()
