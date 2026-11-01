import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from notebooks import embeddings as em
from notebooks.textprep import TEXTPREP_VERSION

VECTORS = {
    "hello": [1.0, 2.0, 3.0],
    "world": [0.5, 0.5, 0.5],
    "!": [10.0, 0.0, 0.0],
}


def write_glove(path):
    lines = [f"{w} " + " ".join(str(x) for x in v) for w, v in VECTORS.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def glove_embedder(path, rtol=0.0):
    return em.Embedder("G", "glove", "G_PATH", 3, str(path), {}, rtol)


class StubModel:
    """Deterministic stand-in for SentenceTransformer: embeds by text length / content."""

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.calls.append((list(texts), batch_size))
        return np.array([[len(t), sum(map(ord, t)) % 97, 1.0, float(batch_size > 0)] for t in texts])


def st_embedder(rtol=1e-4):
    enc = {"batch_size": 32, "normalize_embeddings": False, "convert_to_numpy": True,
           "show_progress_bar": False}
    return em.Embedder("S", "sentence_transformer", "S_PATH", None, "org/model", enc, rtol)


class TestConfig(unittest.TestCase):
    def test_default_config_lists_the_seven_embedders(self):
        env = {e: "x" for e in ("GLOVE_6B_50D_PATH", "GLOVE_6B_100D_PATH", "GLOVE_6B_200D_PATH",
                                "GLOVE_6B_300D_PATH", "DISTILROBERTA_PATH", "SBERT_PATH",
                                "ATTACK_BERT_PATH")}
        e = em.load_embedders(environ=env)
        self.assertEqual(list(e), ["GloVe.6B.50D", "GloVe.6B.100D", "GloVe.6B.200D", "GloVe.6B.300D",
                                   "DistilRoBERTa", "SBERT", "ATT&CK-BERT"])
        self.assertEqual(e["GloVe.6B.200D"].dim, 200)
        self.assertEqual(e["SBERT"].encode["normalize_embeddings"], False)
        self.assertNotIn("source", e["SBERT"].config())

    def test_missing_env_and_unknown_name(self):
        with self.assertRaises(em.ConfigError):
            em.load_embedders(environ={})
        with self.assertRaises(em.ConfigError):
            em.load_embedders(environ={}, names=["nope"], require_source=False)
        self.assertEqual(list(em.load_embedders(environ={}, names=["SBERT"], require_source=False)), ["SBERT"])


class TestGlove(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "vec.txt"
        write_glove(self.path)
        self.emb = glove_embedder(self.path)

    def test_sum_lowercase_oov_empty(self):
        out = em.embed(["Hello WORLD!", "hello unknownword", "", "zzz"], self.emb)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[0], [11.5, 2.5, 3.5])
        np.testing.assert_allclose(out[1], [1, 2, 3])
        np.testing.assert_array_equal(out[2], 0)
        np.testing.assert_array_equal(out[3], 0)

    def test_preprocessing_applied(self):
        out = em.embed(["hello #world @world https://t.co/x"], self.emb)
        np.testing.assert_allclose(out[0], [1, 2, 3])

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            em.embed([float("nan")], self.emb)

    def test_non_finite_output_rejected(self):
        self.path.write_text("hello 1 nan 3\n")
        with self.assertRaises(em.NonFiniteError):
            em.embed(["hello"], glove_embedder(self.path))

    def test_bad_line_rejected(self):
        self.path.write_text("hello 1 2\n")
        with self.assertRaises(em.ConfigError):
            em.embed(["hello"], glove_embedder(self.path))

    def test_fingerprint_is_file_sha_and_path_independent(self):
        fp = em.fingerprint(self.emb)
        other = Path(self.tmp.name) / "copy.txt"
        other.write_bytes(self.path.read_bytes())
        self.assertEqual(fp, em.fingerprint(glove_embedder(other)))
        self.assertEqual(fp["kind"], "file")
        self.assertTrue(em.fingerprints_match(fp, dict(fp)))

    def test_parity_exact(self):
        res = em.parity_check(self.emb, ["hello", "world !"])
        self.assertEqual(res["max_abs_diff"], 0.0)

    def test_empty_input(self):
        self.assertEqual(em.embed([], self.emb).shape, (0, 3))


class TestSentenceTransformer(unittest.TestCase):
    def setUp(self):
        self.model = StubModel()
        em._MODEL_CACHE.clear()
        p = mock.patch.object(em, "_load_sentence_transformer", return_value=self.model)
        self.loader = p.start()
        self.addCleanup(p.stop)

    def test_preprocess_cached_model_device_and_options(self):
        e = st_embedder()
        out = em.embed(["see #tag http://x.co now", "b"], e, device="cpu")
        em.embed(["c"], e, device="cpu")
        self.assertEqual(self.loader.call_count, 1)  # loaded once
        self.assertEqual(self.model.calls[0], (["see now", "b"], 32))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (2, 4))
        em.embed(["c"], e, device="cpu", batch_size=5)
        self.assertEqual(self.model.calls[-1][1], 5)

    def test_device_default_is_recorded_value(self):
        self.assertIn(em.resolve_device(), ("cpu", "cuda"))
        self.assertEqual(em.resolve_device("cpu"), "cpu")

    def test_parity_ok_and_failure(self):
        e = st_embedder()
        self.assertEqual(em.parity_check(e, ["a", "bb"], device="cpu")["max_abs_diff"], 0.0)
        real = self.model.encode

        def noisy(texts, batch_size, **kw):
            return real(texts, batch_size, **kw) + (1.0 if batch_size == 7 else 0.0)

        self.model.encode = noisy
        with self.assertRaises(em.ParityError):
            em.parity_check(e, ["a", "bb"], device="cpu")

    def test_non_finite_rejected(self):
        self.model.encode = lambda texts, **kw: np.full((len(texts), 4), np.nan)
        with self.assertRaises(em.NonFiniteError):
            em.embed(["a"], st_embedder(), device="cpu")

    def test_hf_fingerprint_unresolved_and_match(self):
        with mock.patch.object(em, "_hf_revision", return_value=None):
            fp = em.fingerprint(st_embedder())
        self.assertEqual(fp, {"kind": "hf", "model": "org/model", "revision": "unresolved"})
        self.assertIsNone(em.fingerprints_match(fp, fp))
        a = {"kind": "hf", "model": "m", "revision": "r1"}
        self.assertFalse(em.fingerprints_match(a, {**a, "revision": "r2"}))
        self.assertFalse(em.fingerprints_match(a, {**a, "model": "n"}))


class TestArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        vec = self.dir / "vec.txt"
        write_glove(vec)
        self.emb = glove_embedder(vec)
        self.ids = ["b", "a", "c"]
        self.texts = ["hello", "world", "hello world"]
        self.matrix = em.embed(self.texts, self.emb)
        self.fp = em.fingerprint(self.emb)
        self.out = self.dir / "out"
        self.meta = em.build_meta(self.emb, self.ids, self.texts, corpus="corpus", device="cpu", fp=self.fp)

    def write(self, name="x", **kw):
        return em.write_artifact(self.out, name, self.ids, self.matrix, self.meta, **kw)

    def test_round_trip_and_sidecar_content(self):
        side = self.write()
        art = em.load_artifact(self.out, "x", expected_row_ids=self.ids, expected_fingerprint=self.fp,
                               expected_texts_sha256=em.texts_sha256(self.ids, self.texts))
        np.testing.assert_array_equal(art.select(self.ids), self.matrix)
        self.assertEqual(art.shape, (3, 3))
        self.assertEqual(side["textprep_version"], TEXTPREP_VERSION)
        self.assertEqual(side["dtype"], "float32")
        self.assertEqual(side["shape"], [3, 3])
        self.assertEqual(set(side["versions"]), {"numpy", "torch", "sentence-transformers", "transformers"})
        self.assertEqual(side["embedder"]["name"], "G")
        self.assertEqual([p.name for p in self.out.iterdir() if p.name.startswith(".")], [])

    def test_texts_sha_is_order_independent_and_text_sensitive(self):
        a = em.texts_sha256(self.ids, self.texts)
        self.assertEqual(a, em.texts_sha256(["a", "b", "c"], ["world", "hello", "hello world"]))
        self.assertNotEqual(a, em.texts_sha256(self.ids, ["hello!", "world", "hello world"]))

    def test_select_order_and_missing(self):
        self.write()
        art = em.load_artifact(self.out, "x")
        np.testing.assert_array_equal(art.select(["c", "b", "c"]), self.matrix[[2, 0, 2]])
        with self.assertRaises(em.MissingRowError):
            art.select(["a", "zzz"])
        self.assertFalse(art.matrix.flags.writeable)
        self.assertIn("a", art)

    def test_immutable(self):
        self.write()
        with self.assertRaises(em.ArtifactExistsError):
            self.write()
        self.write(overwrite=True)

    def test_tamper_npy(self):
        self.write()
        p = self.out / "x.npy"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(bytes(data))
        with self.assertRaises(em.ArtifactCorruptError):
            em.load_artifact(self.out, "x")

    def test_tamper_sidecar_and_ids(self):
        self.write()
        p = self.out / "x.json"
        side = json.loads(p.read_text())
        side["row_ids"] = ["b", "a", "d"]
        p.write_text(json.dumps(side))
        with self.assertRaises(em.ArtifactCorruptError):
            em.load_artifact(self.out, "x")
        side["sidecar_sha256"] = "0" * 64
        p.write_text(json.dumps(side))
        with self.assertRaises(em.ArtifactCorruptError):
            em.load_artifact(self.out, "x")

    def test_missing_files_and_truncated_json(self):
        with self.assertRaises(em.ArtifactCorruptError):
            em.load_artifact(self.out, "nope")
        self.write()
        (self.out / "x.json").write_text("{")
        with self.assertRaises(em.ArtifactCorruptError):
            em.load_artifact(self.out, "x")

    def test_expected_ids_and_fingerprint(self):
        self.write()
        with self.assertRaises(em.ArtifactMismatchError):
            em.load_artifact(self.out, "x", expected_row_ids=["a", "q"])
        em.load_artifact(self.out, "x", expected_row_ids=["a"])  # subset of a larger artifact is fine
        with self.assertRaises(em.FingerprintMismatchError):
            em.load_artifact(self.out, "x", expected_fingerprint={"kind": "file", "sha256": "0" * 64})
        with self.assertRaises(em.ArtifactMismatchError):
            em.load_artifact(self.out, "x", expected_texts_sha256="0" * 64)

    def test_unresolved_fails_closed(self):
        unresolved = {"kind": "hf", "model": "m", "revision": "unresolved"}
        meta = {**self.meta, "fingerprint": unresolved}
        em.write_artifact(self.out, "u", self.ids, self.matrix, meta)
        with self.assertRaises(em.FingerprintMismatchError):
            em.load_artifact(self.out, "u")
        with self.assertRaises(em.FingerprintMismatchError):
            em.load_artifact(self.out, "u", expected_fingerprint=unresolved)
        em.load_artifact(self.out, "u", expected_fingerprint=unresolved, allow_unresolved=True)

    def test_non_production_refused(self):
        em.write_artifact(self.out, "n", self.ids, self.matrix, {**self.meta, "non_production": True})
        with self.assertRaises(em.NonProductionError):
            em.load_artifact(self.out, "n")
        self.assertTrue(em.load_artifact(self.out, "n", allow_non_production=True).meta["non_production"])

    def test_textprep_version_mismatch(self):
        self.write()
        with mock.patch.object(em, "TEXTPREP_VERSION", "999"):
            with self.assertRaises(em.ArtifactMismatchError):
                em.load_artifact(self.out, "x")

    def test_write_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            em.write_artifact(self.out, "b", self.ids, self.matrix.astype(np.float64), self.meta)
        bad = self.matrix.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            em.write_artifact(self.out, "b", self.ids, bad, self.meta)
        with self.assertRaises(ValueError):
            em.write_artifact(self.out, "b", ["a", "a", "c"], self.matrix, self.meta)
        with self.assertRaises(ValueError):
            em.write_artifact(self.out, "b", self.ids[:2], self.matrix, self.meta)
        with self.assertRaises(ValueError):
            em.write_artifact(self.out, "b", self.ids, self.matrix, {"corpus": "c"})
        self.assertFalse((self.out / "b.json").exists())

    def test_failed_write_leaves_no_partial_commit(self):
        with mock.patch.object(em.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.write()
        self.assertFalse((self.out / "x.json").exists())
        self.assertEqual(list(self.out.glob(".*")), [])

    def test_artifact_name_is_filesystem_safe(self):
        self.assertEqual(em.artifact_name("corpus", "ATT&CK-BERT"), "corpus__ATT_CK-BERT")


if __name__ == "__main__":
    unittest.main()
