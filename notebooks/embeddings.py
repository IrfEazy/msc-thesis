"""Shared, verifiable text embeddings (one immutable artifact per corpus and embedder).

Every stage and the deployed System embeds text through ``embed`` (which applies
``textprep.preprocess_text`` itself, so callers pass raw text) and reads stored matrices through
``load_artifact``, joining by row id. Labels and the split are never stored in an artifact;
consumers verify the fingerprint and the row-id coverage and fail closed on any mismatch.

Project values (embedder names, dimensions, env var names, encode options, tolerances) live in
``embedders.toml``; the paths and hub ids come from the environment variables it names.

GloVe: the embedding of a text is the SUM of the vectors of its lowercased tokens, tokens being
``re.findall(r"\\w+|[<escaped punctuation>]", text)`` on the preprocessed text (the tokenisation of
``notebooks.utils.preprocess_texts``). Out-of-vocabulary tokens are ignored; a text without any
known token (including the empty text) maps to the zero vector. Sums are accumulated in float64
and stored as float32.

Sentence transformers: the preprocessed text is encoded as is with the options of the config
(an empty text is encoded as the empty string).
"""

import hashlib
import importlib.metadata
import io
import json
import os
import re
import string
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from notebooks.textprep import TEXTPREP_VERSION, preprocess_text

DEFAULT_CONFIG = Path(__file__).with_name("embedders.toml")
KINDS = ("glove", "sentence_transformer")
FORMAT_VERSION = 1
UNRESOLVED = "unresolved"
_ENCODE_KEYS = {"batch_size", "normalize_embeddings", "convert_to_numpy", "show_progress_bar"}
_TOKEN = re.compile(r"\w+|[{}]".format(re.escape(string.punctuation)))


class EmbeddingError(Exception):
    """Base class of every error raised by this module."""


class ConfigError(EmbeddingError):
    """Invalid or incomplete embedder configuration or environment."""


class NonFiniteError(EmbeddingError, ValueError):
    """An embedding matrix contains NaN or infinity."""


class ParityError(EmbeddingError):
    """Re-embedding the same texts with another batch size disagrees beyond the tolerance."""


class ArtifactError(EmbeddingError):
    """Base class of artifact errors."""


class ArtifactExistsError(ArtifactError):
    """The artifact already exists (artifacts are immutable)."""


class ArtifactCorruptError(ArtifactError):
    """Missing, truncated, tampered or internally inconsistent artifact files."""


class ArtifactMismatchError(ArtifactError):
    """The artifact is intact but is not the one the caller expects."""


class FingerprintMismatchError(ArtifactMismatchError):
    """The embedder fingerprint differs from the expected one (or cannot be told apart)."""


class NonProductionError(ArtifactError):
    """Smoke-run artifact refused (pass ``allow_non_production=True`` to read it)."""


class MissingRowError(ArtifactError, LookupError):
    """A requested row id is not in the artifact."""


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class Embedder:
    name: str
    kind: str
    env: str
    dim: Optional[int] = None
    source: Optional[str] = field(default=None, compare=False)  # path or hub id, from the env
    encode: Mapping[str, Any] = field(default_factory=dict)
    parity_rtol: float = 0.0

    def config(self) -> Dict[str, Any]:
        """Path-independent configuration recorded in artifacts (no source value)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "dim": self.dim,
            "env": self.env,
            "encode": dict(self.encode) if self.kind == "sentence_transformer" else None,
        }

    def require_source(self) -> str:
        if not self.source:
            raise ConfigError(f"{self.name}: environment variable {self.env} is not set")
        return self.source


def load_embedders(
    config: os.PathLike = DEFAULT_CONFIG,
    environ: Optional[Mapping[str, str]] = None,
    names: Optional[Iterable[str]] = None,
    require_source: bool = True,
) -> Dict[str, Embedder]:
    """Embedders declared in ``config`` (default ``embedders.toml``), keyed by name in file order.

    ``names`` selects a subset (unknown names raise ``ConfigError``). Sources are read from
    ``environ`` (default ``os.environ``); with ``require_source`` a selected embedder whose env
    var is unset raises ``ConfigError``."""
    environ = os.environ if environ is None else environ
    try:
        with open(config, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"cannot read embedder config {config}: {e}") from e
    defaults = dict(raw.get("encode", {}))
    parity = raw.get("parity", {})
    out: Dict[str, Embedder] = {}
    for entry in raw.get("embedder", []):
        name = entry.get("name")
        kind = entry.get("kind")
        env = entry.get("env")
        if not name or name in out:
            raise ConfigError(f"embedder name missing or duplicated: {name!r}")
        if kind not in KINDS:
            raise ConfigError(f"{name}: kind must be one of {KINDS}, got {kind!r}")
        if not env:
            raise ConfigError(f"{name}: `env` is required")
        dim = entry.get("dim")
        if kind == "glove" and (not isinstance(dim, int) or dim <= 0):
            raise ConfigError(f"{name}: glove needs a positive integer `dim`")
        encode = {**defaults, **entry.get("encode", {})} if kind == "sentence_transformer" else {}
        if kind == "sentence_transformer":
            unknown = set(encode) - _ENCODE_KEYS
            if unknown or _ENCODE_KEYS - set(encode):
                raise ConfigError(f"{name}: encode options must be exactly {sorted(_ENCODE_KEYS)}")
            if encode["convert_to_numpy"] is not True:
                raise ConfigError(f"{name}: convert_to_numpy must be true")
            if not isinstance(encode["batch_size"], int) or encode["batch_size"] <= 0:
                raise ConfigError(f"{name}: batch_size must be a positive integer")
        rtol = float(parity.get(f"{kind}_rtol", 0.0))
        out[name] = Embedder(name, kind, env, dim, environ.get(env) or None, encode, rtol)
    if names is not None:
        names = list(names)
        missing = [n for n in names if n not in out]
        if missing:
            raise ConfigError(f"unknown embedders {missing}; configured: {list(out)}")
        out = {n: out[n] for n in names}
    if require_source:
        unset = [e.name for e in out.values() if not e.source]
        if unset:
            raise ConfigError(f"no source set for {unset} (environment variables: "
                              f"{[out[n].env for n in unset]})")
    return out


# --------------------------------------------------------------------------- embedding

_GLOVE_CACHE: Dict[tuple, tuple] = {}
_MODEL_CACHE: Dict[tuple, Any] = {}


def _load_glove(path: str, dim: int):
    key = (os.path.abspath(path), dim, os.path.getsize(path), os.path.getmtime(path))
    if key not in _GLOVE_CACHE:
        index: Dict[str, int] = {}
        rows: List[np.ndarray] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                values = line.split()
                if len(values) <= dim:
                    raise ConfigError(f"{path}:{lineno}: expected a word and {dim} values")
                word = " ".join(values[:-dim])
                index[word] = len(rows)  # a repeated word keeps its last vector
                rows.append(np.asarray(values[-dim:], dtype=np.float32))
        _GLOVE_CACHE.clear()  # keep one table in memory
        _GLOVE_CACHE[key] = (index, np.vstack(rows))
    return _GLOVE_CACHE[key]


def _embed_glove(texts: Sequence[str], embedder: Embedder) -> np.ndarray:
    index, table = _load_glove(embedder.require_source(), embedder.dim)
    out = np.zeros((len(texts), embedder.dim), dtype=np.float64)
    for i, text in enumerate(texts):
        for token in _TOKEN.findall(text):
            j = index.get(token.lower())
            if j is not None:
                out[i] += table[j]
    return out


def resolve_device(device: Optional[str] = None) -> str:
    """``device`` if given, else ``cuda`` when torch sees a GPU, else ``cpu``."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sentence_transformer(source: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(source, device=device)


def _get_model(source: str, device: str):
    key = (source, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_sentence_transformer(source, device)
    return _MODEL_CACHE[key]


def embed(
    texts: Sequence[str],
    embedder: Embedder,
    *,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Embed raw ``texts``: ``preprocess_text`` is applied here. Returns a finite float32
    ``(n, d)`` matrix; raises ``TypeError`` for a non-string text, ``NonFiniteError`` for
    NaN/inf output. ``batch_size`` overrides the configured one (transformers only)."""
    cleaned = [preprocess_text(t) for t in texts]
    if embedder.kind == "glove":
        raw = _embed_glove(cleaned, embedder)
    elif embedder.kind == "sentence_transformer":
        model = _get_model(embedder.require_source(), resolve_device(device))
        if not cleaned:
            raw = np.zeros((0, int(model.get_sentence_embedding_dimension())))
        else:
            opts = dict(embedder.encode)
            if batch_size is not None:
                opts["batch_size"] = batch_size
            raw = model.encode(cleaned, **opts)
    else:
        raise ConfigError(f"{embedder.name}: unknown kind {embedder.kind!r}")
    out = np.ascontiguousarray(np.asarray(raw), dtype=np.float32)
    if out.ndim != 2 or out.shape[0] != len(cleaned):
        raise EmbeddingError(f"{embedder.name}: unexpected output shape {out.shape}")
    if embedder.dim is not None and out.shape[1] != embedder.dim:
        raise EmbeddingError(f"{embedder.name}: dimension {out.shape[1]} != configured {embedder.dim}")
    if not np.isfinite(out).all():
        raise NonFiniteError(f"{embedder.name}: non-finite values in the embeddings")
    return out


def parity_check(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    alt_batch_size: int = 7,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Embed ``texts`` twice (configured batch size, then ``alt_batch_size``) and require
    ``max|a-b| <= embedder.parity_rtol * max|a|`` (0 means exactly equal). Raises ``ParityError``."""
    a = embed(texts, embedder, device=device)
    b = embed(texts, embedder, batch_size=alt_batch_size, device=device)
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    tol = embedder.parity_rtol * float(np.max(np.abs(a))) if a.size else 0.0
    result = {"embedder": embedder.name, "rows": len(texts), "max_abs_diff": diff,
              "tolerance": tol, "rtol": embedder.parity_rtol, "alt_batch_size": alt_batch_size}
    if diff > tol:
        raise ParityError(f"{embedder.name}: batch-size parity failed {result}")
    return result


# --------------------------------------------------------------------------- fingerprints


def _sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _dir_sha256(path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        h.update(rel.as_posix().encode() + b"\0" + _sha256_file(p).encode() + b"\0")
    return h.hexdigest()


def _hf_revision(model_id: str, online: bool = True) -> Optional[str]:
    """Commit hash of the local snapshot, else (``online``) of the hub; None if unresolvable."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        pass
    if online:
        try:
            from huggingface_hub import model_info

            return model_info(model_id, timeout=10).sha
        except Exception:
            pass
    return None


_FP_CACHE: Dict[str, Dict[str, str]] = {}


def fingerprint(embedder: Embedder, online: bool = True) -> Dict[str, str]:
    """Path-independent content identity of the embedder input: sha256 of the GloVe file or
    model directory, or ``{"kind": "hf", "model": id, "revision": commit}`` (revision
    ``"unresolved"`` when neither the local cache nor the hub tells)."""
    source = embedder.require_source()
    if embedder.kind == "glove" or os.path.exists(source):
        if not os.path.exists(source):
            raise ConfigError(f"{embedder.name}: {source} does not exist")
        if source not in _FP_CACHE:
            if os.path.isdir(source):
                _FP_CACHE[source] = {"kind": "dir", "sha256": _dir_sha256(source)}
            else:
                _FP_CACHE[source] = {"kind": "file", "sha256": _sha256_file(source)}
        return dict(_FP_CACHE[source])
    return {"kind": "hf", "model": source, "revision": _hf_revision(source, online) or UNRESOLVED}


def fingerprints_match(a: Optional[Mapping], b: Optional[Mapping]) -> Optional[bool]:
    """True/False when both are resolved; None when either revision is ``unresolved``."""
    if a is None or b is None or a.get("kind") != b.get("kind"):
        return False
    if a["kind"] == "hf":
        if a["model"] != b["model"]:
            return False
        field_ = "revision"
    else:
        field_ = "sha256"
    if UNRESOLVED in (a[field_], b[field_]):
        return None
    return a[field_] == b[field_]


# --------------------------------------------------------------------------- artifacts


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def row_ids_sha256(row_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canon(list(row_ids))).hexdigest()


def texts_sha256(row_ids: Sequence[str], texts: Sequence[str]) -> str:
    """sha256 over the raw ``(id, text)`` pairs in id order (independent of the given order)."""
    if len(row_ids) != len(texts):
        raise ValueError("row_ids and texts differ in length")
    h = hashlib.sha256()
    for rid, text in sorted(zip(map(str, row_ids), texts), key=lambda p: p[0]):
        h.update(_canon([rid, text]) + b"\n")
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    out = {}
    for pkg in ("numpy", "torch", "sentence-transformers", "transformers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def artifact_name(corpus: str, embedder_name: str) -> str:
    """File stem of the artifact of ``embedder_name`` over ``corpus`` (filesystem-safe)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{corpus}__{embedder_name}")


def build_meta(
    embedder: Embedder,
    row_ids: Sequence[str],
    texts: Sequence[str],
    *,
    corpus: str,
    device: str,
    fp: Optional[Mapping] = None,
    non_production: bool = False,
    extra: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Metadata for ``write_artifact`` from the raw ``texts`` that were embedded."""
    return {
        "corpus": corpus,
        "embedder": embedder.config(),
        "fingerprint": dict(fp) if fp is not None else fingerprint(embedder),
        "texts_sha256": texts_sha256(row_ids, texts),
        "device": device if embedder.kind == "sentence_transformer" else "n/a",
        "non_production": bool(non_production),
        "extra": dict(extra or {}),
    }


_REQUIRED_META = ("corpus", "embedder", "fingerprint", "texts_sha256", "device", "non_production")


def _paths(directory, name):
    d = Path(directory)
    return d / f"{name}.npy", d / f"{name}.json"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _check_matrix(matrix: np.ndarray, where: str, exc=ArtifactCorruptError) -> None:
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise exc(f"{where}: expected a 2-D float32 matrix, got {matrix.dtype} {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise exc(f"{where}: non-finite values")


def write_artifact(
    directory,
    name: str,
    row_ids: Sequence[str],
    matrix: np.ndarray,
    meta: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Atomically write ``<name>.npy`` (float32) and ``<name>.json`` (sidecar) into ``directory``.

    The sidecar is written last (it commits the artifact) and carries the row ids and their
    sha256, the sha256 of the ``.npy`` bytes, ``meta`` (see ``build_meta``), TEXTPREP_VERSION,
    library versions, dtype, shape, creation time and its own digest. Existing artifacts are
    immutable: ``ArtifactExistsError`` unless ``overwrite``. Returns the sidecar dict."""
    row_ids = [str(r) for r in row_ids]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row ids")
    _check_matrix(matrix, name, ValueError)
    if matrix.shape[0] != len(row_ids):
        raise ValueError(f"{matrix.shape[0]} rows for {len(row_ids)} ids")
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta lacks {missing}")
    npy_path, json_path = _paths(directory, name)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists() and not overwrite:
        raise ArtifactExistsError(f"{json_path} exists; artifacts are immutable")
    buf = io.BytesIO()
    np.save(buf, matrix, allow_pickle=False)
    npy_bytes = buf.getvalue()
    side = {
        "format_version": FORMAT_VERSION,
        "name": name,
        **{k: meta[k] for k in _REQUIRED_META},
        "extra": dict(meta.get("extra", {})),
        "row_count": len(row_ids),
        "row_ids": row_ids,
        "row_ids_sha256": row_ids_sha256(row_ids),
        "textprep_version": TEXTPREP_VERSION,
        "dtype": str(matrix.dtype),
        "shape": list(matrix.shape),
        "npy_sha256": hashlib.sha256(npy_bytes).hexdigest(),
        "versions": library_versions(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    side["sidecar_sha256"] = hashlib.sha256(_canon(side)).hexdigest()
    if json_path.exists():
        json_path.unlink()  # overwrite: drop the commit marker first
    _atomic_write(npy_path, npy_bytes)
    _atomic_write(json_path, json.dumps(side, sort_keys=True, indent=1).encode("utf-8"))
    return side


class EmbeddingArtifact:
    """Verified, read-only embedding matrix addressable by row id."""

    def __init__(self, meta: Dict[str, Any], matrix: np.ndarray):
        self.meta = meta
        self.row_ids = tuple(meta["row_ids"])
        matrix.setflags(write=False)
        self.matrix = matrix
        self._index = {r: i for i, r in enumerate(self.row_ids)}

    @property
    def fingerprint(self) -> Dict[str, str]:
        return self.meta["fingerprint"]

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __len__(self) -> int:
        return len(self.row_ids)

    def __contains__(self, row_id) -> bool:
        return str(row_id) in self._index

    def select(self, ids: Iterable) -> np.ndarray:
        """Rows for ``ids`` in the requested order (a new array); a missing id raises
        ``MissingRowError``."""
        ids = [str(i) for i in ids]
        missing = [i for i in ids if i not in self._index]
        if missing:
            raise MissingRowError(f"{len(missing)} row ids not in the artifact, e.g. {missing[:3]}")
        return self.matrix[[self._index[i] for i in ids]]


def load_artifact(
    directory,
    name: str,
    expected_row_ids: Optional[Iterable] = None,
    expected_fingerprint: Optional[Mapping] = None,
    *,
    expected_texts_sha256: Optional[str] = None,
    allow_non_production: bool = False,
    allow_unresolved: bool = False,
) -> EmbeddingArtifact:
    """Load and verify ``<name>.npy/.json``: sidecar digest, ``.npy`` sha256, dtype/shape/finite,
    row-id list, TEXTPREP_VERSION; then the caller's expectations (every id of
    ``expected_row_ids`` present, fingerprint equal to ``expected_fingerprint``, raw-text digest).
    A fingerprint that cannot be told apart (``unresolved``) fails unless ``allow_unresolved``.
    Smoke artifacts raise ``NonProductionError`` unless ``allow_non_production``."""
    npy_path, json_path = _paths(directory, name)
    try:
        side = json.loads(json_path.read_bytes())
        npy_bytes = npy_path.read_bytes()
    except (OSError, ValueError) as e:
        raise ArtifactCorruptError(f"{name}: cannot read artifact files in {directory}: {e}") from e
    if not isinstance(side, dict) or "sidecar_sha256" not in side:
        raise ArtifactCorruptError(f"{name}: malformed sidecar")
    body = {k: v for k, v in side.items() if k != "sidecar_sha256"}
    if hashlib.sha256(_canon(body)).hexdigest() != side["sidecar_sha256"]:
        raise ArtifactCorruptError(f"{name}: sidecar digest mismatch (edited sidecar)")
    if side.get("format_version") != FORMAT_VERSION or side.get("name") != name:
        raise ArtifactCorruptError(f"{name}: unexpected sidecar format or name")
    if hashlib.sha256(npy_bytes).hexdigest() != side["npy_sha256"]:
        raise ArtifactCorruptError(f"{name}: .npy sha256 mismatch (edited or truncated)")
    try:
        matrix = np.load(io.BytesIO(npy_bytes), allow_pickle=False)
    except Exception as e:
        raise ArtifactCorruptError(f"{name}: unreadable .npy: {e}") from e
    _check_matrix(matrix, name)
    if str(matrix.dtype) != side["dtype"] or list(matrix.shape) != side["shape"]:
        raise ArtifactCorruptError(f"{name}: dtype/shape differ from the sidecar")
    ids = side["row_ids"]
    if (len(ids) != side["row_count"] or len(ids) != matrix.shape[0] or len(set(ids)) != len(ids)
            or row_ids_sha256(ids) != side["row_ids_sha256"]):
        raise ArtifactCorruptError(f"{name}: row ids inconsistent with the matrix")
    if side["non_production"] and not allow_non_production:
        raise NonProductionError(f"{name}: NON-PRODUCTION artifact (smoke run); "
                                 "pass allow_non_production=True to read it")
    if side["textprep_version"] != TEXTPREP_VERSION:
        raise ArtifactMismatchError(
            f"{name}: built with TEXTPREP_VERSION {side['textprep_version']}, current is {TEXTPREP_VERSION}")
    if expected_texts_sha256 is not None and side["texts_sha256"] != expected_texts_sha256:
        raise ArtifactMismatchError(f"{name}: raw texts differ from the expected corpus")
    if expected_fingerprint is not None:
        match = fingerprints_match(side["fingerprint"], expected_fingerprint)
        if match is False or (match is None and not allow_unresolved):
            raise FingerprintMismatchError(
                f"{name}: fingerprint {side['fingerprint']} vs expected {dict(expected_fingerprint)}"
                + (" (unresolved revision)" if match is None else ""))
    elif not allow_unresolved and side["fingerprint"].get("revision") == UNRESOLVED:
        raise FingerprintMismatchError(f"{name}: stored fingerprint is unresolved")
    art = EmbeddingArtifact(side, matrix)
    if expected_row_ids is not None:
        expected = [str(i) for i in expected_row_ids]
        missing = [i for i in expected if i not in art._index]
        if missing:
            raise ArtifactMismatchError(
                f"{name}: {len(missing)} expected row ids missing from the artifact, e.g. {missing[:3]}")
    return art
