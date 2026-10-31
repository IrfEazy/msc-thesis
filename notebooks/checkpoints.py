"""Resumable, time-budgeted checkpoints for long notebook experiments.

Generic helper (stdlib + numpy only); it knows nothing about notebooks,
methods or datasets.  Typical use in a notebook shard::

    ident = make_identity(code_paths={"utils": "notebooks/utils.py"},
                          data_paths={"train": train_csv}, config=CFG,
                          modules=["sklearn", "numpy"])
    store = Store.from_env(ident)          # CHECKPOINT_DIR + CHECKPOINT_INPUTS
    budget = Budget.from_env()             # CHECKPOINT_BUDGET_HOURS / _MARGIN_MINUTES
    for unit in units:
        key = ("model", unit.method, unit.fold)
        if store.has(key):                 # verified: hashes the data
            continue
        if not budget.allows(None, kind="fit"):      # estimate from recorded max
            break
        t0 = time.monotonic(); obj = fit(unit)
        store.save(key, obj); budget.record("fit", time.monotonic() - t0)
    write_status(store.out_dir, "INCOMPLETE" if missing else "COMPLETE",
                 done=[...], missing=[...])
    # final assembly stage (fail closed):
    verify_complete(store, expected_keys)          # raises IncompleteError

Trust boundary
--------------
Pickles are executable code.  Only mount (``CHECKPOINT_INPUTS``) outputs of
this project's own kernels.  The sha256/identity checks detect accidents
(truncation, mixing incompatible runs), not malicious files: the sidecar is
not signed.  ``npy`` artifacts are loaded with ``allow_pickle=False`` and
object arrays are refused on save.

Single writer
-------------
Exactly one process writes a given ``out_dir`` at a time (one shard per
output directory).  This is assumed, and enforced per key during ``save`` by
an ``O_EXCL`` lock file ``<base>.lock``; a leftover lock (crashed writer)
raises ``LockError`` naming the file, which must be deleted manually after
confirming no writer is alive.

Protocol
--------
* An artifact for key K consists of ``<base>.pkl`` (or ``.npy``) and the
  sidecar ``<base>.json``; ``base`` = readable sanitised prefix + the full
  sha256 hex of the canonical key (JSON of ``["t", [parts]]`` or
  ``["d", [[k, v], ...]]``; parts must be str without path separators, NUL or
  ``..``).  Data is written to a temp file in the same directory, fsynced,
  hashed, and renamed; only then is the sidecar written the same way.  The
  sidecar is the completion marker: data without a sidecar (crash) is ignored
  and overwritten by a later save.
* The sidecar is validated strictly: schema version, format in
  {pickle, npy}, ``data_file`` == ``base`` + the extension of the format, key
  equal to the requested key.
* Committed artifacts are immutable.  ``save`` first checks ``out_dir`` and
  the search dirs: an existing valid identical artifact is a no-op (returns
  False), different content or identity raises ``ConflictError``.  If a
  sidecar in ``out_dir`` has missing/short/mismatching data, ``save`` raises
  ``RepairRequiredError``; the manual fix is to delete that key's ``.json``
  and data file from ``out_dir`` and re-run.  Nothing is repaired silently.
* Reads search ``out_dir`` then ``search_dirs`` (read-only mounts of earlier
  shards; never written).  Every copy found is checked and an invalid
  higher-priority copy fails closed (never falls back to a lower one).  A key
  found in several directories must carry the same sha256 (``DuplicateError``).
* ``has(key)`` is verified by default (hashes every copy) so a unit is only
  skipped when its data is intact; ``has(key, verify=False)`` is the cheap
  mode (sidecar, identity, size only) and must be requested explicitly.
* Identity = version field, sha256 of named code and data files (logical
  names; list form maps to basenames and rejects duplicates; missing files
  raise), the JSON config (NaN/inf rejected) and module versions.  The python
  version is recorded for information and not part of the digest.
* ``CHECKPOINT_STATUS.json`` records COMPLETE / INCOMPLETE / FAILED;
  ``verify_complete`` re-checks the exact expected key set against the store
  (hashing) and is the authority, not the status file.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import re
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2
IDENTITY_VERSION = 1
_FORMATS = {"pickle": ".pkl", "npy": ".npy"}  # allowlist: format -> extension
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
STATUS_FILE = "CHECKPOINT_STATUS.json"
STATES = ("COMPLETE", "INCOMPLETE", "FAILED")
_PICKLE_PROTOCOL = 4


class CheckpointError(Exception):
    """Base class of all checkpoint errors."""


class CorruptArtifactError(CheckpointError):
    """Sidecar or data is unreadable, truncated or fails its sha256."""


class IdentityMismatchError(CheckpointError):
    """Artifact was produced under a different run identity."""


class ConflictError(CheckpointError):
    """Different content (or identity) for an already committed key."""


class DuplicateError(CheckpointError):
    """The same key exists in several directories with different sha256."""


class RepairRequiredError(CorruptArtifactError):
    """Sidecar in out_dir whose data is missing or damaged; needs manual fix."""


class LockError(CheckpointError):
    """Per-key write lock is held (concurrent writer or stale lock file)."""


class IncompleteError(CheckpointError):
    """verify_complete found missing, invalid or duplicate units."""

    def __init__(self, message, missing=(), invalid=None, duplicates=()):
        super().__init__(message)
        self.missing = list(missing)
        self.invalid = dict(invalid or {})
        self.duplicates = list(duplicates)


# --------------------------------------------------------------- identity
def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _canon(obj):
    """Canonical JSON; NaN/inf are rejected (ValueError)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _hash_files(paths):
    """{logical name: sha256}.  Mapping keeps its names; a sequence maps to
    basenames and raises on duplicates.  Missing files raise."""
    if isinstance(paths, (str, os.PathLike)):
        raise TypeError("paths must be a mapping {name: path} or a sequence")
    if isinstance(paths, dict):
        items = [(str(n), p) for n, p in paths.items()]
    else:
        items = [(os.path.basename(os.fspath(p)), p) for p in paths]
    out = {}
    for name, p in items:
        if name in out:
            raise ValueError(f"duplicate logical name in identity: {name}")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"identity file missing: {p}")
        out[name] = file_sha256(p)
    return out


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        mod = sys.modules.get(name)
        return getattr(mod, "__version__", None) if mod else None


_INFORMATIONAL = ("digest", "python")


def identity_digest(identity):
    """sha256 over the identity content (``digest`` and ``python`` ignored)."""
    return hashlib.sha256(
        _canon({k: v for k, v in identity.items() if k not in _INFORMATIONAL}).encode()
    ).hexdigest()


def make_identity(code_paths=(), data_paths=(), config=None, modules=()):
    """Stable identity dict: code/data file hashes, config, module versions.

    ``code_paths``/``data_paths``: ``{logical_name: path}`` or a list of paths
    (named by basename; duplicates raise).  ``config`` must be JSON
    serialisable without NaN/inf.  The result includes ``digest``.
    """
    ident = {
        "identity_version": IDENTITY_VERSION,
        "code": _hash_files(code_paths),
        "data": _hash_files(data_paths),
        "config": json.loads(_canon(config if config is not None else {})),
        "versions": {m: _version(m) for m in modules},
        "python": sys.version.split()[0],
    }
    ident["digest"] = identity_digest(ident)
    return ident


# ------------------------------------------------------------------- keys
def _check_part(p):
    if not isinstance(p, str):
        raise TypeError("key parts must be str")
    if "/" in p or "\\" in p or "\x00" in p or ".." in p:
        raise ValueError(f"key part {p!r} contains a path separator, '..' or NUL")
    return p


def _normalize_key(key):
    """-> (canonical JSON str, readable parts) for a tuple/list/dict of str."""
    if isinstance(key, str):
        key = (key,)
    if isinstance(key, dict):
        items = sorted((_check_part(k), _check_part(v)) for k, v in key.items())
        if not items:
            raise TypeError("key dict must not be empty")
        return _canon(["d", [list(i) for i in items]]), [f"{k}={v}" for k, v in items]
    key = tuple(key)
    if not key:
        raise TypeError("key must be a non-empty tuple of str or a dict of str")
    parts = [_check_part(p) for p in key]
    return _canon(["t", parts]), parts


def _key_from_canon(canon):
    kind, val = json.loads(canon)
    return dict(map(tuple, val)) if kind == "d" else tuple(val)


def key_basename(key):
    """Deterministic safe filename stem: sanitised readable prefix + the full
    sha256 hex of the canonical key."""
    canon, parts = _normalize_key(key)
    prefix = re.sub(r"[^A-Za-z0-9._=-]+", "-", "__".join(parts)).strip("-.")[:60]
    return f"{prefix or 'key'}-{hashlib.sha256(canon.encode()).hexdigest()}"


# ------------------------------------------------------------------ io
def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:  # e.g. Windows
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path, writer):
    """writer(fileobj) -> temp file in same dir, fsync, rename onto path."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(obj, fmt, f):
    if fmt == "npy":
        import numpy as np

        if not isinstance(obj, np.ndarray):
            raise TypeError("fmt='npy' requires a numpy ndarray")
        if obj.dtype.hasobject:
            raise TypeError("object arrays are refused (they need pickle)")
        np.save(f, obj, allow_pickle=False)
    else:
        pickle.dump(obj, f, protocol=_PICKLE_PROTOCOL)


class _HashingWriter:
    """Write-through to ``f`` (or nowhere if None) while hashing."""

    def __init__(self, f=None):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b) if self.f is not None else len(b)


# ------------------------------------------------------------------ store
class Store:
    """Immutable, identity-checked artifact store (see module docstring)."""

    def __init__(self, out_dir, search_dirs=(), identity=None):
        if identity is None:
            raise ValueError("identity is required (use make_identity)")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir.resolve()
        seen, self.search_dirs = {out}, []
        for d in search_dirs:
            r = Path(d).resolve()
            if r not in seen:
                seen.add(r)
                self.search_dirs.append(Path(d))
        self.identity = dict(identity)
        self.digest = identity_digest(self.identity)

    @classmethod
    def from_env(cls, identity):
        return cls(checkpoint_dir_from_env(), input_dirs_from_env(), identity)

    # -- reading
    def _dirs(self):
        return [self.out_dir, *self.search_dirs]

    def _read_sidecar(self, d, key):
        """Strictly validated sidecar of key in d, or None if not committed."""
        canon, _ = _normalize_key(key)
        base = key_basename(key)
        side = Path(d) / (base + ".json")
        if not side.is_file():
            return None
        try:
            meta = json.loads(side.read_text("utf-8"))
            fmt = meta["format"]
            ok = (
                meta["schema"] == SCHEMA_VERSION
                and meta["key"] == canon
                and fmt in _FORMATS
                and meta["data_file"] == base + _FORMATS[fmt]
                and isinstance(meta["sha256"], str)
                and _SHA_RE.match(meta["sha256"])
                and isinstance(meta["identity_digest"], str)
                and isinstance(meta["size"], int)
                and not isinstance(meta["size"], bool)
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise CorruptArtifactError(f"{side}: unreadable sidecar ({e})") from e
        if not ok:
            raise CorruptArtifactError(
                f"{side}: invalid sidecar (schema/key/format/data_file/sha256)"
            )
        meta["_dir"], meta["_side"] = Path(d), side
        return meta

    def _checked(self, d, key, verify):
        """Sidecar of key in d after identity/size (and sha256 if verify)."""
        m = self._read_sidecar(d, key)
        if m is None:
            return None
        if m["identity_digest"] != self.digest:
            raise IdentityMismatchError(
                f"{m['_side']}: identity {m['identity_digest'][:12]} != "
                f"current {self.digest[:12]}"
            )
        data = m["_dir"] / m["data_file"]
        if not data.is_file() or data.stat().st_size != m["size"]:
            raise CorruptArtifactError(f"{data}: missing or wrong size")
        if verify and file_sha256(data) != m["sha256"]:
            raise CorruptArtifactError(f"{data}: sha256 mismatch")
        return m

    def _find(self, key, verify=True):
        """All committed copies across dirs, each checked; any invalid copy
        raises (fail closed, no fallback to a lower-priority dir)."""
        found = [m for d in self._dirs() if (m := self._checked(d, key, verify))]
        shas = {m["sha256"] for m in found}
        if len(shas) > 1:
            raise DuplicateError(
                f"key {key!r} has different sha256 in: "
                + ", ".join(f"{m['_dir']} ({m['sha256'][:12]})" for m in found)
            )
        return found

    def has(self, key, verify=True):
        """True if a committed artifact exists.  ``verify=True`` (default)
        hashes the data of every copy, so a skip is safe; ``verify=False`` is
        the explicit cheap mode (sidecar, identity, size).  Raises on identity
        mismatch, damage in any copy or conflicting duplicates."""
        return bool(self._find(key, verify))

    def verify(self, key):
        """Full check including sha256; returns the sidecar dict.
        Raises KeyError if absent."""
        found = self._find(key, True)
        if not found:
            raise KeyError(key)
        return {k: v for k, v in found[0].items() if not k.startswith("_")}

    def load(self, key):
        found = self._find(key, True)
        if not found:
            raise KeyError(key)
        m = found[0]
        path = m["_dir"] / m["data_file"]
        if m["format"] == "npy":
            import numpy as np

            return np.load(path, allow_pickle=False)
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_keys(self):
        """Keys (tuple or dict) with a valid committed sidecar in any dir
        (cheap checks; invalid or conflicting ones are skipped)."""
        keys, seen = [], set()
        for d in self._dirs():
            if not Path(d).is_dir():
                continue
            for side in sorted(Path(d).glob("*.json")):
                if side.name == STATUS_FILE:
                    continue
                try:
                    canon = json.loads(side.read_text("utf-8"))["key"]
                    key = _key_from_canon(canon)
                    if key_basename(key) + ".json" != side.name or canon in seen:
                        continue
                    if self._find(key, False):
                        seen.add(canon)
                        keys.append(key)
                except (OSError, ValueError, KeyError, TypeError, CheckpointError):
                    continue
        return keys

    # -- writing (out_dir only)
    def _lock(self, base):
        path = self.out_dir / (base + ".lock")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LockError(
                f"{path} exists: another writer is active or a previous one "
                "crashed.  Confirm no process writes this out_dir, then delete "
                "the lock file."
            ) from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return path

    def save(self, key, obj, fmt="pickle"):
        """Commit obj under key.  fmt 'npy' requires a non-object ndarray.
        Returns True if written, False if an identical valid artifact already
        exists (out_dir or search dirs).  Different content or identity raises
        ConflictError; a damaged sidecar/data pair in out_dir raises
        RepairRequiredError (delete that key's files by hand)."""
        if fmt not in _FORMATS:
            raise ValueError(f"fmt must be one of {sorted(_FORMATS)}")
        canon, _ = _normalize_key(key)
        base = key_basename(key)
        data_name = base + _FORMATS[fmt]
        lock = self._lock(base)
        try:
            try:
                found = self._find(key, True)
            except IdentityMismatchError as e:
                raise ConflictError(f"{key!r} committed under another identity: {e}") from e
            except CorruptArtifactError as e:
                try:
                    self._checked(self.out_dir, key, True)
                except CorruptArtifactError:
                    raise RepairRequiredError(
                        f"{key!r}: sidecar in {self.out_dir} has missing or damaged "
                        f"data ({e}).  Not repaired automatically: delete "
                        f"{base}.json and {base}.pkl/.npy from out_dir by hand and "
                        "re-run."
                    ) from e
                raise
            if found:
                sink = _HashingWriter()
                _dump(obj, fmt, sink)
                if sink.h.hexdigest() == found[0]["sha256"] and found[0]["format"] == fmt:
                    return False
                raise ConflictError(
                    f"{key!r} already committed in {found[0]['_dir']} with different "
                    f"content ({found[0]['sha256'][:12]} vs {sink.h.hexdigest()[:12]})"
                )
            hw = {}

            def write(f):
                hw["w"] = _HashingWriter(f)
                _dump(obj, fmt, hw["w"])

            _atomic_write(self.out_dir / data_name, write)  # 1. data
            w = hw["w"]
            meta = {
                "schema": SCHEMA_VERSION,
                "key": canon,
                "format": fmt,
                "data_file": data_name,
                "sha256": w.h.hexdigest(),
                "size": w.n,
                "identity_digest": self.digest,
                "identity": self.identity,
                "created": _now(),
            }
            _atomic_write(
                self.out_dir / (base + ".json"),  # 2. completion marker, last
                lambda f: f.write(json.dumps(meta, indent=1, sort_keys=True).encode()),
            )
            return True
        finally:
            try:
                lock.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------- budget
def _nonneg(name, v):
    v = float(v)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {v}")
    return v


class Budget:
    """Wall-clock budget on a monotonic clock.  ``hours=None`` = unlimited.

    Time already spent before construction counts: pass ``started_at``
    (epoch seconds, e.g. the session start; env ``CHECKPOINT_SESSION_START``)
    and/or ``spent_seconds`` (monotonic offset).  ``margin_minutes`` is kept
    free at the end; ``finalise_minutes`` is additionally reserved by
    ``allows`` for the finalisation stage.
    """

    def __init__(self, hours=None, margin_minutes=0.0, clock=time.monotonic,
                 started_at=None, spent_seconds=0.0, finalise_minutes=0.0,
                 wall=time.time):
        self.hours = None if hours is None else _nonneg("hours", hours)
        self.margin_seconds = _nonneg("margin_minutes", margin_minutes) * 60.0
        self.finalise_seconds = _nonneg("finalise_minutes", finalise_minutes) * 60.0
        spent = _nonneg("spent_seconds", spent_seconds)
        if started_at is not None:
            started_at = float(started_at)
            if not math.isfinite(started_at):
                raise ValueError("started_at must be finite")
            spent += _nonneg("time since started_at", wall() - started_at)
        self._clock = clock
        self._t0 = clock() - spent
        self._stats = {}  # kind -> [count, total, max]

    @classmethod
    def from_env(cls, env=None, **kw):
        env = os.environ if env is None else env
        g = lambda k: env.get(k, "").strip()
        return cls(
            float(g("CHECKPOINT_BUDGET_HOURS")) if g("CHECKPOINT_BUDGET_HOURS") else None,
            float(g("CHECKPOINT_MARGIN_MINUTES") or 0.0),
            started_at=float(g("CHECKPOINT_SESSION_START")) if g("CHECKPOINT_SESSION_START") else None,
            finalise_minutes=float(g("CHECKPOINT_FINALISE_MINUTES") or 0.0),
            **kw,
        )

    def elapsed(self):
        return self._clock() - self._t0

    def remaining(self):
        """Seconds left before the safety margin (inf when unlimited)."""
        if self.hours is None:
            return float("inf")
        return self.hours * 3600.0 - self.margin_seconds - self.elapsed()

    def record(self, kind, seconds):
        seconds = _nonneg("seconds", seconds)
        s = self._stats.setdefault(kind, [0, 0.0, 0.0])
        s[0] += 1
        s[1] += seconds
        s[2] = max(s[2], seconds)

    def stats(self, kind):
        """{'count','mean','max'} or None if nothing recorded."""
        s = self._stats.get(kind)
        return None if not s else {"count": s[0], "mean": s[1] / s[0], "max": s[2]}

    def allows(self, estimate_seconds=None, kind=None):
        """True if a job of the estimated duration fits while leaving the
        finalisation reserve.  With no explicit estimate the recorded max of
        ``kind`` is used (0 if none yet)."""
        if estimate_seconds is None:
            st = self.stats(kind) if kind else None
            estimate_seconds = st["max"] if st else 0.0
        estimate_seconds = _nonneg("estimate_seconds", estimate_seconds)
        return estimate_seconds <= self.remaining() - self.finalise_seconds


# ----------------------------------------------------------------- status
def write_status(out_dir, state, done=(), missing=(), failed=None, identity_digest=None):
    """Atomically write CHECKPOINT_STATUS.json (keys are rendered to str)."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}")
    doc = {
        "schema": SCHEMA_VERSION,
        "state": state,
        "updated": _now(),
        "identity_digest": identity_digest,
        "done": [_render(k) for k in done],
        "missing": [_render(k) for k in missing],
        "failed": {_render(k): str(v) for k, v in (failed or {}).items()},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _atomic_write(
        Path(out_dir) / STATUS_FILE,
        lambda f: f.write(json.dumps(doc, indent=1, sort_keys=True).encode()),
    )
    return doc


def read_status(dir_):
    """Status dict, or None if the file does not exist."""
    p = Path(dir_) / STATUS_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError) as e:
        raise CorruptArtifactError(f"{p}: unreadable status ({e})") from e


def _render(key):
    if isinstance(key, str):
        return key
    try:
        return _normalize_key(key)[0]
    except (TypeError, ValueError):
        return repr(key)


def verify_complete(store, expected_keys, allow_empty=False):
    """Fail closed: every expected key must be committed and fully valid
    (sha256 verified).

    Returns ``{'missing': [], 'invalid': {}}`` when complete, otherwise raises
    IncompleteError listing each missing, invalid (with reason) and duplicate
    key.  An empty expected set raises ValueError unless ``allow_empty``.
    Keys outside the expected set are ignored.
    """
    expected = list(expected_keys)
    if not expected and not allow_empty:
        raise ValueError("expected_keys is empty (pass allow_empty=True to accept)")
    missing, invalid, dups, seen = [], {}, [], set()
    for k in expected:
        r = _render(k)
        try:
            canon = _normalize_key(k)[0]
        except (TypeError, ValueError) as e:
            invalid[r] = f"{type(e).__name__}: {e}"
            continue
        if canon in seen:
            if k not in dups:
                dups.append(k)
            continue
        seen.add(canon)
        try:
            store.verify(k)
        except KeyError:
            missing.append(k)
        except CheckpointError as e:
            invalid[r] = f"{type(e).__name__}: {e}"
    if missing or invalid or dups:
        lines = [
            f"{len(missing)} missing, {len(invalid)} invalid, "
            f"{len(dups)} duplicate of {len(expected)} expected"
        ]
        lines += [f"  missing: {_render(k)}" for k in missing]
        lines += [f"  invalid: {k}: {why}" for k, why in invalid.items()]
        lines += [f"  duplicate: {_render(k)}" for k in dups]
        raise IncompleteError("\n".join(lines), missing, invalid, dups)
    return {"missing": [], "invalid": {}}


# -------------------------------------------------------------------- env
def input_dirs_from_env(env=None):
    """Directories from CHECKPOINT_INPUTS (os.pathsep-separated, globs ok).
    Missing directories are ignored with a warning."""
    env = os.environ if env is None else env
    dirs = []
    for pat in filter(None, env.get("CHECKPOINT_INPUTS", "").split(os.pathsep)):
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        found = [h for h in hits if os.path.isdir(h)]
        if not found:
            warnings.warn(f"CHECKPOINT_INPUTS: no directory matches {pat!r}; ignored")
        dirs += [h for h in found if h not in dirs]
    return dirs


def checkpoint_dir_from_env(env=None):
    env = os.environ if env is None else env
    return env.get("CHECKPOINT_DIR") or "./checkpoints"
