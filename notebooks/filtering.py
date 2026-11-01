"""Relevance-filter experiment logic (leakage-free selection, tuning, refit, metrics).

Generic: project values (embedders, classifier families, search spaces, max_evals) come from
``notebooks/filter.toml``. Protocol (see ``notebooks/AGENTS.md``):

* every candidate (embedder x family, a ``StandardScaler`` + classifier ``Pipeline``) is scored with
  grouped 5-fold CV on the TRAIN rows, folds = the manifest ``inner_fold`` (``PredefinedSplit``);
  the scaler is fitted inside the pipeline on each fold's training part only;
* the best candidate is the highest mean balanced accuracy (ties: lower std, then name);
* its family is tuned with hyperopt TPE, objective = mean grouped-CV balanced accuracy on train;
* the tuned pipeline is refit on all train rows and evaluated once on the test rows, with 95%
  group-bootstrap confidence intervals (whole manifest groups are resampled; Cotov has no groups, so rows);
* CV scores are selection diagnostics (the best of 28 is optimistically biased); fold metrics are
  aggregated as the mean (and std, ddof=1) over the 5 folds.

Functions that fit or choose take the train ids bound to their X (same length and order, checked) and
the test ids (``forbidden_ids``) and raise ``LeakageError`` if they overlap, so test rows can never
enter a fit, CV, tuning or selection call.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

TABLE_INDEX_NAME = "Model (CV: selection diagnostics, mean over the 5 folds)"
METRICS = ("balanced_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc")
_ESTIMATORS = {  # class name (as written in filter.toml) -> module
    "LogisticRegression": "sklearn.linear_model",
    "GaussianNB": "sklearn.naive_bayes",
    "RandomForestClassifier": "sklearn.ensemble",
    "XGBClassifier": "xgboost",
}


class FilterError(Exception):
    pass


class LeakageError(FilterError):
    """Test rows would enter a fit, CV, tuning or selection call."""


# --------------------------------------------------------------------------- config
def load_config(path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def effective_config(cfg: Mapping, smoke: bool = False) -> Dict[str, Any]:
    """Config with the ``[smoke]`` overrides applied when ``smoke`` (tiny, NON-PRODUCTION)."""
    out = json.loads(json.dumps(cfg))
    out["smoke"] = bool(smoke)
    if smoke:
        s = cfg["smoke"]
        out["selection"]["embedders"] = list(s["embedders"])
        out["selection"]["families"] = list(s["families"])
        out["tuning"]["max_evals"] = int(s["max_evals"])
        out["bootstrap"]["B"] = int(s.get("bootstrap_B", out["bootstrap"]["B"]))
    return out


# --------------------------------------------------------------------------- guards
def assert_no_test(ids: Iterable, forbidden_ids: Iterable, what: str = "fit") -> None:
    """Raise ``LeakageError`` if any of ``ids`` is a test row id."""
    forbidden = {str(i) for i in forbidden_ids}
    hit = sorted({str(i) for i in ids} & forbidden)
    if hit:
        raise LeakageError(f"{what}: {len(hit)} test row ids among the rows used, e.g. {hit[:3]}")


def check_bound(X, y, folds, ids, forbidden_ids, what: str) -> None:
    """X, y, folds and the train id list must describe the same rows (length) and no id may be a test id."""
    if ids is None:
        raise FilterError(f"{what}: the train id list bound to X is required")
    ids = list(ids)
    assert_no_test(ids, forbidden_ids, what)
    if not (len(ids) == len(X) == len(y) == len(folds)):
        raise FilterError(f"{what}: X ({len(X)}), y ({len(y)}), folds ({len(folds)}) and train ids ({len(ids)}) differ in length")
    if len(set(ids)) != len(ids):
        raise FilterError(f"{what}: duplicate train ids")


def require_cotov(smoke: bool, allow_missing: bool, problem: Optional[str]) -> Optional[str]:
    """Cotov policy. ``problem`` is None when labels and artifact are available (returns None).
    Otherwise a smoke run or ``FILTER_ALLOW_NO_COTOV`` yields the reason (recorded as ``cotov: blocked``);
    a production run without the override raises, so it can never finish COMPLETE without Cotov."""
    if problem is None:
        return None
    if smoke or allow_missing:
        return problem
    raise FilterError(f"Cotov evaluation unavailable ({problem}); set FILTER_ALLOW_NO_COTOV=1 to proceed without it")


STATUS_COMPLETE = "complete"
STATUS_COMPLETE_NO_COTOV = "complete-cotov-blocked"
COMPLETE_STATUSES = (STATUS_COMPLETE, STATUS_COMPLETE_NO_COTOV)


def staged(path) -> str:
    """Temporary name next to ``path``; outputs are written here and renamed only after validation."""
    return str(path) + ".tmp"


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def publish_filter(pickle_path, csv_path, sidecar: Mapping, blocked_reason: Optional[str] = None) -> Dict[str, Any]:
    """Atomically rename the staged pickle and CSV (``staged(path)``) to their final names and write the
    sidecar last, carrying ``status`` ``complete`` or, with ``blocked_reason`` (Cotov skipped on purpose),
    ``complete-cotov-blocked``. Call it only after the Cotov evaluation succeeded or was explicitly blocked:
    nothing is written under the final names before. The sidecar's ``pickle_sha256`` must match the staged pickle."""
    sp, sc = staged(pickle_path), staged(csv_path)
    for f in (sp, sc):
        if not os.path.exists(f):
            raise FilterError(f"staged file missing: {f}")
    if sidecar.get("pickle_sha256") != _sha256(sp):
        raise FilterError("sidecar pickle_sha256 does not match the staged pickle")
    doc = dict(sidecar)
    doc["status"] = STATUS_COMPLETE if blocked_reason is None else STATUS_COMPLETE_NO_COTOV
    sidecar_path = str(pickle_path) + ".json"
    tmp_sidecar = staged(sidecar_path)
    Path(tmp_sidecar).write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(sp, pickle_path)
    os.replace(sc, csv_path)
    os.replace(tmp_sidecar, sidecar_path)
    return doc


def load_deployed_filter(path):
    """Load the deployed filter through its sidecar: the pickle sha256 must match and the sidecar status must be
    ``complete*``. Returns ``(model, metadata)``. All consumers must use this instead of a bare ``joblib.load``."""
    import joblib

    side = Path(str(path) + ".json")
    if not side.exists():
        raise FilterError(f"no sidecar {side}; refusing to load an unvalidated filter")
    meta = json.loads(side.read_text())
    status = meta.get("status")
    if status not in COMPLETE_STATUSES:
        raise FilterError(f"filter sidecar status is {status!r}, not complete; refusing to load {path}")
    if meta.get("pickle_sha256") != _sha256(path):
        raise FilterError(f"pickle sha256 does not match its sidecar; refusing to load {path}")
    return joblib.load(path), meta


def refuse_production_output(path, production_path) -> None:
    """Smoke runs must never write the production file (same resolved path or same name)."""
    if production_path and (Path(path).resolve() == Path(production_path).resolve()
                            or Path(path).name == Path(production_path).name):
        raise FilterError(f"smoke run refuses to write the production output {production_path}")


# --------------------------------------------------------------------------- split
@dataclass(frozen=True)
class SplitData:
    train_ids: List[str]
    y_train: np.ndarray
    folds: np.ndarray
    test_ids: List[str]
    y_test: np.ndarray


def to_binary(values: Iterable, as_bool: Callable) -> np.ndarray:
    return np.array([1 if as_bool(v) else 0 for v in values], dtype=np.int64)


def prepare_split(train_df, test_df, manifest, id_col: str, label_col: str, as_bool: Callable) -> SplitData:
    """Ids, binary labels and manifest inner folds of the train rows; ids and labels of the test rows.

    The manifest must say ``train`` with a fold >= 0 for every train row and ``test`` for every test
    row (verify the manifest against the corpus first, ``splits.verify_manifest``)."""
    m = manifest.set_index(manifest["row_id"].astype(str))
    tr = train_df[id_col].astype(str).tolist()
    te = test_df[id_col].astype(str).tolist()
    assert_no_test(tr, te, "split")
    if len(set(tr)) != len(tr) or len(set(te)) != len(te):
        raise FilterError("duplicate row ids in the split files")
    try:
        if not (m.loc[tr, "split"] == "train").all() or not (m.loc[te, "split"] == "test").all():
            raise FilterError("a CSV row's manifest split does not match its file")
    except KeyError as e:
        raise FilterError(f"row id missing from the manifest: {e}") from e
    folds = m.loc[tr, "inner_fold"].to_numpy(dtype=np.int64)
    if folds.min() < 0:
        raise FilterError("train rows must have inner_fold >= 0")
    return SplitData(tr, to_binary(train_df[label_col], as_bool), folds, te, to_binary(test_df[label_col], as_bool))


def cotov_rows(df, as_label: str = "Related", text_col: str = "full_text", lang_col: str = "lang"):
    """English rows of the Cotov CSV: ids ``cotov-<csv position>``, raw texts, labels in {0,1}.

    ``Related`` must be 0/1 (int or '0'/'1'); anything else raises."""
    ids, texts, y = [], [], []
    for pos, (lang, text, rel) in enumerate(zip(df[lang_col], df[text_col], df[as_label])):
        if lang != "en":
            continue
        r = str(rel).strip()
        if r not in ("0", "1"):
            raise FilterError(f"Cotov row {pos}: {as_label} must be 0 or 1, got {rel!r}")
        ids.append(f"cotov-{pos}")
        texts.append(text)
        y.append(int(r))
    return ids, texts, np.array(y, dtype=np.int64)


# --------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    name: str
    embedder: str
    family: str


def build_candidates(cfg: Mapping) -> List[Candidate]:
    """Embedders x families (embedder-major, config order); names like ``XGB SBERT``."""
    sel = cfg["selection"]
    unknown = [f for f in sel["families"] if f not in cfg["families"]]
    if unknown:
        raise FilterError(f"families without a [families.*] entry: {unknown}")
    return [Candidate(f"{fam.upper()} {emb}", emb, fam) for emb in sel["embedders"] for fam in sel["families"]]


def make_estimator(cfg: Mapping, family: str, seed: int, device: Optional[str] = None, params: Optional[Mapping] = None):
    """Classifier of ``family`` with the fixed config arguments overlaid by ``params`` (tuned values)."""
    spec = cfg["families"][family]
    cls = getattr(importlib.import_module(_ESTIMATORS[spec["estimator"]]), spec["estimator"])
    kw = {**spec.get("params", {}), **(params or {})}
    sig = cls().get_params()
    if "random_state" in sig:
        kw["random_state"] = seed
    if spec["estimator"] == "XGBClassifier" and device is not None:
        kw["device"] = device
    return cls(**kw)


def make_pipeline(cfg: Mapping, family: str, seed: int, device: Optional[str] = None, params: Optional[Mapping] = None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("estimator", make_estimator(cfg, family, seed, device, params))])


# --------------------------------------------------------------------------- metrics and CV
def metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def bootstrap_ci(y_true, y_pred, y_score, groups=None, *, B: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    """Percentile bootstrap CIs of every metric. ``groups`` (one label per row) -> whole groups are
    resampled with replacement (the manifest groups for the test split); ``None`` -> rows are
    resampled (Cotov has no groups). Seeded; replicates with a single class (no AUC) are skipped."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    n = len(y_true)
    rng = np.random.default_rng(seed)
    if groups is not None:
        _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        cuts = np.cumsum(np.bincount(inv))[:-1]
        members = np.split(order, cuts)
        n_units = len(members)
    else:
        members, n_units = None, n
    draws: Dict[str, List[float]] = {m: [] for m in METRICS}
    skipped = 0
    for _ in range(B):
        if members is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, n_units, n_units)])
        if len(np.unique(y_true[idx])) < 2:
            skipped += 1
            continue
        m = metrics(y_true[idx], y_pred[idx], y_score[idx])
        for k in METRICS:
            draws[k].append(m[k])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ci = {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [float("nan")] * 2
          for k, v in draws.items()}
    return {"ci": ci, "level": level, "B": int(B), "seed": int(seed), "unit": "group" if members is not None else "row",
            "n_units": int(n_units), "n_replicates": int(B - skipped)}


def evaluate(pipeline, X, y, groups=None, *, B: int = 0, seed: int = 0) -> Dict[str, Any]:
    """Metrics (AUC from predict_proba), prevalence and confusion matrix [[tn, fp], [fn, tp]] of a
    FITTED pipeline; with ``B > 0`` also 95% bootstrap CIs (``bootstrap_ci``; ``groups`` or rows)."""
    from sklearn.metrics import confusion_matrix
    pred = pipeline.predict(X)
    score = pipeline.predict_proba(X)[:, 1]
    out = metrics(y, pred, score)
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    out["n"] = int(len(y))
    out["prevalence"] = float(np.mean(y))
    if B > 0:
        out["bootstrap"] = bootstrap_ci(y, pred, score, groups, B=B, seed=seed)
    return out


def grouped_cv(pipeline, X, y, folds, ids, forbidden_ids) -> Dict[str, Any]:
    """Grouped CV of ``pipeline`` over (X, y) with predefined ``folds`` (>= 0; PredefinedSplit).

    The pipeline (scaler included) is cloned and fitted on each fold's training part only.
    Returns per-fold metrics, mean/std (ddof=1) per metric and the fit seconds per fold."""
    from sklearn.base import clone
    from sklearn.model_selection import PredefinedSplit
    folds = np.asarray(folds)
    check_bound(X, y, folds, ids, forbidden_ids, "grouped_cv")
    if folds.min() < 0:
        raise FilterError("grouped_cv needs one fold >= 0 per training row")
    per_fold: Dict[str, List[float]] = {m: [] for m in METRICS}
    seconds: List[float] = []
    for tr, va in PredefinedSplit(test_fold=folds).split(X, y):
        est = clone(pipeline)
        t0 = time.perf_counter()
        est.fit(X[tr], y[tr])
        seconds.append(time.perf_counter() - t0)
        m = metrics(y[va], est.predict(X[va]), est.predict_proba(X[va])[:, 1])
        for k in METRICS:
            per_fold[k].append(m[k])
    return {
        "folds": per_fold,
        "mean": {k: float(np.mean(v)) for k, v in per_fold.items()},
        "std": {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in per_fold.items()},
        "fit_seconds": seconds,
    }


def rank_candidates(records: Mapping[str, Mapping], metric: str = "balanced_accuracy") -> List[str]:
    """Candidate names best first: highest mean ``metric`` (rounded to 12 digits), then lower
    std of that metric, then name (ascending) -- fully deterministic."""
    def key(name):
        r = records[name]
        return (-round(r["mean"][metric], 12), round(r["std"][metric], 12), name)
    return sorted(records, key=key)


def select_best(records: Mapping[str, Mapping], metric: str = "balanced_accuracy") -> str:
    if not records:
        raise FilterError("no candidate records")
    return rank_candidates(records, metric)[0]


def performance_table(records: Mapping[str, Mapping], order: Optional[Sequence[str]] = None):
    """Per-candidate CV table in percent (2 decimals): mean of every metric, std of the selection
    metric. Columns: CV Balanced Accuracy, CV Std, Accuracy, Precision, Recall, F1, ROC AUC.
    The index header marks them as selection diagnostics (means over the folds)."""
    import pandas as pd
    rows = {}
    for name in (order or list(records)):
        r = records[name]
        rows[name] = {
            "CV Balanced Accuracy": r["mean"]["balanced_accuracy"], "CV Std": r["std"]["balanced_accuracy"],
            "Accuracy": r["mean"]["accuracy"], "Precision": r["mean"]["precision"],
            "Recall": r["mean"]["recall"], "F1": r["mean"]["f1"], "ROC AUC": r["mean"]["roc_auc"],
        }
    df = (pd.DataFrame.from_dict(rows, orient="index") * 100).round(2)
    df.index.name = TABLE_INDEX_NAME
    return df


# --------------------------------------------------------------------------- search spaces
def build_space(spec: Mapping[str, Mapping]) -> Dict[str, Any]:
    """Hyperopt space from the config table of one family (see filter.toml)."""
    from hyperopt import hp
    space = {}
    for name, s in spec.items():
        t = s["type"]
        if t == "uniform":
            space[name] = hp.uniform(name, s["low"], s["high"])
        elif t == "quniform":
            space[name] = hp.quniform(name, s["low"], s["high"], s["q"])
        elif t == "loguniform":
            space[name] = hp.loguniform(name, math.log(s["low"]), math.log(s["high"]))
        elif t == "choice":
            space[name] = hp.choice(name, list(s["options"]))
        else:
            raise FilterError(f"{name}: unknown space type {t!r}")
    return space


def cast_params(params: Mapping, spec: Mapping[str, Mapping]) -> Dict[str, Any]:
    """Sampled values as constructor arguments (``int = true`` entries cast to int)."""
    return {k: (int(v) if spec[k].get("int") else (float(v) if isinstance(v, (float, np.floating)) else v))
            for k, v in params.items()}


# --------------------------------------------------------------------------- checkpointed work
def candidate_key(name: str):
    return ("filter-cv", name)


def trial_key(model: str, index: int):
    return ("filter-tune", model, f"{index:04d}")


def run_candidates(candidates: Sequence[Candidate], cfg: Mapping, features: Callable[[str, Sequence[str]], np.ndarray], y, folds,
                   train_ids, forbidden_ids, *, seed: int, device: Optional[str], store, budget,
                   log: Callable = print) -> Dict[str, Dict]:
    """Score every candidate not yet checkpointed (one checkpoint per candidate).

    ``features(embedder, train_ids)`` returns the matrix of exactly those TRAIN ids, in order
    (length checked against ``y``/``folds``/``train_ids`` before any fit). Returns the
    records of all candidates available (done earlier or now); stops cleanly, leaving the rest
    missing, when the budget does not allow another candidate (estimate = longest so far)."""
    assert_no_test(train_ids, forbidden_ids, "run_candidates")
    train_ids = list(train_ids)
    records: Dict[str, Dict] = {}
    for c in candidates:
        key = candidate_key(c.name)
        if store.has(key):
            records[c.name] = store.load(key)
            log(f"· {c.name}: checkpoint")
            continue
        if not budget.allows(None, kind="candidate"):
            log(f"BUDGET: stopping before {c.name}")
            break
        t0 = time.monotonic()
        res = grouped_cv(make_pipeline(cfg, c.family, seed, device), features(c.embedder, train_ids), y, folds,
                         train_ids, forbidden_ids)
        res.update(name=c.name, embedder=c.embedder, family=c.family)
        store.save(key, res)
        budget.record("candidate", time.monotonic() - t0)
        records[c.name] = res
        m, s = res["mean"]["balanced_accuracy"], res["std"]["balanced_accuracy"]
        log(f"· {c.name}: CV balanced accuracy {m * 100:.2f} +- {s * 100:.2f} ({time.monotonic() - t0:.1f}s)")
    return records


def run_tuning(model: str, family: str, cfg: Mapping, X, y, folds, train_ids, forbidden_ids, *, seed: int,
               device: Optional[str], store, budget, max_evals: int, log: Callable = print) -> Dict[str, Any]:
    """Hyperopt TPE over the family's space, one checkpoint per evaluation.

    Evaluation i draws with ``rstate = default_rng([seed, i])``, so a resumed run follows exactly
    the trajectory of an uninterrupted one. Returns ``{'complete', 'n_done', 'best_params',
    'best_cv', 'trials'}`` (``best_*`` None while no trial exists)."""
    from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
    check_bound(X, y, folds, train_ids, forbidden_ids, "run_tuning")
    spec = cfg["spaces"][family]
    space = build_space(spec)

    def objective(params):
        p = cast_params(params, spec)
        res = grouped_cv(make_pipeline(cfg, family, seed, device, p), X, y, folds, train_ids, forbidden_ids)
        return {"loss": -res["mean"]["balanced_accuracy"], "status": STATUS_OK,
                "std": res["std"]["balanced_accuracy"], "mean": res["mean"], "fit_seconds": res["fit_seconds"]}

    trials = Trials()
    docs = []
    while store.has(trial_key(model, len(docs))):
        docs.append(store.load(trial_key(model, len(docs))))
    if docs:
        trials.insert_trial_docs(docs)
        trials._ids.update(d["tid"] for d in docs)
        trials.refresh()
        log(f"resumed {len(docs)} tuning trials from checkpoints")
    while len(trials.trials) < max_evals:
        i = len(trials.trials)
        if not budget.allows(None, kind="trial"):
            log(f"BUDGET: stopping tuning at {i}/{max_evals} trials")
            break
        t0 = time.monotonic()
        fmin(objective, space, algo=tpe.suggest, max_evals=i + 1, trials=trials,
             rstate=np.random.default_rng([seed, i]), show_progressbar=False, verbose=False)
        doc = trials.trials[-1]
        store.save(trial_key(model, i), doc)
        budget.record("trial", time.monotonic() - t0)
        log(f"· trial {i + 1}/{max_evals}: CV balanced accuracy {-doc['result']['loss'] * 100:.2f} "
            f"({time.monotonic() - t0:.1f}s)")
    out = {"complete": len(trials.trials) >= max_evals, "n_done": len(trials.trials),
           "best_params": None, "best_cv": None, "trials": trials}
    if trials.trials:
        best = trials.best_trial  # first minimum
        vals = {k: v[0] for k, v in best["misc"]["vals"].items() if v}
        out["best_params"] = cast_params(space_eval(space, vals), spec)
        out["best_cv"] = {"mean": best["result"]["mean"], "std": best["result"]["std"],
                          "balanced_accuracy_mean": -best["result"]["loss"]}
    return out
