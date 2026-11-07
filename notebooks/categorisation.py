"""Two-Step categorisation experiment logic (leakage-free selection, refit, test evaluation, publishing).

Generic: project values (levels, embedders, methods, classifiers, grids, bootstrap) come from
``notebooks/categorisation.toml``. Protocol (see ``notebooks/rebuild.md``):

* per level the population and labels come from the corpus rows and the taxonomy only: ``kind = "top"``
  uses the gold-relevant rows (``taxonomy.as_bool``) and ``top_labels``; ``kind = "sub"`` the gold-relevant
  rows whose GOLD top labels contain ``parent`` and ``sub_labels``. Label matrices always have the fixed
  taxonomy columns (``taxonomy.classes``). A level with ``population = "all"`` (One-Step top, scheme
  ``one_step``) keeps every row, labels irrelevant ones ``other``, and its test evaluation adds an implicit
  relevance block (``relevance_block``);
* the row ids are bound to the split manifest (train rows ``train`` with ``inner_fold`` >= 0, test rows
  ``test``; the manifest folds must be ``0..cv_folds-1``);
* every fitting partition (each inner-fold training part, and all eligible train rows for the refit) drops
  the rows whose labelset is rare IN THAT PARTITION (``splits.frequent_labelsets``) from fitting only;
  validation rows are all eligible rows of the held-out fold. A class without positives in a fitting
  partition is left out of that fit and predicted as constant negative (score 0), keeping its column;
* every candidate (method x classifier, a ``StandardScaler`` + MLC method ``Pipeline``, scaler fitted on
  the fitting rows only) produces exactly one out-of-fold prediction and score per eligible train row,
  checkpointed per (level, embedder, candidate, fold);
* selection: per embedder, each label takes the candidate with the highest pooled OOF F1
  (``zero_division=0``; ties: the earlier candidate in the canonical config order); the composed OOF
  predictions give the embedder's macro F1 over the fixed classes; the highest wins (ties: the earlier
  embedder in config order). OOF scores are SELECTION SCORES (optimistic after the search), not estimates;
* refit: each selected candidate once on all eligible train rows (rare-labelset filter recomputed there);
* test: one evaluation of the locked categoriser on all eligible test rows (primary) and on the
  thesis-comparable subset (test rows whose labelset is frequent in the eligible train rows), with 95%
  percentile bootstrap CIs over whole manifest groups, conditional on the selected pipeline.

Functions that fit or choose take the train ids bound to their X and the test ids (``forbidden_ids``) and
raise ``LeakageError`` if they overlap.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
import json
import os
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from notebooks import filtering as _fl
from notebooks import splits as _sp
from notebooks import taxonomy as _tx

SELECTION_NOTE = "OOF selection scores (pooled over the inner folds; optimistic after the per-label search)"
STATUS_COMPLETE = "complete"
OK, INFEASIBLE, FAILED = "ok", "infeasible", "failed"
# Deterministic fit failures recorded as 'failed' (anything else propagates and stops the run).
EXPECTED_FIT_ERRORS = (ValueError, TypeError, np.linalg.LinAlgError)
SCALAR_METRICS = ("micro_f1", "macro_f1", "weighted_f1", "subset_accuracy", "hamming_loss", "macro_roc_auc")


class CategorisationError(Exception):
    pass


class LeakageError(CategorisationError, _fl.LeakageError):
    """Test rows would enter a fit, OOF run, selection or refit call."""


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
        for k in ("levels", "embedders", "methods", "classifiers"):
            if k in s:
                out["selection"][k] = list(s[k])
        out["bootstrap"]["B"] = int(s.get("bootstrap_B", out["bootstrap"]["B"]))
    return out


def _import(path: str):
    module, _, name = path.rpartition(".")
    if not module:
        raise CategorisationError(f"{path!r} is not a dotted import path")
    return getattr(importlib.import_module(module), name)


def _accepts(cls, name: str) -> bool:
    return name in inspect.signature(cls.__init__).parameters


# --------------------------------------------------------------------------- guards
def assert_no_test(ids: Iterable, forbidden_ids: Iterable, what: str = "fit") -> None:
    forbidden = {str(i) for i in forbidden_ids}
    hit = sorted({str(i) for i in ids} & forbidden)
    if hit:
        raise LeakageError(f"{what}: {len(hit)} test row ids among the rows used, e.g. {hit[:3]}")


def _check_bound(X, ids, forbidden_ids, what: str) -> List[str]:
    if ids is None:
        raise CategorisationError(f"{what}: the train id list bound to X is required")
    ids = [str(i) for i in ids]
    assert_no_test(ids, forbidden_ids, what)
    if len(ids) != len(X):
        raise CategorisationError(f"{what}: X ({len(X)}) and train ids ({len(ids)}) differ in length")
    if len(set(ids)) != len(ids):
        raise CategorisationError(f"{what}: duplicate train ids")
    return ids


# --------------------------------------------------------------------------- level data
@dataclass(frozen=True)
class LevelData:
    """Eligible rows of one level: ids, fixed-column label matrices, labelsets, manifest folds and groups."""
    level: str
    classes: Tuple[str, ...]
    train_ids: List[str]
    Y_train: np.ndarray
    labelsets_train: List[Tuple[str, ...]]
    folds: np.ndarray
    groups_train: np.ndarray
    test_ids: List[str]
    Y_test: np.ndarray
    labelsets_test: List[Tuple[str, ...]]
    groups_test: np.ndarray
    relevant_train: Optional[np.ndarray] = None   # gold relevance (bool) of the eligible rows, same order
    relevant_test: Optional[np.ndarray] = None


POPULATIONS = ("relevant", "all")


def validate_level_cfg(lvl: Mapping) -> str:
    """Fail closed on an inconsistent level definition; returns the population (default 'relevant').
    'all' keeps every corpus row and is only defined for the One-Step top level (scheme one_step, kind top);
    scheme one_step needs population 'all' (irrelevant rows have no Two-Step labels)."""
    kind, scheme = lvl.get("kind"), lvl.get("scheme")
    if kind not in ("top", "sub"):
        raise CategorisationError(f"unknown level kind {kind!r}")
    pop = lvl.get("population", "relevant")
    if pop not in POPULATIONS:
        raise CategorisationError(f"unknown level population {pop!r} (expected one of {list(POPULATIONS)})")
    if pop == "all" and (kind != "top" or scheme != "one_step"):
        raise CategorisationError("population 'all' requires kind 'top' and scheme 'one_step'")
    if scheme == "one_step" and pop != "all":
        raise CategorisationError("scheme 'one_step' requires population 'all'")
    return pop


def _labels_matrix(labelsets: Sequence[Sequence[str]], classes: Sequence[str]) -> np.ndarray:
    pos = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((len(labelsets), len(classes)), dtype=np.int64)
    for i, ls in enumerate(labelsets):
        for lab in ls:
            if lab not in pos:
                raise CategorisationError(f"label {lab!r} is not a class of this level {list(classes)}")
            Y[i, pos[lab]] = 1
    return Y


def level_rows_relevance(df, level_cfg: Mapping, taxonomy: _tx.Taxonomy, id_col: str, relevant_col: str,
                         watson_col: str) -> Tuple[List[str], List[Tuple[str, ...]], List[bool]]:
    """Ids, labelsets (sorted tuples) and gold relevance of the rows of ``df`` eligible for the level.
    Population 'relevant' (default): gold-relevant rows only; 'all': every row, labels from
    ``top_labels(wl, relevant, scheme)``."""
    kind = level_cfg["kind"]
    pop = validate_level_cfg(level_cfg)
    ids, labelsets, rels = [], [], []
    for rid, rel, raw in zip(df[id_col], df[relevant_col], df[watson_col]):
        rel = _tx.as_bool(rel)
        if pop == "relevant" and not rel:
            continue
        wl = _tx.watson_list(raw)
        top = taxonomy.top_labels(wl, rel if pop == "all" else True, level_cfg["scheme"])
        if kind == "top":
            labels = top
            if "other" in labels and len(labels) > 1:  # invariant of the taxonomy rule: fail closed
                raise CategorisationError(f"gold labelset of row {rid!r} mixes 'other' with real top labels: {sorted(labels)}")
        else:
            if level_cfg["parent"] not in top:
                continue
            labels = taxonomy.sub_labels(wl, level_cfg["parent"])
        ids.append(str(rid))
        labelsets.append(tuple(sorted(labels)))
        rels.append(bool(rel))
    return ids, labelsets, rels


def level_rows(df, level_cfg: Mapping, taxonomy: _tx.Taxonomy, id_col: str, relevant_col: str,
               watson_col: str) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Ids and labelsets (sorted tuples) of the rows of ``df`` eligible for the level (gold labels only)."""
    ids, labelsets, _ = level_rows_relevance(df, level_cfg, taxonomy, id_col, relevant_col, watson_col)
    return ids, labelsets


def prepare_level(train_df, test_df, manifest, level: str, cfg: Mapping, taxonomy: Optional[_tx.Taxonomy] = None,
                  columns: Optional[Mapping[str, str]] = None) -> LevelData:
    """Bind the split files to the manifest (as ``filtering.prepare_split``: no overlap or duplicates, every
    train row ``train`` with ``inner_fold`` >= 0, every test row ``test``, manifest train folds exactly
    ``0..cv_folds-1``) and derive the level's eligible rows and labels. Verify the manifest against the
    corpus first (``splits.verify_manifest``)."""
    tax = taxonomy or _tx.default_taxonomy()
    cols = {**_sp.DEFAULT_COLUMNS, **(columns or {})}
    lvl = cfg["levels"][level]
    m = manifest.set_index(manifest["row_id"].astype(str))
    tr_all = train_df[cols["id"]].astype(str).tolist()
    te_all = test_df[cols["id"]].astype(str).tolist()
    assert_no_test(tr_all, te_all, "split")
    if len(set(tr_all)) != len(tr_all) or len(set(te_all)) != len(te_all):
        raise CategorisationError("duplicate row ids in the split files")
    try:
        if not (m.loc[tr_all, "split"] == "train").all() or not (m.loc[te_all, "split"] == "test").all():
            raise CategorisationError("a CSV row's manifest split does not match its file")
    except KeyError as e:
        raise CategorisationError(f"row id missing from the manifest: {e}") from e
    manifest_folds = sorted(set(m.loc[m["split"] == "train", "inner_fold"].astype(int)))
    if manifest_folds != list(range(int(cfg["selection"]["cv_folds"]))):
        raise CategorisationError(f"manifest inner folds {manifest_folds} != 0..cv_folds-1 "
                                  f"(cv_folds={cfg['selection']['cv_folds']})")
    classes = tuple(tax.classes(_tx.TOP if lvl["kind"] == "top" else lvl["parent"]))
    tr, ls_tr, rel_tr = level_rows_relevance(train_df, lvl, tax, cols["id"], cols["relevant"], cols["watson"])
    te, ls_te, rel_te = level_rows_relevance(test_df, lvl, tax, cols["id"], cols["relevant"], cols["watson"])
    folds = m.loc[tr, "inner_fold"].to_numpy(dtype=np.int64)
    if len(folds) and folds.min() < 0:
        raise CategorisationError("train rows must have inner_fold >= 0")
    return LevelData(level, classes, tr, _labels_matrix(ls_tr, classes), ls_tr, folds,
                     m.loc[tr, "group"].astype(str).to_numpy(), te, _labels_matrix(ls_te, classes), ls_te,
                     m.loc[te, "group"].astype(str).to_numpy(),
                     np.asarray(rel_tr, dtype=bool), np.asarray(rel_te, dtype=bool))


# --------------------------------------------------------------------------- fitting partitions
def frequent_rows(labelsets: Sequence[Tuple[str, ...]], idx: Sequence[int], min_count: int) -> np.ndarray:
    """Positions among ``idx`` whose labelset is frequent IN ``idx`` (counts from those rows only)."""
    idx = np.asarray(idx, dtype=np.int64)
    freq = _sp.frequent_labelsets([labelsets[i] for i in idx], min_count)
    return idx[[labelsets[i] in freq for i in idx]] if len(idx) else idx


@dataclass(frozen=True)
class FoldPlan:
    fold: int
    fit_idx: np.ndarray       # fitting rows after the rare-labelset filter of this partition
    val_idx: np.ndarray       # ALL eligible rows of the held-out fold
    n_partition: int          # fitting partition size before the filter


def plan_folds(data: LevelData, min_count: int) -> List[FoldPlan]:
    """One plan per fold that has eligible validation rows; validation rows cover every train row once."""
    plans = []
    for f in sorted(set(data.folds.tolist())):
        val = np.flatnonzero(data.folds == f)
        part = np.flatnonzero(data.folds != f)
        fit = frequent_rows(data.labelsets_train, part, min_count)
        if len(fit) == 0:
            raise CategorisationError(f"{data.level}: fold {f} fitting partition ({len(part)} rows) has no labelset "
                                      f"with >= {min_count} rows")
        plans.append(FoldPlan(int(f), fit, val, len(part)))
    covered = np.concatenate([p.val_idx for p in plans]) if plans else np.array([], dtype=np.int64)
    if len(covered) != len(data.train_ids) or len(set(covered.tolist())) != len(covered):
        raise CategorisationError("inner folds do not cover every eligible train row exactly once")
    return plans


# --------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    classifier: str
    params: Tuple[Tuple[str, Any], ...] = ()


def build_candidates(cfg: Mapping) -> List[Candidate]:
    """Methods (grid points in config order) x classifiers, in the canonical (tie-break) order."""
    sel = cfg["selection"]
    out = []
    for meth in sel["methods"]:
        spec = cfg["methods"][meth]
        grid = spec.get("grid", {})
        keys = list(grid)
        points = [()] if not keys else [tuple(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
        for point in points:
            label = meth if not point else f"{meth}({','.join(f'{k}={v}' for k, v in point)})"
            for clf in sel["classifiers"]:
                if clf not in cfg["classifiers"]:
                    raise CategorisationError(f"classifier {clf!r} has no [classifiers.*] entry")
                out.append(Candidate(f"{label}[{clf.upper()}]", meth, clf, point))
    names = [c.name for c in out]
    if len(set(names)) != len(names):
        raise CategorisationError("duplicate candidate names")
    return out


def make_classifier(cfg: Mapping, name: str, seed: int):
    spec = cfg["classifiers"][name]
    cls = _import(spec["estimator"])
    kw = dict(spec.get("params", {}))
    if _accepts(cls, "random_state"):
        kw["random_state"] = seed
    est = cls(**kw)
    if spec.get("one_vs_rest"):
        from sklearn.multiclass import OneVsRestClassifier
        est = OneVsRestClassifier(est)
    return est


def make_pipeline(cfg: Mapping, cand: Candidate, seed: int):
    """``StandardScaler`` + MLC method on the candidate's base classifier (unfitted)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    spec = cfg["methods"][cand.method]
    cls = _import(spec["class"])
    kw = {**spec.get("params", {}), **dict(cand.params)}
    if _accepts(cls, "random_state"):
        kw["random_state"] = seed
    return Pipeline([("scaler", StandardScaler()), ("mlc", cls(make_classifier(cfg, cand.classifier, seed), **kw))])


def infeasible_reason(cfg: Mapping, cand: Candidate, Y_fit: np.ndarray) -> Optional[str]:
    """Reason why ``cand`` cannot be fitted on these rows by design (config
    ``feasible_if_labelset_count_reaches``), else None."""
    param = cfg["methods"][cand.method].get("feasible_if_labelset_count_reaches")
    if not param:
        return None
    need = dict(cand.params).get(param, cfg["methods"][cand.method].get("params", {}).get(param))
    top = max(Counter(map(tuple, np.asarray(Y_fit).tolist())).values()) if len(Y_fit) else 0
    if need is not None and top < int(need):
        return f"no labelset reaches {param}={need} (maximum count {top})"
    return None


class ColumnModel:
    """A fitted pipeline on the active columns (classes with positives in its fitting rows), expanded to all
    ``n_classes`` columns; inactive columns are predicted 0 with score 0."""

    def __init__(self, estimator, active: Sequence[int], n_classes: int):
        self.estimator = estimator
        self.active = np.asarray(active, dtype=np.int64)
        self.n_classes = int(n_classes)

    def _expand(self, part, n, dtype):
        out = np.zeros((n, self.n_classes), dtype=dtype)
        if len(self.active):
            out[:, self.active] = np.asarray(part, dtype=dtype).reshape(n, len(self.active))
        return out

    def predict(self, X) -> np.ndarray:
        n = len(X)
        return self._expand(self.estimator.predict(X) if len(self.active) else None, n, np.int64)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        return self._expand(self.estimator.predict_proba(X) if len(self.active) else None, n, np.float64)


def fit_candidate(cfg: Mapping, cand: Candidate, X, Y, fit_idx, seed: int):
    """Fit on ``X[fit_idx]`` (rows already filtered). Returns a ``ColumnModel`` or the infeasibility reason (str)."""
    Xf, Yf = np.asarray(X)[fit_idx], np.asarray(Y)[fit_idx]
    reason = infeasible_reason(cfg, cand, Yf)
    if reason:
        return reason
    active = np.flatnonzero(Yf.sum(axis=0) > 0)
    est = make_pipeline(cfg, cand, seed)
    if len(active):
        est.fit(Xf, Yf[:, active])
    return ColumnModel(est, active, Y.shape[1])


# --------------------------------------------------------------------------- checkpointed OOF runs
def oof_key(level: str, embedder: str, candidate: str, fold: int):
    return ("cat-oof", level, embedder, candidate, f"fold{int(fold)}")


def expected_keys(data: LevelData, embedders: Sequence[str], candidates: Sequence[Candidate]) -> List[tuple]:
    folds = sorted(set(data.folds.tolist()))
    return [oof_key(data.level, e, c.name, f) for e in embedders for c in candidates for f in folds]


@dataclass
class OOFRun:
    records: Dict[str, Dict[str, Any]]   # candidate -> {'status', 'pred', 'score', ...}
    complete: bool
    missing: List[tuple] = field(default_factory=list)


def _assemble(data: LevelData, plans: Sequence[FoldPlan], fold_recs: Mapping[int, Mapping]) -> Dict[str, Any]:
    statuses = {r["status"] for r in fold_recs.values()}
    if INFEASIBLE in statuses:
        return {"status": INFEASIBLE, "reason": {f: r.get("reason") for f, r in fold_recs.items()
                                                 if r["status"] == INFEASIBLE}}
    if FAILED in statuses:
        return {"status": FAILED, "reason": {f: r.get("reason") for f, r in fold_recs.items()
                                             if r["status"] == FAILED and not r.get("skipped")}}
    n, q = data.Y_train.shape
    pred = np.zeros((n, q), dtype=np.int64)
    score = np.zeros((n, q), dtype=np.float64)
    hits = np.zeros(n, dtype=np.int64)
    pos = {r: i for i, r in enumerate(data.train_ids)}
    for p in plans:
        r = fold_recs[p.fold]
        idx = np.array([pos[i] for i in r["val_ids"]], dtype=np.int64)
        pred[idx], score[idx] = r["pred"], r["score"]
        np.add.at(hits, idx, 1)
    if not (hits == 1).all():
        raise CategorisationError(f"OOF coverage broken: {int((hits == 0).sum())} rows uncovered, "
                                  f"{int((hits > 1).sum())} duplicated")
    return {"status": OK, "pred": pred, "score": score,
            "fit_seconds": {f: r.get("fit_seconds") for f, r in fold_recs.items()}}


def run_oof(data: LevelData, embedder: str, candidates: Sequence[Candidate], cfg: Mapping,
            features: Callable[[str, Sequence[str]], np.ndarray], forbidden_ids, *, seed: int, store, budget,
            log: Callable = print) -> OOFRun:
    """Out-of-fold predictions of every candidate for one (level, embedder), one checkpoint per fold.

    ``features(embedder, train_ids)`` returns the matrix of exactly those TRAIN ids, in order. A resumed
    run reuses stored folds (their validation ids and classes must match). PSt-like candidates that cannot
    reach their labelset count in some fold are stored as 'infeasible' in every fold without fitting. Stops
    cleanly (``complete=False``) when the budget does not allow another fit."""
    assert_no_test(data.train_ids, forbidden_ids, "run_oof")
    assert_no_test(data.train_ids, data.test_ids, "run_oof")
    X = np.asarray(features(embedder, data.train_ids))
    _check_bound(X, data.train_ids, forbidden_ids, "run_oof")
    plans = plan_folds(data, int(cfg["selection"]["min_labelset_count"]))
    records: Dict[str, Dict[str, Any]] = {}
    missing: List[tuple] = []
    stopped = False
    for cand in candidates:
        fold_recs: Dict[int, Mapping] = {}
        failed_fold = None   # first fold of this candidate that failed; later folds are not fitted
        reasons = {}
        for p in plans:
            r = infeasible_reason(cfg, cand, data.Y_train[p.fit_idx])
            if r:
                reasons[p.fold] = r
        for p in plans:
            key = oof_key(data.level, embedder, cand.name, p.fold)
            val_ids = [data.train_ids[i] for i in p.val_idx]
            if store.has(key):
                rec = store.load(key)
                if rec.get("classes") != list(data.classes) or rec.get("val_ids") != val_ids:
                    raise CategorisationError(f"checkpoint {key} does not match the current rows or classes")
                fold_recs[p.fold] = rec
                if rec["status"] == FAILED and not rec.get("skipped") and failed_fold is None:
                    failed_fold = p.fold
                continue
            base = {"classes": list(data.classes), "val_ids": val_ids, "n_fit": int(len(p.fit_idx)),
                    "n_partition": p.n_partition}
            if reasons:
                why = reasons.get(p.fold) or f"infeasible in fold(s) {sorted(reasons)}"
                rec = {**base, "status": INFEASIBLE, "reason": why}
                store.save(key, rec)
                fold_recs[p.fold] = rec
                continue
            if failed_fold is not None:   # the candidate already failed: no further fits, failure recorded once
                rec = {**base, "status": FAILED, "skipped": True,
                       "reason": f"skipped: fold {failed_fold} of this candidate already failed"}
                store.save(key, rec)
                fold_recs[p.fold] = rec
                continue
            if stopped or not budget.allows(None, kind="fit"):
                if not stopped:
                    log(f"BUDGET: stopping before {data.level} / {embedder} / {cand.name} fold {p.fold}")
                stopped = True
                missing.append(key)
                continue
            t0 = time.monotonic()
            model = None
            try:   # only the fit is guarded; errors in predict / predict_proba are bugs and propagate
                model = fit_candidate(cfg, cand, X, data.Y_train, p.fit_idx, seed)
            except EXPECTED_FIT_ERRORS as e:
                log(f"{data.level} / {embedder} / {cand.name} fold {p.fold} failed: {type(e).__name__}: {e}")
                rec = {**base, "status": FAILED, "reason": f"{type(e).__name__}: {e}"}
                failed_fold = p.fold
            if model is not None:
                Xv = X[p.val_idx]
                rec = {**base, "status": OK, "pred": model.predict(Xv), "score": model.predict_proba(Xv),
                       "active": model.active.tolist(), "fit_seconds": time.monotonic() - t0}
            store.save(key, rec)
            budget.record("fit", time.monotonic() - t0)
            fold_recs[p.fold] = rec
        if len(fold_recs) == len(plans):
            records[cand.name] = _assemble(data, plans, fold_recs)
            if records[cand.name]["status"] == INFEASIBLE:
                log(f"· {cand.name}: infeasible by design")
    return OOFRun(records, not missing, missing)


# --------------------------------------------------------------------------- metrics
def _prf(Y, P):
    Y, P = np.asarray(Y, dtype=bool), np.asarray(P, dtype=bool)
    tp = (Y & P).sum(axis=0).astype(float)
    fp = (~Y & P).sum(axis=0).astype(float)
    fn = (Y & ~P).sum(axis=0).astype(float)
    return tp, fp, fn


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def label_f1(Y, P) -> np.ndarray:
    """Per-column F1 with zero_division=0 (as sklearn ``f1_score(..., zero_division=0)``)."""
    tp, fp, fn = _prf(Y, P)
    return _div(2 * tp, 2 * tp + fp + fn)


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def multilabel_metrics(Y, P, S, classes: Sequence[str], with_labels: bool = True) -> Dict[str, Any]:
    """Per-label precision/recall/F1/support/prevalence/ROC AUC; micro/macro/weighted F1 (zero_division=0,
    macro over ALL fixed classes), subset accuracy, hamming loss, macro ROC AUC over the labels where it is
    defined (NaN if none)."""
    Y, P, S = np.asarray(Y), np.asarray(P), np.asarray(S)
    n = len(Y)
    tp, fp, fn = _prf(Y, P)
    support = np.asarray(Y).sum(axis=0).astype(float)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    aucs = np.array([_auc(Y[:, j], S[:, j]) for j in range(Y.shape[1])]) if n else np.full(Y.shape[1], np.nan)
    out = {
        "n": int(n),
        "micro_f1": float(_div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float(_div((f1 * support).sum(), support.sum())),
        "subset_accuracy": float((Y == P).all(axis=1).mean()) if n else float("nan"),
        "hamming_loss": float((Y != P).mean()) if n else float("nan"),
        "macro_roc_auc": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
    }
    if with_labels:
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        out["labels"] = {c: {"precision": float(prec[j]), "recall": float(rec[j]), "f1": float(f1[j]),
                             "support": int(support[j]), "prevalence": float(support[j] / n) if n else float("nan"),
                             "roc_auc": float(aucs[j])}
                         for j, c in enumerate(classes)}
    out["_label_f1"] = f1
    return out


def group_bootstrap(Y, P, S, groups, classes: Sequence[str], *, B: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    """95% percentile bootstrap CIs (whole groups resampled with replacement, seeded) of the scalar metrics
    and the per-label F1; NaN replicate values (e.g. AUC without both classes) are skipped per metric."""
    Y, P, S = map(np.asarray, (Y, P, S))
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    n_units = len(members)
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {m: [] for m in SCALAR_METRICS}
    label_draws: List[np.ndarray] = []
    for _ in range(B):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_units, n_units)])
        m = multilabel_metrics(Y[idx], P[idx], S[idx], classes, with_labels=False)
        for k in SCALAR_METRICS:
            if np.isfinite(m[k]):
                draws[k].append(m[k])
        label_draws.append(m["_label_f1"])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def ci(v):
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [float("nan")] * 2

    L = np.array(label_draws) if label_draws else np.zeros((0, len(classes)))
    return {"ci": {k: ci(v) for k, v in draws.items()},
            "label_f1_ci": {c: ci(L[:, j]) for j, c in enumerate(classes)},
            "n_replicates": {k: len(v) for k, v in draws.items()},
            "level": level, "B": int(B), "seed": int(seed), "unit": "group", "n_units": int(n_units)}


# --------------------------------------------------------------------------- selection
@dataclass(frozen=True)
class Selection:
    level: str
    classes: Tuple[str, ...]
    embedder: str
    label_candidates: Tuple[str, ...]          # selected candidate per class (fixed order)
    embedder_scores: Dict[str, float]          # composed OOF macro F1 per embedder (selection score)
    rows: Tuple[Dict[str, Any], ...]           # per (embedder, label) selection table rows
    excluded: Dict[str, Dict[str, str]]        # embedder -> candidate -> status (infeasible / failed)

    def table(self):
        import pandas as pd
        df = pd.DataFrame(list(self.rows))
        df.attrs["note"] = SELECTION_NOTE
        return df


def select(data: LevelData, oof: Mapping[str, Mapping[str, Mapping]], candidate_order: Sequence[str],
           embedder_order: Sequence[str], forbidden_ids=(), allow_failed: bool = False) -> Selection:
    """Per embedder: per label the candidate with the highest pooled OOF F1 (12 digits; ties: earlier in
    ``candidate_order``); composed macro F1 over the fixed classes; best embedder (ties: earlier in
    ``embedder_order``). Every candidate must have a record; 'infeasible' ones are excluded, 'failed' ones
    raise unless ``allow_failed``."""
    assert_no_test(data.train_ids, forbidden_ids, "select")
    Y = data.Y_train
    scores, picks, rows, excluded = {}, {}, [], {}
    for emb in embedder_order:
        recs = oof.get(emb)
        if recs is None:
            raise CategorisationError(f"no OOF records for embedder {emb}")
        miss = [c for c in candidate_order if c not in recs]
        if miss:
            raise CategorisationError(f"{emb}: OOF incomplete, missing {miss[:3]} ({len(miss)})")
        excluded[emb] = {c: recs[c]["status"] for c in candidate_order if recs[c]["status"] != OK}
        failed = [c for c, s in excluded[emb].items() if s == FAILED]
        if failed and not allow_failed:
            raise CategorisationError(f"{emb}: failed candidates {failed[:3]}; set allow_failed to exclude them")
        ok = [c for c in candidate_order if recs[c]["status"] == OK]
        if not ok:
            raise CategorisationError(f"{emb}: no feasible candidate")
        f1 = np.vstack([np.round(label_f1(Y, recs[c]["pred"]), 12) for c in ok])  # (cands, labels)
        best = f1.argmax(axis=0)  # first maximum = earliest candidate in canonical order
        chosen = tuple(ok[b] for b in best)
        composed = np.column_stack([recs[chosen[j]]["pred"][:, j] for j in range(Y.shape[1])])
        m = multilabel_metrics(Y, composed, np.column_stack([recs[chosen[j]]["score"][:, j]
                                                             for j in range(Y.shape[1])]), data.classes)
        scores[emb] = m["macro_f1"]
        picks[emb] = chosen
        for j, c in enumerate(data.classes):
            lab = m["labels"][c]
            rows.append({"embedder": emb, "label": c, "candidate": chosen[j], "oof_f1": lab["f1"],
                         "oof_precision": lab["precision"], "oof_recall": lab["recall"], "support": lab["support"],
                         "composed_oof_macro_f1": m["macro_f1"]})
    best_emb = max(embedder_order, key=lambda e: (round(scores[e], 12), -list(embedder_order).index(e)))
    rows = [{**r, "selected_embedder": r["embedder"] == best_emb} for r in rows]
    return Selection(data.level, data.classes, best_emb, picks[best_emb], scores, tuple(rows), excluded)


# --------------------------------------------------------------------------- composed categoriser
class Categoriser:
    """Composed per-label categoriser of one level: column j comes from the candidate selected for class j.
    ``predict``/``predict_proba`` return (n, len(classes)) arrays in the fixed class order."""

    def __init__(self, level: str, embedder: str, classes: Sequence[str], models: Mapping[str, ColumnModel],
                 label_candidates: Sequence[str], info: Optional[Mapping] = None):
        if len(label_candidates) != len(classes):
            raise CategorisationError("one candidate per class is required")
        self.level, self.embedder = level, embedder
        self.classes = tuple(classes)
        self.models = dict(models)
        self.label_candidates = tuple(label_candidates)
        self.info = dict(info or {})

    def _compose(self, X, attr, dtype):
        out = np.zeros((len(X), len(self.classes)), dtype=dtype)
        for name, m in self.models.items():
            cols = [j for j, c in enumerate(self.label_candidates) if c == name]
            if cols:
                out[:, cols] = getattr(m, attr)(X)[:, cols]
        return out

    def predict(self, X) -> np.ndarray:
        return self._compose(X, "predict", np.int64)

    def predict_proba(self, X) -> np.ndarray:
        return self._compose(X, "predict_proba", np.float64)


def refit(data: LevelData, selection: Selection, cfg: Mapping, features: Callable[[str, Sequence[str]], np.ndarray],
          forbidden_ids, *, seed: int) -> Categoriser:
    """Refit each selected candidate once on ALL eligible train rows (rare-labelset filter recomputed on them)."""
    if selection.level != data.level or tuple(selection.classes) != tuple(data.classes):
        raise CategorisationError("selection does not belong to this level")
    assert_no_test(data.train_ids, data.test_ids, "refit")
    X = np.asarray(features(selection.embedder, data.train_ids))
    _check_bound(X, data.train_ids, forbidden_ids, "refit")
    min_count = int(cfg["selection"]["min_labelset_count"])
    fit_idx = frequent_rows(data.labelsets_train, np.arange(len(data.train_ids)), min_count)
    if len(fit_idx) == 0:
        raise CategorisationError(f"{data.level}: no labelset with >= {min_count} train rows")
    by_name = {c.name: c for c in build_candidates(cfg)}
    models = {}
    for name in dict.fromkeys(selection.label_candidates):
        m = fit_candidate(cfg, by_name[name], X, data.Y_train, fit_idx, seed)
        if isinstance(m, str):
            raise CategorisationError(f"{name} infeasible at refit: {m}")
        models[name] = m
    info = {"n_train": len(data.train_ids), "n_fit": int(len(fit_idx)), "fit_ids_sha256": hashlib.sha256(
        "\n".join(data.train_ids[i] for i in fit_idx).encode()).hexdigest()}
    return Categoriser(data.level, selection.embedder, data.classes, models, selection.label_candidates, info)


def _population(data: LevelData, cfg: Mapping) -> str:
    return validate_level_cfg(cfg["levels"][data.level])


def relevance_block(data: LevelData, P, S, sel, *, B: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    """Implicit relevance of an All-population (One-Step) top level on the test rows ``sel``: predicted relevant =
    any non-'other' class predicted (empty prediction = not relevant; 'other' never vetoes a real label); score = max
    non-'other' class score. It is an implicit relevance measure, not equivalent to the binary filter: relevant-unmapped
    posts have gold 'other'. ``empty_predictions`` and ``other_with_real`` count the raw predictions with no class and
    with 'other' plus at least one real label (no exclusivity rule).
    Binary metrics vs gold relevance (``filtering.metrics``; ROC AUC NaN when gold has one class), confusion
    counts [[tn, fp], [fn, tp]], prevalence, group-bootstrap CIs (``filtering.bootstrap_ci``), plus the
    relevance recall on the relevant-unmapped rows (gold relevant, labelset ('other',)) and the mapped-relevant rows."""
    if data.relevant_test is None:
        raise CategorisationError("gold relevance of the test rows is required for the relevance block")
    cls = list(data.classes)
    if "other" not in cls:
        raise CategorisationError("relevance block needs an 'other' class")
    keep = [j for j, c in enumerate(cls) if c != "other"]
    P, S = np.asarray(P), np.asarray(S)
    gold = np.asarray(data.relevant_test, dtype=bool)[sel].astype(int)
    pred = (P[:, keep] > 0).any(axis=1).astype(int) if keep else np.zeros(len(P), dtype=int)
    score = S[:, keep].max(axis=1) if keep else np.zeros(len(P))
    oi = cls.index("other")
    n_empty = int((~(P > 0).any(axis=1)).sum())
    n_other_real = int(((P[:, oi] > 0) & pred.astype(bool)).sum())
    groups = data.groups_test[sel]
    ls = [l for l, k in zip(data.labelsets_test, sel) if k]
    unmapped = np.array([l == ("other",) for l in ls], dtype=bool) & (gold == 1)
    mapped = (gold == 1) & ~unmapped
    tn = int(((gold == 0) & (pred == 0)).sum()); fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum()); tp = int(((gold == 1) & (pred == 1)).sum())
    two = len(np.unique(gold)) == 2
    if two:
        m = _fl.metrics(gold, pred, score)
    else:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
        m = {"balanced_accuracy": float(balanced_accuracy_score(gold, pred)),
             "accuracy": float(accuracy_score(gold, pred)),
             "precision": float(precision_score(gold, pred, zero_division=0)),
             "recall": float(recall_score(gold, pred, zero_division=0)),
             "f1": float(f1_score(gold, pred, zero_division=0)), "roc_auc": float("nan")}
    out = {**m, "n": int(len(gold)), "prevalence": float(gold.mean()) if len(gold) else float("nan"),
           "confusion_matrix": [[tn, fp], [fn, tp]], "predicted_relevant": int(pred.sum()),
           "empty_predictions": n_empty, "other_with_real": n_other_real}
    if B > 0 and two:
        out["bootstrap"] = _fl.bootstrap_ci(gold, pred, score, groups, B=B, seed=seed, level=level)
    def recall(mask):
        return float(pred[mask].mean()) if mask.any() else float("nan")
    out["relevant_unmapped"] = {"n": int(unmapped.sum()), "recall": recall(unmapped)}
    out["relevant_mapped"] = {"n": int(mapped.sum()), "recall": recall(mapped)}
    return out


def evaluate_test(categoriser: Categoriser, data: LevelData, cfg: Mapping,
                  features: Callable[[str, Sequence[str]], np.ndarray], *, B: int, seed: int,
                  level: float = 0.95) -> Dict[str, Any]:
    """Single test evaluation: all eligible test rows (primary) and the thesis-comparable subset (rows whose
    labelset is frequent in the eligible train rows), each with group-bootstrap CIs."""
    if categoriser.level != data.level or categoriser.classes != tuple(data.classes):
        raise CategorisationError("categoriser does not belong to this level")
    X = np.asarray(features(categoriser.embedder, data.test_ids))
    if len(X) != len(data.test_ids):
        raise CategorisationError("test features and test ids differ in length")
    P, S = categoriser.predict(X), categoriser.predict_proba(X)
    freq = _sp.frequent_labelsets(data.labelsets_train, int(cfg["selection"]["min_labelset_count"]))
    mask = np.array([ls in freq for ls in data.labelsets_test], dtype=bool)
    out = {}
    for name, sel in (("all", np.ones(len(X), dtype=bool)), ("thesis_comparable", mask)):
        if not sel.any():
            out[name] = {"n": 0}
            continue
        m = multilabel_metrics(data.Y_test[sel], P[sel], S[sel], data.classes)
        m.pop("_label_f1")
        if B > 0:
            m["bootstrap"] = group_bootstrap(data.Y_test[sel], P[sel], S[sel], data.groups_test[sel], data.classes,
                                             B=B, seed=seed, level=level)
        if _population(data, cfg) == "all":
            m["relevance"] = relevance_block(data, P[sel], S[sel], sel, B=B, seed=seed, level=level)
        out[name] = m
    out["primary"] = "all"
    return out


# --------------------------------------------------------------------------- publishing
def staged(path) -> str:
    return str(path) + ".tmp"


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def refuse_production_output(path, production_path) -> None:
    """Smoke runs must never write a production file (same resolved path or same name)."""
    if production_path and (Path(path).resolve() == Path(production_path).resolve()
                            or Path(path).name == Path(production_path).name):
        raise CategorisationError(f"smoke run refuses to write the production output {production_path}")


def stage_categoriser(categoriser: Categoriser, pickle_path) -> str:
    """joblib-dump to ``staged(pickle_path)``; returns its sha256 (put it in the sidecar)."""
    import joblib
    joblib.dump(categoriser, staged(pickle_path))
    return _sha256(staged(pickle_path))


def publish_categoriser(pickle_path, sidecar: Mapping, extra_paths: Sequence = (), *, smoke: bool = False,
                        production_paths: Sequence = ()) -> Dict[str, Any]:
    """Rename the staged pickle and extras (``staged(path)``) to their final names and write the sidecar
    LAST with ``status`` complete. The sidecar must carry ``level`` and the staged pickle's ``pickle_sha256``.
    A smoke run refuses any production output name and is marked non-production."""
    finals = [pickle_path, *extra_paths]
    if smoke:
        for f in finals:
            for prod in production_paths:
                refuse_production_output(f, prod)
    for f in finals:
        if not os.path.exists(staged(f)):
            raise CategorisationError(f"staged file missing: {staged(f)}")
    if not sidecar.get("level"):
        raise CategorisationError("sidecar must name the level")
    if sidecar.get("pickle_sha256") != _sha256(staged(pickle_path)):
        raise CategorisationError("sidecar pickle_sha256 does not match the staged pickle")
    doc = {**sidecar, "status": STATUS_COMPLETE, "production": not smoke}
    side = str(pickle_path) + ".json"
    Path(staged(side)).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    for f in finals:
        os.replace(staged(f), f)
    os.replace(staged(side), side)
    return doc


def load_deployed_categoriser(path, level: str, *, allow_non_production: bool = False):
    """Load a published categoriser through its sidecar; fails closed on a missing sidecar, a status other
    than complete, a sha256 mismatch, another level or (unless allowed) a non-production artifact.
    Returns ``(categoriser, metadata)``."""
    import joblib
    side = Path(str(path) + ".json")
    if not side.exists():
        raise CategorisationError(f"no sidecar {side}; refusing to load an unvalidated categoriser")
    meta = json.loads(side.read_text())
    if meta.get("status") != STATUS_COMPLETE:
        raise CategorisationError(f"sidecar status is {meta.get('status')!r}, not complete; refusing {path}")
    if meta.get("level") != level:
        raise CategorisationError(f"sidecar level {meta.get('level')!r} != requested {level!r}")
    if not meta.get("production", False) and not allow_non_production:
        raise CategorisationError(f"{path} is a NON-PRODUCTION (smoke) categoriser")
    if meta.get("pickle_sha256") != _sha256(path):
        raise CategorisationError(f"pickle sha256 does not match its sidecar; refusing to load {path}")
    obj = joblib.load(path)
    if not isinstance(obj, Categoriser) or obj.level != level:
        raise CategorisationError(f"{path} does not hold a categoriser of level {level!r}")
    return obj, meta
