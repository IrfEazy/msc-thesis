"""Locked evaluation of the composed systems on the common test set (pure: no fitting, no I/O).

``evaluate_systems`` takes ``system.SystemResult`` objects (identical ``row_ids`` and ``labels``), the
public gold matrix (``system.gold_matrix``, ``public_labels`` order), gold relevance and the manifest
groups, and returns a JSON-serialisable dict. Point estimates use the original test rows; every CI is a
percentile interval over ``B`` group-bootstrap replicates. A replicate draws the groups uniformly with
replacement (multinomial group counts) and keeps every row of a drawn group with its multiplicity (integer
row weights, rows are never deduplicated). All systems, the paired differences and the oracle blocks use the
SAME replicate draws; the oracle populations are subsets of the test rows and take their weights from the
group draw by group name.

Metrics are computed vectorised from weighted counts (``W @ matrix``) and a weighted rank-based ROC AUC, so
B in the thousands on ~4300 rows x ~20 labels stays within minutes on CPU. AUC is computed from scores only;
a class or population with a single gold class is undefined (None / NaN, never 0 or 0.5). Per-class AUC CIs
skip undefined replicates (``n_valid``). Macro AUC averages a FIXED class set (classes with both gold classes
at positive weight on the original population); a replicate where any of them is undefined is invalid for
macro AUC. Paired differences are valid only where both systems' metrics are defined (zero-division F1 is 0,
so it never invalidates a replicate). The weighted AUC is the weighted pair statistic with ties counted 1/2,
i.e. exactly the AUC of the literally repeated rows. Gold and every system's public matrix must satisfy the
public-label invariants (``check_public``). AUC blocks carry ``population`` and weighted class supports.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from notebooks import taxonomy as _tx

_CHUNK = 250


class SystemEvalError(Exception):
    pass


# --------------------------------------------------------------------------- weights and AUC
def group_draws(n_groups: int, B: int, seed: int) -> np.ndarray:
    """(B, n_groups) integer counts: how often each group is drawn (uniform, with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B)


def row_weights(counts: np.ndarray, group_index: np.ndarray) -> np.ndarray:
    """Per-row integer weights (R, n) from group counts (R, G) and each row's group index."""
    return counts[:, group_index].astype(float)


class _Auc:
    """Weighted ROC AUC of one score vector against binary labels, for many weight vectors at once."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        score, y = np.asarray(score, float), np.asarray(y).astype(float)
        self.n = len(score)
        if self.n:
            self.o = np.argsort(score, kind="stable")
            s = score[self.o]
            self.y = y[self.o]
            self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def __call__(self, W: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.full(W.shape[0], np.nan)
        Ws = W[:, self.o]
        pos, neg = Ws * self.y, Ws * (1.0 - self.y)
        pg = np.add.reduceat(pos, self.starts, axis=1)
        ng = np.add.reduceat(neg, self.starts, axis=1)
        below = np.cumsum(ng, axis=1) - ng
        num = (pg * (below + 0.5 * ng)).sum(axis=1)
        den = pos.sum(axis=1) * neg.sum(axis=1)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _div(a, b):
    b = np.asarray(b, float)
    return np.where(b > 0, np.asarray(a, float) / np.where(b > 0, b, 1.0), 0.0)


def _prf(tp, fp, fn):
    return _div(tp, tp + fp), _div(tp, tp + fn), _div(2 * tp, 2 * tp + fp + fn)


def _nanmean(arrs: Sequence[np.ndarray]) -> np.ndarray:
    a = np.stack(arrs, axis=0)
    ok = ~np.isnan(a)
    cnt = ok.sum(axis=0)
    return np.where(cnt > 0, np.where(ok, a, 0.0).sum(axis=0) / np.where(cnt > 0, cnt, 1), np.nan)


# --------------------------------------------------------------------------- output helpers
def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def _entry(point: float, boot: np.ndarray, level: float) -> Dict[str, Any]:
    ok = boot[~np.isnan(boot)]
    if len(ok):
        a = (1.0 - level) / 2.0 * 100.0
        lo, hi = np.percentile(ok, [a, 100.0 - a])
        ci = [_num(lo), _num(hi)]
    else:
        ci = [None, None]
    return {"value": _num(point), "ci": ci, "n_valid": int(len(ok))}


def _put(tree: dict, path: Tuple[str, ...], value: Any) -> None:
    for p in path[:-1]:
        tree = tree.setdefault(p, {})
    tree[path[-1]] = value


# --------------------------------------------------------------------------- public-label invariants
def check_public(matrix, labels: Sequence[str], taxonomy: Optional[_tx.Taxonomy] = None, what: str = "matrix") -> None:
    """Raise ``SystemEvalError`` unless ``matrix`` is a valid public-label matrix: binary, each sub label
    implies its parent, ``other`` is set iff no mapped label is, and no row is empty."""
    tx = taxonomy or _tx.default_taxonomy()
    other = tx.other
    labels = list(labels)
    M = np.asarray(matrix)
    if M.ndim != 2 or M.shape[1] != len(labels):
        raise SystemEvalError(f"{what}: shape {M.shape} does not match {len(labels)} labels")
    if not np.isin(M, (0, 1)).all():
        raise SystemEvalError(f"{what}: not binary")
    ix = {c: j for j, c in enumerate(labels)}
    if other not in ix:
        raise SystemEvalError(f"{what}: public labels lack {other!r}")
    for parent in tx.parents():
        if parent not in ix:
            raise SystemEvalError(f"{what}: public labels lack parent {parent!r}")
        for c in tx.sub_classes[parent]:
            if c == other:
                continue
            if c not in ix:
                raise SystemEvalError(f"{what}: public labels lack sub label {c!r}")
            bad = np.flatnonzero((M[:, ix[c]] == 1) & (M[:, ix[parent]] == 0))
            if len(bad):
                raise SystemEvalError(f"{what}: sub label {c!r} without parent {parent!r} in row(s) {bad[:5].tolist()}")
    mapped = np.delete(M, ix[other], axis=1).sum(axis=1) > 0
    empty = np.flatnonzero((M.sum(axis=1) == 0))
    if len(empty):
        raise SystemEvalError(f"{what}: empty row(s) {empty[:5].tolist()}")
    bad = np.flatnonzero(mapped == (M[:, ix[other]] == 1))
    if len(bad):
        raise SystemEvalError(f"{what}: `{other}` must be set iff no mapped label is (row(s) {bad[:5].tolist()})")


# --------------------------------------------------------------------------- evaluation
def _check(results, gold, relevant, groups):
    if not results:
        raise SystemEvalError("no results given")
    names = list(results)
    first = results[names[0]]
    for nme in names[1:]:
        r = results[nme]
        if tuple(r.row_ids) != tuple(first.row_ids):
            raise SystemEvalError(f"{nme}: row_ids differ from {names[0]}")
        if tuple(r.labels) != tuple(first.labels):
            raise SystemEvalError(f"{nme}: labels differ from {names[0]}")
    n, L = len(first.row_ids), len(first.labels)
    gold = np.asarray(gold)
    if gold.shape != (n, L):
        raise SystemEvalError(f"gold shape {gold.shape} != {(n, L)}")
    if not np.isin(gold, (0, 1)).all():
        raise SystemEvalError("gold must be binary")
    relevant = np.asarray(relevant).astype(bool)
    groups = np.asarray(groups)
    if relevant.shape != (n,) or groups.shape != (n,):
        raise SystemEvalError("relevant and groups must have one entry per row")
    return gold.astype(np.int64), relevant, groups


def _sub_columns(result, other: str) -> Dict[str, List[int]]:
    idx = {c: j for j, c in enumerate(result.labels)}
    return {p: [idx[c] for c in cls if c != other] for p, cls in result.sub_classes.items()}


def evaluate_systems(results: Mapping[str, Any], gold, relevant, groups, *, oracle: Optional[Mapping[str, Tuple]] = None,
                     B: int = 1000, seed: int = 0, level: float = 0.95,
                     compare: Optional[Sequence[Tuple[str, str]]] = None,
                     taxonomy: Optional[_tx.Taxonomy] = None) -> Dict[str, Any]:
    tx = taxonomy or _tx.default_taxonomy()
    other = tx.other
    gold, relevant, groups = _check(results, gold, relevant, groups)
    names = list(results)
    if {"paired", "oracle"} & set(names):
        raise SystemEvalError("system names 'paired' and 'oracle' are reserved")
    check_public(gold, results[names[0]].labels, tx, "gold")
    for nme, r in results.items():
        check_public(r.P, r.labels, tx, f"system {nme!r} P")
    if compare is None:
        compare = [(names[0], names[1])] if len(names) > 1 else []
    for a, b in compare:
        if a not in results or b not in results:
            raise SystemEvalError(f"compare pair {(a, b)} names an unknown system")
    labels = list(results[names[0]].labels)
    L, n = len(labels), len(groups)
    if other not in labels:
        raise SystemEvalError(f"public labels lack {other!r}")
    oi = labels.index(other)
    lab_idx = {c: j for j, c in enumerate(labels)}
    ug, gidx = np.unique(groups, return_inverse=True)
    G = len(ug)
    gpos = {g: i for i, g in enumerate(ug.tolist())}
    rel_f = relevant.astype(float)
    mapped_gold = gold[:, oi] == 0
    if (~relevant & mapped_gold).any():
        raise SystemEvalError("gold has a mapped label on a row with relevant=False")

    # --- fixed (replicate-independent) structure: AUC preparations ---------------------------
    auc_jobs: List[Tuple[Tuple[str, ...], _Auc, np.ndarray, np.ndarray]] = []   # (path, prep, row idx, y)
    macro_specs: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = []
    populations: Dict[Tuple[str, ...], str] = {}     # node path -> descriptive population name
    allrows = np.arange(n)

    def add_auc(path, idx, score, y, pop):
        auc_jobs.append((path, _Auc(score, y), idx, np.asarray(y, float)))
        populations[path] = pop

    for nme, r in results.items():
        base = (nme,)
        fs = np.asarray(r.filter_score, float)
        if n and not np.isnan(fs).all():
            add_auc(base + ("stage_auc", "filter"), allrows, fs, rel_f, f"{nme} filter: all rows")
        ran = ~np.isnan(np.asarray(r.top_scores, float)).all(axis=1) if r.top_scores.shape[1] else np.zeros(n, bool)
        idx = np.flatnonzero(ran)
        kids = []
        has_filter = bool(n) and not np.isnan(fs).all()
        top_pop = f"{nme} top: " + ("filter-accepted rows" if has_filter else "rows where the top stage ran")
        for j, c in enumerate(r.top_classes):
            p = base + ("stage_auc", "top", "per_class", c)
            add_auc(p, idx, r.top_scores[idx, j], gold[idx, lab_idx[c]], top_pop)
            kids.append(p)
        macro_specs.append((base + ("stage_auc", "top", "macro"), kids))
        populations[base + ("stage_auc", "top", "macro")] = top_pop
        sub_kids = []
        for parent, cls in r.sub_classes.items():
            idx = np.flatnonzero(r.sub_applied[parent])
            sub_pop = f"{nme} sub {parent}: rows with predicted parent"
            kids = []
            for j, c in enumerate(cls):
                if c == other:
                    continue
                p = base + ("stage_auc", "sub", parent, "per_class", c)
                add_auc(p, idx, r.sub_scores[parent][idx, j], gold[idx, lab_idx[c]], sub_pop)
                kids.append(p)
            macro_specs.append((base + ("stage_auc", "sub", parent, "macro"), kids))
            populations[base + ("stage_auc", "sub", parent, "macro")] = sub_pop
            sub_kids += kids
        macro_specs.append((base + ("stage_auc", "sub", "macro_all"), sub_kids))
        populations[base + ("stage_auc", "sub", "macro_all")] = f"{nme} sub: rows with predicted parent (per parent)"

    # --- oracle populations ----------------------------------------------------------------
    orc: Dict[str, Dict[str, Any]] = {}
    for stage, (pred, score, ogold, ogroups, classes) in (oracle or {}).items():
        pred, ogold = np.asarray(pred).astype(np.int64), np.asarray(ogold).astype(np.int64)
        score = np.asarray(score, float)
        m = len(ogroups)
        if not (np.isin(pred, (0, 1)).all() and np.isin(ogold, (0, 1)).all()):
            raise SystemEvalError(f"oracle {stage}: pred and gold must be binary 0/1")
        if not np.isfinite(score).all():
            raise SystemEvalError(f"oracle {stage}: scores must be finite")
        if not (pred.shape == ogold.shape == score.shape == (m, len(classes))):
            raise SystemEvalError(f"oracle {stage}: inconsistent shapes")
        try:
            og = np.array([gpos[g] for g in np.asarray(ogroups).tolist()], dtype=int)
        except KeyError as e:
            raise SystemEvalError(f"oracle {stage}: group {e} is not a test group") from None
        orc[stage] = {"pred": pred, "gold": ogold, "classes": list(classes), "g": og,
                      "auc": [_Auc(score[:, j], ogold[:, j]) for j in range(len(classes))]}
        opop = f"oracle {stage}: " + ("gold-parent rows" if stage.startswith("sub") else "gold-relevant rows")
        populations[("oracle", stage)] = opop
        for c in classes:
            populations[("oracle", stage, "per_class", c, "auc")] = opop
        populations[("oracle", stage, "macro_auc")] = opop
        macro_specs.append((("oracle", stage, "macro_auc"),
                            [("oracle", stage, "per_class", c, "auc") for c in classes]))

    # --- vectorised metric computation for a block of group-count rows ------------------------
    def compute(C: np.ndarray, fixed: Optional[Dict[Tuple[str, ...], List[Tuple[str, ...]]]] = None
                ) -> Dict[Tuple[str, ...], np.ndarray]:
        W = row_weights(C, gidx)
        out: Dict[Tuple[str, ...], np.ndarray] = {}
        nw = W.sum(axis=1)
        for nme, r in results.items():
            base = (nme,)
            P = np.asarray(r.P).astype(float)
            Y = gold.astype(float)
            tp, fp, fn = W @ (Y * P), W @ ((1 - Y) * P), W @ (Y * (1 - P))
            pl_p, pl_r, pl_f = _prf(tp, fp, fn)
            sup = W @ Y
            for j, c in enumerate(labels):
                for k, v in (("precision", pl_p), ("recall", pl_r), ("f1", pl_f), ("support_rows", sup)):
                    out[base + ("per_label", c, k)] = v[:, j]

            def micro_macro(cols):
                a, b, c_ = tp[:, cols].sum(1), fp[:, cols].sum(1), fn[:, cols].sum(1)
                p, rr, f = _prf(a, b, c_)
                return p, rr, f, (pl_f[:, cols].mean(axis=1) if len(cols) else np.zeros(len(a)))

            p, rr, f, mf = micro_macro(list(range(L)))
            e2e = base + ("end_to_end",)
            out[e2e + ("micro_precision",)], out[e2e + ("micro_recall",)], out[e2e + ("micro_f1",)] = p, rr, f
            out[e2e + ("macro_f1",)] = mf
            out[e2e + ("subset_accuracy",)] = _div(W @ (gold == np.asarray(r.P)).all(axis=1).astype(float), nw)
            out[e2e + ("hamming_loss",)] = _div(W @ (gold != np.asarray(r.P)).sum(axis=1).astype(float), nw * L)
            top_cols = [lab_idx[c] for c in r.top_classes if c != other] + [oi]
            sub_cols = sorted({j for v in _sub_columns(r, other).values() for j in v})
            proj = {"top_only": top_cols, "sub_only": sub_cols, "mapped_only": [j for j in range(L) if j != oi]}
            for k, cols in proj.items():
                if cols:
                    _, _, f, mf = micro_macro(cols)
                else:  # empty column set: undefined, not 0
                    f = mf = np.full(len(nw), np.nan)
                out[base + ("projections", k, "micro_f1")], out[base + ("projections", k, "macro_f1")] = f, mf

            # implicit relevance (any mapped public label predicted) vs gold relevance
            pr = np.asarray(r.P)[:, oi] == 0
            rt, rp = relevant, pr
            a = W @ (rt & rp).astype(float)
            b = W @ (~rt & rp).astype(float)
            c_ = W @ (rt & ~rp).astype(float)
            d = W @ (~rt & ~rp).astype(float)
            rec = _div(a, a + c_)
            spec_ok = (b + d) > 0
            bal = np.where((a + c_) > 0, 0.5 * (rec + np.where(spec_ok, d / np.where(spec_ok, b + d, 1.0), np.nan)), np.nan)
            ip, ir, iff = _prf(a, b, c_)
            ib = base + ("implicit_relevance",)
            out[ib + ("balanced_accuracy",)], out[ib + ("precision",)] = bal, ip
            out[ib + ("recall",)], out[ib + ("f1",)] = ir, iff

            # filter diagnostics
            fsc = np.asarray(r.filter_score, float)
            if n and not np.isnan(fsc).all():
                rej = np.asarray(r.filter_reject).astype(bool)
                tpf = W @ (rt & ~rej).astype(float)
                fnf = W @ (rt & rej).astype(float)
                fb = base + ("filter",)
                out[fb + ("recall",)] = _div(tpf, tpf + fnf)
                out[fb + ("false_negative_rate",)] = _div(fnf, tpf + fnf)
                out[fb + ("dropped_relevant_mapped",)] = W @ (rt & rej & mapped_gold).astype(float)
                out[fb + ("dropped_relevant_unmapped",)] = W @ (rt & rej & ~mapped_gold).astype(float)
        for path, prep, idx, y in auc_jobs:
            out[path] = prep(W[:, idx])
            Wi = W[:, idx]
            out[("_sup",) + path + ("positive_support",)] = Wi @ y
            out[("_sup",) + path + ("negative_support",)] = Wi @ (1.0 - y)
        for a_, b_ in compare:
            for k in ("micro_f1", "macro_f1", "subset_accuracy"):
                x, z = out[(a_, "end_to_end", k)], out[(b_, "end_to_end", k)]
                # joint validity: a replicate counts only if both systems' metric is defined.
                # The current paired metrics (F1 with zero-division -> 0, subset accuracy) are always
                # defined, so this guard is defensive and never fires today.
                out[("paired", f"{a_}_minus_{b_}", k)] = np.where(np.isnan(x) | np.isnan(z), np.nan, x - z)
        for stage, o in orc.items():
            Wo = C[:, o["g"]].astype(float)
            Y, P = o["gold"].astype(float), o["pred"].astype(float)
            tp, fp, fn = Wo @ (Y * P), Wo @ ((1 - Y) * P), Wo @ (Y * (1 - P))
            pp, rr, ff = _prf(tp, fp, fn)
            ob = ("oracle", stage)
            empty = Wo.sum(axis=1) == 0          # no oracle row drawn: the population is empty -> undefined
            out[("_empty", stage)] = empty.astype(float)
            out[ob + ("micro_f1",)] = np.where(empty, np.nan, _prf(tp.sum(1), fp.sum(1), fn.sum(1))[2])
            out[ob + ("macro_f1",)] = np.where(empty, np.nan, ff.mean(axis=1))
            for j, c in enumerate(o["classes"]):
                out[ob + ("per_class", c, "f1")] = np.where(empty, np.nan, ff[:, j])
                p = ob + ("per_class", c, "auc")
                out[p] = o["auc"][j](Wo)
                out[("_sup",) + p + ("positive_support",)] = Wo @ Y[:, j]
                out[("_sup",) + p + ("negative_support",)] = Wo @ (1.0 - Y[:, j])
        for path, kids in macro_specs:
            if fixed is None:                    # point estimate: average the defined classes
                out[path] = _nanmean([out[k] for k in kids]) if kids else np.full(W.shape[0], np.nan)
            else:                                # replicates: the fixed set, invalid if any member is undefined
                ks = fixed[path]
                out[path] = np.stack([out[k] for k in ks]).mean(axis=0) if ks else np.full(W.shape[0], np.nan)
        return out

    point = compute(np.ones((1, G)))
    fixed = {path: [k for k in kids if not np.isnan(point[k][0])] for path, kids in macro_specs}
    parts: List[Dict[Tuple[str, ...], np.ndarray]] = []
    if B > 0:
        counts = group_draws(G, B, seed)
        for s in range(0, B, _CHUNK):
            parts.append(compute(counts[s:s + _CHUNK], fixed))
    boot = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)) for k in point}

    # --- assemble ------------------------------------------------------------------------------
    res: Dict[str, Any] = {"n_rows": int(n), "n_groups": int(G), "B": int(B), "seed": int(seed),
                           "level": float(level), "labels": labels,
                           "compare": [list(c) for c in compare], "systems": {}, "paired": {}, "oracle": {}}
    sup: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for k in point:
        e = _entry(point[k][0], boot[k], level)
        if k[0] == "_sup":
            sup.setdefault(k[1:-1], {})[k[-1]] = e
        elif k[0] == "_empty":
            continue
        elif k[0] == "paired":
            _put(res["paired"], k[1:], e)
        elif k[0] == "oracle":
            _put(res["oracle"], k[1:], e)
        else:
            _put(res["systems"], k, e)
    def node_at(path):
        node = res["oracle"] if path[0] == "oracle" else res["systems"]
        for p in (path[1:] if path[0] == "oracle" else path):
            node = node[p]
        return node

    for path, d in sup.items():
        node_at(path).update(d)
    for path, pop in populations.items():
        node_at(path)["population"] = pop
    # point-only additions
    for nme, r in results.items():
        s = res["systems"][nme]
        for j, c in enumerate(labels):
            s["per_label"][c]["support_groups"] = int(len(np.unique(gidx[gold[:, j] == 1])))
        rc = r.routing_counts()
        routing: Dict[str, Any] = {}
        for k, v in rc.items():
            if isinstance(v, dict):
                routing[k] = {p: {"count": int(c), "rate": float(c / n) if n else None} for p, c in v.items()}
            else:
                routing[k] = {"count": int(v), "rate": float(v / n) if n else None}
        routing["sub_applied"] = {p: {"count": int(np.asarray(v).sum()),
                                      "rate": float(np.asarray(v).sum() / n) if n else None}
                                  for p, v in r.sub_applied.items()}
        s["routing"] = routing
        fsc = np.asarray(r.filter_score, float)
        if n and not np.isnan(fsc).all():
            rej = np.asarray(r.filter_reject).astype(bool)
            f = s["filter"]
            f["confusion"] = [[int((~relevant & rej).sum()), int((~relevant & ~rej).sum())],
                              [int((relevant & rej).sum()), int((relevant & ~rej).sum())]]
            f["auc"] = s["stage_auc"]["filter"]
        for path, kids in macro_specs:
            if path[0] == nme and path[0] != "oracle":
                node = s
                for p in path[1:]:
                    node = node[p]
                node["n_undefined"] = int(sum(np.isnan(point[k][0]) for k in kids))
    for stage, o in orc.items():
        node = res["oracle"][stage]
        node["n"] = int(len(o["g"]))
        node["population"] = populations[("oracle", stage)]
        node["n_empty_replicates"] = int(boot[("_empty", stage)].sum()) if B > 0 else 0
        node["macro_auc"]["n_undefined"] = int(sum(np.isnan(point[("oracle", stage, "per_class", c, "auc")][0])
                                                   for c in o["classes"]))
    return res


# --------------------------------------------------------------------------- oracle inputs
def oracle_inputs(system, row_ids: Sequence[str], features, watson_lists: Iterable[Iterable[str]],
                  relevant: Iterable[Any], groups: Sequence, taxonomy: Optional[_tx.Taxonomy] = None
                  ) -> Dict[str, Tuple]:
    """Run each component of ``system`` directly on its gold population (no fitting).

    Top model: gold-relevant rows, gold = ``top_labels(.., two_step)`` over the top classes. Sub model of
    ``parent``: rows whose gold top contains the parent, gold = ``sub_labels`` over the sub classes with
    sub ``other`` kept as a class (the component's own label space). Returns
    ``{"top": (pred, score, gold, groups, classes), "sub:<parent>": ...}``.
    """
    tx = taxonomy or getattr(system, "taxonomy", None) or _tx.default_taxonomy()
    ids = [str(r) for r in row_ids]
    wls = [list(w) for w in watson_lists]
    rel = [_tx.as_bool(x) for x in relevant]
    groups = list(groups)
    if not (len(ids) == len(wls) == len(rel) == len(groups)):
        raise SystemEvalError("oracle_inputs: row_ids, watson_lists, relevant and groups differ in length")

    def onehot(lists, classes, stage):
        ix = {c: j for j, c in enumerate(classes)}
        out = np.zeros((len(lists), len(classes)), dtype=np.int64)
        for i, labs in enumerate(lists):
            for x in labs:
                if x not in ix:
                    raise SystemEvalError(f"oracle_inputs: gold label {x!r} is not among the classes of stage {stage!r}")
                out[i, ix[x]] = 1
        return out

    out: Dict[str, Tuple] = {}
    sel = [i for i in range(len(ids)) if rel[i]]
    tops = {i: list(tx.top_labels(wls[i], True, "two_step")) for i in sel}
    if sel:
        X = np.asarray(features(system.top.embedder, [ids[i] for i in sel]))
        out["top"] = (system.top.predict(X), system.top.predict_proba(X),
                      onehot([tops[i] for i in sel], system.top.classes, "top"),
                      [groups[i] for i in sel], list(system.top.classes))
    for parent, ad in system.subs.items():
        psel = [i for i in sel if parent in tops[i]]
        if not psel:
            continue
        X = np.asarray(features(ad.embedder, [ids[i] for i in psel]))
        out[f"sub:{parent}"] = (ad.predict(X), ad.predict_proba(X),
                                onehot([tx.sub_labels(wls[i], parent) for i in psel], ad.classes, f"sub:{parent}"),
                                [groups[i] for i in psel], list(ad.classes))
    return out
