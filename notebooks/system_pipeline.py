"""Orchestration of the locked System evaluation (``SystemLabelAssignment.ipynb``): fail closed, no training.

``run(settings, cfg)`` verifies the pinned split and keeps the TEST rows only, loads the deployed filter and
categorisers through their sidecars, builds the Two-Step and One-Step systems (``system``), predicts the test
rows from the embedding artifacts, evaluates both with the paired group bootstrap (``system_eval``) and the
Cotov (row bootstrap, no groups) relevance blocks, and publishes ``system_performance.json`` and
``system_predictions.csv`` atomically: everything is computed first, written to ``<name>.tmp`` and renamed,
the summary last, so a failed run leaves nothing under the final names. Project values (bootstrap, pairs,
level -> env var names) live in ``system.toml``; this module is generic and pure (no nltk/torch import).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from notebooks import splits
from notebooks import system as sysm
from notebooks import system_eval as se
from notebooks import taxonomy as tx
from notebooks.textprep import TEXTPREP_VERSION

FROZEN_SEED = 42
PERFORMANCE_NAME = "system_performance.json"
PREDICTIONS_NAME = "system_predictions.csv"
SMOKE_PREFIX = "smoke_"
STATUS_COMPLETE = "complete"
STATUS_COMPLETE_COTOV_BLOCKED = "complete-cotov-blocked"
SYSTEMS = ("two_step", "one_step")


class SystemPipelineError(Exception):
    pass


def env_flag(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- config and settings
def load_config(path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def effective_config(cfg: Mapping, smoke: bool = False) -> Dict[str, Any]:
    """Config with the ``[smoke]`` bootstrap size applied when ``smoke`` (NON-PRODUCTION)."""
    out = json.loads(json.dumps(cfg))
    out["smoke"] = bool(smoke)
    if smoke:
        out["bootstrap"]["B"] = int(cfg["smoke"]["B"])
    return out


def validate_config(cfg: Mapping, taxonomy: Optional[tx.Taxonomy] = None) -> None:
    t = taxonomy or tx.default_taxonomy()
    levels = cfg["levels"]
    need = {cfg["systems"]["two_step_top"], cfg["systems"]["one_step_top"], *t.parents()}
    missing = sorted(need - set(levels))
    if missing:
        raise SystemPipelineError(f"[levels] lacks {missing}")
    for pair in cfg["bootstrap"]["compare"]:
        if len(pair) != 2 or any(p not in SYSTEMS for p in pair):
            raise SystemPipelineError(f"compare pair {pair!r} must name two of {list(SYSTEMS)}")


@dataclass(frozen=True)
class Settings:
    smoke: bool
    overwrite: bool
    allow_no_cotov: bool
    allow_unresolved: bool
    manifest_csv: Path
    train_csv: Path
    test_csv: Path
    manifest_sha256: str
    embeddings_dir: Path
    filter_pickle: Path
    categoriser_paths: Dict[str, Path]      # categoriser level -> pickle path
    cotov_csv: Optional[Path]
    out_dir: Path
    smoke_dir: Optional[Path]


def settings_from_env(env: Mapping[str, str], cfg: Mapping) -> Settings:
    """Read the run settings from ``env`` (``os.environ``); a missing required variable raises."""
    smoke = env_flag(env, "SYSTEM_SMOKE")
    names = {"SPLIT_MANIFEST_CSV", "SPLIT_TRAIN_CSV", "SPLIT_TEST_CSV", "SPLIT_MANIFEST_SHA256", "EMBEDDINGS_DIR",
             "SYSTEM_OUT_DIR", cfg["filter"]["pickle_env"], *cfg["levels"].values()}
    if smoke:
        names.add("SYSTEM_SMOKE_OUT_DIR")
    missing = sorted(n for n in names if not env.get(n))
    if missing:
        raise SystemPipelineError(f"missing environment variables: {missing}")
    cotov = env.get("COTOV_CSV")
    return Settings(
        smoke=smoke, overwrite=env_flag(env, "OVERWRITE"), allow_no_cotov=env_flag(env, "SYSTEM_ALLOW_NO_COTOV"),
        allow_unresolved=env_flag(env, "ALLOW_UNRESOLVED_EMBEDDERS"),
        manifest_csv=Path(env["SPLIT_MANIFEST_CSV"]), train_csv=Path(env["SPLIT_TRAIN_CSV"]),
        test_csv=Path(env["SPLIT_TEST_CSV"]), manifest_sha256=env["SPLIT_MANIFEST_SHA256"],
        embeddings_dir=Path(env["EMBEDDINGS_DIR"]), filter_pickle=Path(env[cfg["filter"]["pickle_env"]]),
        categoriser_paths={lvl: Path(env[var]) for lvl, var in cfg["levels"].items()},
        cotov_csv=Path(cotov) if cotov else None, out_dir=Path(env["SYSTEM_OUT_DIR"]),
        smoke_dir=Path(env["SYSTEM_SMOKE_OUT_DIR"]) if smoke else None)


# --------------------------------------------------------------------------- outputs
def refuse_production_output(path, production_path) -> None:
    """Smoke runs must never write a production file (same resolved path or same name)."""
    if production_path and (Path(path).resolve() == Path(production_path).resolve()
                            or Path(path).name == Path(production_path).name):
        raise SystemPipelineError(f"smoke run refuses to write the production output {production_path}")


def staged(path) -> str:
    return str(path) + ".tmp"


def output_paths(s: Settings) -> Tuple[Path, Path]:
    """(performance JSON, predictions CSV) final paths; smoke runs write ``smoke_*`` names under the smoke dir only."""
    prod = (s.out_dir / PERFORMANCE_NAME, s.out_dir / PREDICTIONS_NAME)
    if not s.smoke:
        return prod
    paths = (s.smoke_dir / (SMOKE_PREFIX + PERFORMANCE_NAME), s.smoke_dir / (SMOKE_PREFIX + PREDICTIONS_NAME))
    if s.smoke_dir.resolve() == s.out_dir.resolve():
        raise SystemPipelineError("SYSTEM_SMOKE_OUT_DIR must differ from SYSTEM_OUT_DIR")
    for p, q in zip(paths, prod):
        refuse_production_output(p, q)
    return paths


def check_overwrite(paths: Sequence, overwrite: bool) -> None:
    exist = [str(p) for p in paths if Path(p).exists()]
    if exist and not overwrite:
        raise FileExistsError(f"{exist} exist; set OVERWRITE=true to replace them")


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats (-> None) so the dump is strict JSON."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def publish(perf_path, pred_path, summary: Dict[str, Any], predictions: pd.DataFrame, overwrite: bool) -> Dict[str, Any]:
    """Stage both outputs as ``<name>.tmp`` (the summary carries the predictions sha256) and rename them: the
    predictions first, the summary LAST. The summary is the completion marker: a reader must require it (with
    its ``status``) and may check ``outputs.predictions_sha256``. An existing summary is removed before the
    first rename, and a failed rename removes the new predictions, so a summary never sits beside other files."""
    check_overwrite([perf_path, pred_path], overwrite)
    Path(perf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    tp, tj = staged(pred_path), staged(perf_path)
    try:
        predictions.to_csv(tp, index=False)
        doc = dict(summary)
        doc["outputs"] = {"performance": Path(perf_path).name, "predictions": Path(pred_path).name,
                          "predictions_sha256": _sha256(tp)}
        text = json.dumps(json_safe(doc), indent=1, allow_nan=False)
        json.loads(text)
        Path(tj).write_text(text)
    except BaseException:
        for f in (tp, tj):
            if os.path.exists(f):
                os.remove(f)
        raise
    try:
        if os.path.exists(perf_path):
            os.remove(perf_path)
        os.replace(tp, pred_path)
        os.replace(tj, perf_path)
    except BaseException:
        for f in (tp, tj, pred_path, perf_path):
            if os.path.exists(f):
                os.remove(f)
        raise
    return doc


# --------------------------------------------------------------------------- split
@dataclass(frozen=True)
class TestRows:
    __test__ = False
    df: pd.DataFrame
    row_ids: List[str]
    groups: List[str]
    cols: Dict[str, str]


def load_test_rows(s: Settings) -> TestRows:
    """Verify the pinned manifest against the whole corpus (seed 42, pinned sha256) and keep the TEST rows."""
    train_df, test_df = pd.read_csv(s.train_csv), pd.read_csv(s.test_csv)
    manifest = splits.read_manifest(s.manifest_csv)
    full = pd.concat([train_df, test_df], ignore_index=True)
    splits.verify_manifest(full, manifest, expected_seed=FROZEN_SEED, expected_manifest_sha256=s.manifest_sha256)
    cols = dict(manifest.attrs["metadata"]["columns"])
    ids = test_df[cols["id"]].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise SystemPipelineError("duplicate row ids in the test split")
    m = manifest.set_index(manifest["row_id"].astype(str))
    if not (m.loc[ids, "split"] == "test").all():
        raise SystemPipelineError("a row of the test file is not a test row in the manifest")
    return TestRows(test_df.reset_index(drop=True), ids, m.loc[ids, "group"].astype(str).tolist(), cols)


# --------------------------------------------------------------------------- systems
def _embedder(meta: Mapping, fallback: Optional[str]):
    emb = (meta or {}).get("embedder")
    if isinstance(emb, Mapping):
        return emb.get("name") or fallback, emb.get("fingerprint")
    return (emb if isinstance(emb, str) else fallback), None


def _record(name: str, path, adapter, meta: Mapping) -> Dict[str, Any]:
    return {"component": name, "level": getattr(adapter, "source_level", None), "path": str(path),
            "sidecar": str(path) + ".json", "pickle_sha256": meta.get("pickle_sha256"),
            "manifest_sha256": meta.get("manifest_sha256"), "identity_digest": meta.get("identity_digest"),
            "embedder": adapter.embedder, "fingerprint": adapter.fingerprint}


def load_systems(s: Settings, cfg: Mapping, taxonomy: Optional[tx.Taxonomy] = None):
    """Both systems from the deployed models (sidecar-verified; smoke allows NON-PRODUCTION artifacts). Every
    component sidecar must carry ``manifest_sha256`` equal to the pinned split manifest (a model trained on
    another split would be scored on rows it saw). The four sub categorisers are loaded once and shared.
    Returns ``(two_step, one_step, components)``."""
    t = taxonomy or tx.default_taxonomy()
    validate_config(cfg, t)
    top2, top1 = cfg["systems"]["two_step_top"], cfg["systems"]["one_step_top"]
    comps: Dict[str, Dict[str, Any]] = {}

    def hook(component, path, meta, adapter):
        if meta.get("manifest_sha256") != s.manifest_sha256:
            raise SystemPipelineError(f"{component}: sidecar manifest_sha256 {meta.get('manifest_sha256')!r} != the "
                                      f"pinned SPLIT_MANIFEST_SHA256 (trained on another split?): {path}")
        if component == "filter" and meta.get("non_production") and not s.smoke:
            raise SystemPipelineError(f"{path} is a NON-PRODUCTION filter")
        comps[component] = _record(component, path, adapter, meta)

    paths = s.categoriser_paths
    two = sysm.load_two_step(s.filter_pickle, {top2: paths[top2], **{p: paths[p] for p in t.parents()}}, taxonomy=t,
                             allow_non_production=s.smoke, hook=hook)
    one = sysm.load_one_step({top1: paths[top1]}, taxonomy=t, allow_non_production=s.smoke, hook=hook, subs=two.subs)
    return two, one, comps


def features_for(system, s: Settings, corpus: str, row_ids: Sequence[str], **extra):
    """Artifact features of ``system`` bound to ``row_ids`` (fingerprints checked; smoke allows non-production)."""
    return sysm.artifact_features(system, s.embeddings_dir, corpus, row_ids, allow_unresolved=s.allow_unresolved,
                                  allow_non_production=s.smoke, **extra)


# --------------------------------------------------------------------------- Cotov
def cotov_problem(s: Settings, embedders: Sequence[str]) -> Optional[str]:
    """Why the Cotov evaluation is impossible, or None."""
    from notebooks import embeddings as em
    if not (s.cotov_csv and Path(s.cotov_csv).exists()):
        return "Cotov dataset unavailable (COTOV_CSV, used for the Related labels)"
    for e in embedders:
        if not (Path(s.embeddings_dir) / f"{em.artifact_name('cotov', e)}.json").exists():
            return f"no cotov embedding artifact for {e} in EMBEDDINGS_DIR"
    return None


def _binary_block(y, pred, score, boot: Mapping, definition: str, score_definition: str) -> Dict[str, Any]:
    """Binary decision metrics vs ``y`` with a row bootstrap. ``score`` None -> decision metrics only (no ROC AUC)."""
    from sklearn.metrics import confusion_matrix
    from notebooks import filtering as fl
    y, pred = np.asarray(y).astype(np.int64), np.asarray(pred).astype(np.int64)
    sc = np.asarray(pred if score is None else score, float)
    out = fl.metrics(y, pred, sc)
    ci = fl.bootstrap_ci(y, pred, sc, None, B=boot["B"], seed=boot["seed"], level=boot["level"])
    if score is None:
        out.pop("roc_auc")
        ci["ci"].pop("roc_auc", None)
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    out["n"], out["prevalence"] = int(len(y)), float(y.mean())
    out["bootstrap"] = ci
    out["definition"], out["score"] = definition, score_definition
    return out


def cotov_evaluation(s: Settings, cfg: Mapping, two, one, features_factory: Callable) -> Dict[str, Any]:
    """Row-bootstrap relevance of both systems on the English Cotov rows (artifact features, texts digest checked).
    Bare filter decision (score: the filter probability, so ROC AUC); One-Step implicit relevance = a mapped
    label (score: the maximum mapped top-class score, as in categorisation's relevance block); Two-Step implicit
    relevance = filter accept AND a mapped label (no score is defined: decision metrics only)."""
    from notebooks import embeddings as em
    from notebooks import filtering as fl
    ids, texts, y = fl.cotov_rows(pd.read_csv(s.cotov_csv))
    boot = cfg["bootstrap"]
    kw = {"expected_texts_sha256": em.texts_sha256(ids, texts)}
    other = two.taxonomy.other
    out: Dict[str, Any] = {"status": "complete", "n": int(len(ids)), "prevalence": float(y.mean()),
                           "unit": "row (no groups)", "systems": {}}
    for name, system in (("two_step", two), ("one_step", one)):
        res = system.predict(ids, features_factory(system, s, "cotov", ids, **kw))
        mapped = np.asarray(res.P)[:, list(res.labels).index(other)] == 0
        node: Dict[str, Any] = {}
        if name == "two_step":
            accept = ~np.asarray(res.filter_reject).astype(bool)
            node["filter_decision"] = _binary_block(
                y, accept, res.filter_score, boot, "filter accepts the row", "filter probability of the relevant class")
            node["implicit_relevance"] = _binary_block(
                y, accept & mapped, None, boot, "filter accepts AND at least one mapped public label",
                "none (decision metrics only)")
        else:
            cols = [j for j, c in enumerate(res.top_classes) if c != other]
            score = np.asarray(res.top_scores)[:, cols].max(axis=1)
            node["implicit_relevance"] = _binary_block(
                y, mapped, score, boot, "at least one mapped public label (no filter)",
                "maximum mapped top-class score")
        out["systems"][name] = node
    return out


# --------------------------------------------------------------------------- predictions table
def predictions_frame(rows: TestRows, gold, relevant, results: Mapping[str, Any]) -> pd.DataFrame:
    labels = list(next(iter(results.values())).labels)
    if any("|" in c for c in labels):
        raise SystemPipelineError("a public label contains '|'")

    def join(mat):
        return ["|".join(c for c, v in zip(labels, row) if v) for row in np.asarray(mat)]

    df = pd.DataFrame({"row_id": rows.row_ids, "group": rows.groups, "relevant": np.asarray(relevant).astype(int),
                       "gold": join(gold)})
    for name, r in results.items():
        df[name] = join(r.P)
        for flag in ("filter_reject", "top_other", "empty_top_fallback", "other_conflict"):
            df[f"{name}_{flag}"] = np.asarray(getattr(r, flag)).astype(int)
        for flag in ("sub_none_conflict", "parent_without_child"):
            parents = list(getattr(r, flag))
            df[f"{name}_{flag}"] = ["|".join(p for p in parents if getattr(r, flag)[p][i]) for i in range(len(rows.row_ids))]
    return df


def library_versions() -> Dict[str, str]:
    out = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib", "torch", "sentence-transformers",
                "transformers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


# --------------------------------------------------------------------------- run
def run(s: Settings, cfg: Mapping, *, features_factory: Optional[Callable] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The whole locked evaluation; returns the published summary. ``features_factory(system, settings, corpus,
    row_ids, **kw)`` defaults to the embedding artifacts (tests may inject a stub provider)."""
    from notebooks import filtering as fl
    ff = features_factory or features_for
    t = tx.default_taxonomy()
    perf_path, pred_path = output_paths(s)
    check_overwrite([perf_path, pred_path], s.overwrite)
    validate_config(cfg, t)
    boot = cfg["bootstrap"]
    log(f"{'NON-PRODUCTION SMOKE' if s.smoke else 'PRODUCTION'}: B={boot['B']} seed={boot['seed']}")

    rows = load_test_rows(s)
    cols = rows.cols
    relevant = np.array([tx.as_bool(v) for v in rows.df[cols["relevant"]]], dtype=bool)
    gold = sysm.gold_matrix_frame(rows.df, cols["watson"], cols["relevant"], t)
    wls = [tx.watson_list(w) for w in rows.df[cols["watson"]]]
    log(f"manifest verified ({s.manifest_sha256[:12]}...): {len(rows.row_ids)} test rows, {len(set(rows.groups))} groups")

    two, one, comps = load_systems(s, cfg, t)
    problem = cotov_problem(s, sorted(set(two.embedders) | set(one.embedders)))
    blocked = fl.require_cotov(s.smoke, s.allow_no_cotov, problem)   # production fails closed here, before any work
    if blocked:
        log(f"BLOCKED: {blocked}")

    ids = rows.row_ids
    f2, f1 = ff(two, s, "corpus", ids), ff(one, s, "corpus", ids)
    results = {"two_step": two.predict(ids, f2), "one_step": one.predict(ids, f1)}
    o2 = se.oracle_inputs(two, ids, f2, wls, relevant, rows.groups, t)
    o1 = se.oracle_inputs(one, ids, f1, wls, relevant, rows.groups, t)
    oracle = {f"{k}:{n}" if k == "top" else k: v for n, o in (("two_step", o2), ("one_step", o1))
              for k, v in o.items() if k == "top" or n == "two_step"}   # the sub models are shared
    compare = [tuple(p) for p in boot["compare"]]
    evaluation = se.evaluate_systems(results, gold, relevant, rows.groups, oracle=oracle, B=int(boot["B"]),
                                     seed=int(boot["seed"]), level=float(boot["level"]), compare=compare, taxonomy=t)
    cotov = ({"status": "blocked", "reason": blocked} if blocked
             else cotov_evaluation(s, cfg, two, one, ff))
    predictions = predictions_frame(rows, gold, relevant, results)

    summary = {
        "status": STATUS_COMPLETE_COTOV_BLOCKED if blocked else STATUS_COMPLETE,
        "production": not s.smoke, "non_production": s.smoke,
        "config": {"B": int(boot["B"]), "seed": int(boot["seed"]), "level": float(boot["level"]),
                   "compare": [list(p) for p in compare], "smoke": s.smoke,
                   "systems": dict(cfg["systems"])},
        "manifest_sha256": s.manifest_sha256, "taxonomy_sha256": tx.taxonomy_sha256(t),
        "textprep_version": TEXTPREP_VERSION, "n_test_rows": len(ids), "n_groups": len(set(rows.groups)),
        "labels": list(two.labels), "components": comps,
        "routing": {n: r.routing_counts() for n, r in results.items()},
        "evaluation": evaluation, "cotov": cotov,
        "library_versions": library_versions(), "python": platform.python_version(),
    }
    doc = publish(perf_path, pred_path, summary, predictions, s.overwrite)
    log(f"saved {perf_path} and {pred_path}")
    return doc


# --------------------------------------------------------------------------- report
def _fmt(e: Optional[Mapping], digits: int = 3) -> str:
    if not e or e.get("value") is None:
        return "n/a"
    ci = e.get("ci") or [None, None]
    lo, hi = ("n/a" if v is None else f"{v:.{digits}f}" for v in ci)
    return f"{e['value']:.{digits}f} [{lo}, {hi}]"


def report_lines(summary: Mapping) -> List[str]:
    """Main table of a published summary: per system micro P/R/F1, macro F1, subset accuracy with CIs, the paired
    differences, the filter false-negative split, routing counts and the BLOCKED line when Cotov is blocked."""
    ev = summary["evaluation"]
    lines = [f"{'NON-PRODUCTION SMOKE' if summary.get('non_production') else 'PRODUCTION'} | status {summary['status']} | "
             f"{summary['n_test_rows']} test rows, {summary['n_groups']} groups, B={summary['config']['B']}"]
    for name, sy in ev["systems"].items():
        e = sy["end_to_end"]
        lines.append(f"{name}: micro P {_fmt(e['micro_precision'])} R {_fmt(e['micro_recall'])} F1 {_fmt(e['micro_f1'])} | "
                     f"macro F1 {_fmt(e['macro_f1'])} | subset acc {_fmt(e['subset_accuracy'])}")
    for pair, d in ev.get("paired", {}).items():
        lines.append(f"paired {pair}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in d.items()))
    f = ev["systems"].get("two_step", {}).get("filter")
    if f:
        v = lambda k: (f.get(k) or {}).get("value")
        lines.append(f"filter: recall {_fmt(f.get('recall'))}, FN rate {_fmt(f.get('false_negative_rate'))}, dropped relevant "
                     f"mapped {v('dropped_relevant_mapped')} / unmapped {v('dropped_relevant_unmapped')}")
    for name, r in summary["routing"].items():
        lines.append(f"routing {name}: " + ", ".join(f"{k} {v}" for k, v in r.items() if not isinstance(v, dict))
                     + "; sub_none_conflict " + str(r["sub_none_conflict"]) + "; parent_without_child " + str(r["parent_without_child"]))
    cv = summary["cotov"]
    if cv.get("status") == "blocked":
        lines.append(f"BLOCKED: {cv['reason']}")
    else:
        for name, node in cv["systems"].items():
            for k, b in node.items():
                ci = b["bootstrap"]["ci"]
                lines.append(f"Cotov {name} {k} (n={b['n']}; {b['definition']}): F1 {b['f1']:.3f} {ci['f1']}, recall {b['recall']:.3f}, "
                             f"precision {b['precision']:.3f}")
    return lines
