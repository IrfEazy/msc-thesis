#!/usr/bin/env python3
"""Run the notebooks declared in a manifest (default: experiments.toml) headlessly.

All policy (targets, working directories, environment, inputs, outputs, timeouts,
static blocks, archives) lives in the manifest; this script only executes it.
Run from anywhere with: uv run python scripts/run_notebooks.py --help
"""
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "experiments.toml"
BLOCKED_PREFIX = "BLOCKED:"
FAILING = {"failed", "timeout"}
ALLOWED_TARGETS = ("local", "kaggle")
DEFAULT_TARGET = ALLOWED_TARGETS[0]
REMOTE_TARGETS = ("kaggle",)  # never executed by this runner


def utcnow():
    return datetime.now(timezone.utc)


def subst(text, run_dir):
    return text.replace("{repo}", str(REPO)).replace("{run_dir}", str(run_dir))


def resolve_path(text, run_dir):
    p = Path(subst(text, run_dir))
    return p if p.is_absolute() else REPO / p


def load_manifest(path):
    with open(path, "rb") as f:
        manifest = tomllib.load(f)
    settings = {"runs_dir": "runs", "kernel": "python3", "timeout": 3600}
    settings.update(manifest.get("settings", {}))
    manifest["settings"] = settings
    manifest.setdefault("env", {})
    manifest.setdefault("archives", [])
    manifest.setdefault("notebooks", [])
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    seen = set()
    for i, n in enumerate(manifest["notebooks"]):
        missing = [k for k in ("name", "path") if k not in n]
        if missing:
            raise ValueError(f"notebooks[{i}]: missing required key(s) {missing}")
        if n["name"] in seen:
            raise ValueError(f"duplicate notebook name: {n['name']!r}")
        seen.add(n["name"])
        if n.get("target") not in ALLOWED_TARGETS:
            raise ValueError(f"notebook {n['name']!r}: target {n.get('target')!r} "
                             f"not in {list(ALLOWED_TARGETS)}")


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def revision():
    head = git("rev-parse", "HEAD") or "unknown"
    if git("status", "--porcelain", "--untracked-files=no"):
        head += "-dirty"
    return head


def lock_hash():
    lock = REPO / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


# ---------------------------------------------------------------- prepare

def safe_extract(zip_path, dest):
    """Extract without overwriting existing files or escaping dest."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for info in members:  # validate everything before extracting anything
            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                raise ValueError(f"unsafe path in {zip_path}: {info.filename}")
        for info in members:
            target = (dest / info.filename).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "xb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            extracted += 1
    return extracted


def prepare(manifest):
    ok = True
    for arc in manifest["archives"]:
        zip_path, dest = resolve_path(arc["path"], ""), resolve_path(arc["dest"], "")
        if dest.is_dir() and any(dest.iterdir()):
            print(f"prepare: {arc['dest']} already present, skipped")
        elif not zip_path.is_file():
            print(f"prepare: BLOCKED: archive {arc['path']} not found")
            ok = False
        else:
            try:
                n = safe_extract(zip_path, dest)
            except (ValueError, zipfile.BadZipFile, OSError) as exc:
                print(f"prepare: FAILED: {arc['path']}: {exc}")
                ok = False
                continue
            print(f"prepare: extracted {n} files from {arc['path']} to {arc['dest']}")
    return ok


# ---------------------------------------------------------------- execution

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def blocked_lines(nb):
    found = []
    for idx, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in text.splitlines():
                if line.startswith(BLOCKED_PREFIX):
                    found.append({"cell": idx, "line": line})
    return found


def run_notebook(entry, manifest, run_dir):
    result = {
        "name": entry["name"], "path": entry["path"], "target": entry["target"], "status": None,
        "duration_s": 0.0, "executed_path": None, "failed_cell": None, "error": None,
        "blocked_cells": [], "missing_inputs": [], "missing_outputs": [],
    }
    try:
        return _run_notebook(entry, manifest, run_dir, result)
    except Exception as exc:  # setup or bookkeeping failure: isolate to this notebook
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return result


def _run_notebook(entry, manifest, run_dir, result):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    settings = manifest["settings"]
    name = entry["name"]
    if entry.get("blocked"):
        result["status"] = "blocked"
        result["error"] = entry["blocked"]
        return result
    cwd = resolve_path(entry.get("cwd", "{repo}"), run_dir)
    inputs = [resolve_path(p, run_dir) for p in entry.get("inputs", [])]
    outputs = [resolve_path(p, run_dir) for p in entry.get("outputs", [])]
    result["missing_inputs"] = [str(p) for p in inputs if not p.exists()]
    if result["missing_inputs"]:
        result["status"] = "blocked"
        return result

    # Manifest env ([env] then notebook env); variables already in the process win.
    manifest_env = {k: subst(str(v), run_dir)
                    for k, v in {**manifest["env"], **entry.get("env", {})}.items()}
    kernel_env = {**os.environ}
    for k, v in manifest_env.items():
        kernel_env.setdefault(k, v)

    before = {}
    for p in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
        before[p] = p.stat().st_mtime_ns if p.exists() else None
    timeout = entry.get("timeout", settings["timeout"])
    nb = nbformat.read(resolve_path(entry["path"], run_dir), as_version=4)
    nb = copy.deepcopy(nb)
    executed = run_dir / "notebooks" / f"{safe_name(name)}.ipynb"
    log = run_dir / "logs" / f"{safe_name(name)}.log"
    executed.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    current = {"cell": None}

    def on_start(cell=None, cell_index=None, **_):
        current["cell"] = cell_index

    client = NotebookClient(
        nb, timeout=timeout, kernel_name=settings["kernel"], allow_errors=False,
        resources={"metadata": {"path": str(cwd)}},
        on_cell_start=on_start,
    )
    start = time.monotonic()
    try:
        client.execute(env=kernel_env)
        result["status"] = "passed"
    except CellExecutionError as exc:
        result["status"] = "failed"
        result["failed_cell"] = current["cell"]
        text = f"{exc.ename}: {exc.evalue}" if getattr(exc, "ename", None) else str(exc)
        result["error"] = text[:500]
    except TimeoutError as exc:
        result["status"] = "timeout"
        result["failed_cell"] = current["cell"]
        result["error"] = f"cell timeout after {timeout}s"
    except Exception as exc:  # kernel start failure, bad cwd, ...
        result["status"] = "failed"
        result["failed_cell"] = current["cell"]
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    result["duration_s"] = round(time.monotonic() - start, 2)

    nbformat.write(nb, executed)
    result["executed_path"] = str(executed)
    blocked = blocked_lines(nb)
    result["blocked_cells"] = blocked
    if result["status"] == "passed":
        # An output must be created or modified by this run, not merely pre-exist.
        result["missing_outputs"] = [
            str(p) for p in outputs
            if not p.exists() or p.stat().st_mtime_ns == before[p]]
        if result["missing_outputs"]:
            result["status"] = "failed"
            result["error"] = "declared outputs missing or not updated by execution"
        elif blocked:
            result["status"] = "passed-with-blocked"

    with open(log, "w", encoding="utf-8") as f:
        f.write(f"{name}: {result['status']} ({result['duration_s']}s)\n")
        if result["error"]:
            f.write(f"error (cell {result['failed_cell']}): {result['error']}\n")
        for idx, cell in enumerate(nb.cells):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    text = out.get("text", "")
                    f.write(f"--- cell {idx} {out.get('name')} ---\n"
                            f"{''.join(text) if isinstance(text, list) else text}\n")
                elif out.get("output_type") == "error":
                    f.write(f"--- cell {idx} error ---\n{out.get('ename')}: {out.get('evalue')}\n")
    return result


def write_summary(run_dir, summary):
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = ["| name | target | status | duration_s | detail |", "|---|---|---|---|---|"]
    for n in summary["notebooks"]:
        detail = []
        if n["failed_cell"] is not None:
            detail.append(f"cell {n['failed_cell']}")
        if n["error"]:
            detail.append(n["error"].replace("|", "\\|").replace("\n", " "))
        if n["missing_inputs"]:
            detail.append("missing inputs: " + ", ".join(n["missing_inputs"]))
        if n["missing_outputs"]:
            detail.append("missing outputs: " + ", ".join(n["missing_outputs"]))
        for b in n["blocked_cells"]:
            detail.append(f"cell {b['cell']}: {b['line']}".replace("|", "\\|"))
        rows.append(f"| {n['name']} | {n['target']} | {n['status']} | {n['duration_s']} | "
                    f"{'; '.join(detail)} |")
    header = (f"# Notebook run\n\n- revision: {summary['revision']}\n- target: {summary['target']}\n"
              f"- started: {summary['started']}\n- finished: {summary['finished']}\n"
              f"- manifest: {summary['manifest']}\n\n")
    (run_dir / "summary.md").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--list", action="store_true", help="list manifest notebooks and exit")
    ap.add_argument("--prepare", action="store_true",
                    help="extract missing archives; alone it does not run notebooks")
    ap.add_argument("--target", choices=ALLOWED_TARGETS, default=None)
    ap.add_argument("--only", action="append", default=[], metavar="NAME")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"invalid manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2
    entries = manifest["notebooks"]
    if args.list:
        for n in entries:
            print("\t".join([n["name"], n["target"], n["path"], n.get("blocked", "")]))
        return 0
    unknown = set(args.only) - {n["name"] for n in entries}
    if unknown:
        print(f"unknown notebook name(s): {sorted(unknown)}", file=sys.stderr)
        return 2
    if args.prepare:
        ok = prepare(manifest)
        if not ok:
            print("prepare failed; notebooks not executed", file=sys.stderr)
            return 1
        if args.target is None and not args.only:
            return 0

    target = args.target or DEFAULT_TARGET
    if args.only:
        entries = [n for n in entries if n["name"] in args.only]
    if args.out:
        run_dir = Path(args.out).resolve()
    else:
        run_dir = resolve_path(manifest["settings"]["runs_dir"], "") / utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = utcnow()
    results = []
    for entry in entries:
        if entry["target"] != target or target in REMOTE_TARGETS:
            r = {"name": entry["name"], "path": entry["path"], "target": entry["target"],
                 "status": "skipped", "duration_s": 0.0, "executed_path": None, "failed_cell": None,
                 "error": None, "blocked_cells": [], "missing_inputs": [], "missing_outputs": []}
        else:
            print(f"running {entry['name']} ...", flush=True)
            r = run_notebook(entry, manifest, run_dir)
        print(f"{r['status']:>20}  {r['name']}"
              + (f"  [{r['error']}]" if r["error"] and r["status"] != "passed" else ""), flush=True)
        results.append(r)

    summary = {
        "revision": revision(), "uv_lock_sha256": lock_hash(),
        "started": started.isoformat(), "finished": utcnow().isoformat(),
        "manifest": str(Path(args.manifest).resolve()), "target": target, "notebooks": results,
    }
    write_summary(run_dir, summary)
    print(f"summary: {run_dir / 'summary.md'}")
    return 1 if any(r["status"] in FAILING for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
