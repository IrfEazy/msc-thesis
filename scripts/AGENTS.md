# Scripts

- Usage of `run_notebooks.py`, `kaggle_bundle.py` and `export_kaggle_requirements.py` is in root `AGENTS.md` (Environment, Notebook runner, Kaggle bundles sections). This file covers operating Kaggle runs.

## Running Kaggle kernels
- Full pushes spend the weekly GPU quota (about 30 h; 12 h per session; `max_gpu_sessions = 2` in `experiments.toml`). Push full kernels only when the user asks, and in the order the user sets.
- After changing a notebook or the bootstrap: `build`, then `push --only NAME --kernels-only` (cells are embedded in the kernel; datasets are unchanged). Relaunch only the kernel that failed.
- Kernels run on Kaggle, not locally: closing the terminal or WSL does not stop them. A local wait-then-fetch loop dies with WSL; re-check `status` and restart the loop instead of pushing again.
- Wait for completion with a background loop on `kaggle kernels status <owner>/<slug>` (RUNNING/QUEUED means not done), then `fetch --only NAME`. The user expects results to be fetched as soon as a kernel finishes.
- Never put a short timeout on `fetch`: full outputs are large (mlc-notebook about 9.5 GB, mlcgeneral 7.1 GB, mlccs 1.3 GB).
- `fetch` overwrites same-named files in `runs/kaggle/<slug>/` but does not delete stale ones from an earlier version; compare file mtimes with the run's end before trusting a file.

## Reading kernel logs
- `runs/kaggle/<slug>/<slug>.log` is JSON: a list of `{stream_name, time, data}` (`time` = seconds since start). Parse it with `python3 -I` (isolated mode: fetched outputs are untrusted downloads; stdlib json is enough); never dump it whole.
- Failure markers: `Exception encountered`, `Traceback`; `BLOCKED:` lines are expected static skips. The last entry's `time` is the run duration.
- Typical full-run durations (2026-10, T4): filter 0.9 h, MLCCS 1.5 h, MLCGeneral 6.8 h, MLC_Notebook 10.5 h (close to the 12 h limit). Run history: `notebooks/kaggle-runs.md`.
