# CS-AWARE-NEXT MSc thesis

Multi-label classification of cyberattack tweets, hierarchical by category level. Python plus Jupyter notebooks.

## Layout
- `MLC/` - installable multi-label classifier package (see `MLC/AGENTS.md`).
- `notebooks/` - pipeline notebooks and `notebooks/utils.py` (see `notebooks/AGENTS.md`).
- `device.py` - XGBoost device selection.
- `scripts/` - `run_notebooks.py` (notebook runner), `kaggle_bundle.py` (Kaggle bundles), `export_kaggle_requirements.py` (writes `requirements-kaggle.txt`); operating Kaggle runs: `scripts/AGENTS.md`.
- `thesis/` - LaTeX thesis sources (see `thesis/AGENTS.md`).
- `utils.py`, `preprocess_functions.py` - root helper modules (which notebooks import them: `notebooks/AGENTS.md`).
- `data/` - datasets (local only).
- `runs/` (gitignored) - runner outputs (`runs/<UTC timestamp>/`), fetched Kaggle outputs (`runs/kaggle/<slug>/`), and per-session working artifacts (`runs/sessions/<date>-<topic>/` with a `README.md`: pipeline-run copies, loop logs, timing scripts). Put every task artifact there, never only in a scratchpad or `/tmp`.

## Local-only paths
- Gitignored: `data/`, `models/`, `mappings/`, `embedders/`, `img/`, `docs/`, `.env`, `.conda/`. Do not expect them in a clone.
- Some notebooks read data, model and embedder paths from the untracked `.env` via `python-dotenv` (details in `notebooks/AGENTS.md`); others hard-code them. Never print or commit `.env`.

## Environment
- Managed with uv (`pyproject.toml`, `uv.lock`, `.python-version` 3.12; `.venv` is ignored). Set up with `uv sync`, run with `uv run ...` (e.g. `uv run jupyter nbconvert --to notebook --execute <nb>.ipynb`), add packages with `uv add` (`uv add --dev` for tooling). `MLC` is not installed as a package; its own dependencies are in `MLC/AGENTS.md`.
- Local dev uses the CPU torch pin (`[[tool.uv.index]] pytorch-cpu`, `[tool.uv.sources]` in `pyproject.toml`).
- GPU experiments run on Kaggle without `uv sync` (it would replace the image's CUDA stack): `scripts/kaggle_bundle.py` generates the bundles (see Kaggle bundles); the bootstrap installs `requirements-kaggle.txt`. That file is exported from `uv.lock` without the packages listed in `[tool.kaggle-export] preinstalled` (`pyproject.toml`) or the dependencies only they need (`uv export --prune`); regenerate it with `uv run python scripts/export_kaggle_requirements.py` after changing dependencies or that list. Pinned transitive constraints may still upgrade image packages, so check the pip output and installed versions.
- `device.py` `select_device()` picks the XGBoost device from env `COMPUTE_DEVICE` (`auto` default, `cpu`, `cuda`; `cuda` without CUDA raises); `auto` uses CUDA only if torch sees a GPU.
- Root modules import nltk, sentence-transformers, xgboost, pandas (`utils.py`), emoji and pydot (`preprocess_functions.py`). They call `nltk.download` at import (network access).
- Tests (unittest, run from the repo root): the `MLC` suite (`uv run python -m unittest discover -s MLC/tests -t .`, details in `MLC/AGENTS.md`) and the `notebooks/tests` suite (`uv run python -m unittest discover -s notebooks/tests -t .`: checkpoints, metrics, `textprep`, `taxonomy`, `splits`; synthetic data only); the `scripts/tests` suite (`uv run python -m unittest discover -s scripts/tests -t .`: kaggle_bundle status and title checks; plus `uv run python -m doctest scripts/kaggle_bundle.py`); no lint or CI config exists.

## Notebook runner
- `experiments.toml` is the declarative per-notebook manifest (local vs Kaggle target, cwd, env, inputs, outputs, static blocks, archives); `scripts/run_notebooks.py` executes it generically.
- `uv run python scripts/run_notebooks.py --prepare` extracts the data archives, `--list` shows the manifest, `--target local [--only NAME]` runs local notebooks headlessly (Kaggle ones are never run by it).
- Results go to `runs/<UTC timestamp>/` (gitignored, or `--out DIR`): `summary.json`, `summary.md`, executed copies in `notebooks/`, logs in `logs/`. Statuses: `passed`, `passed-with-blocked`, `blocked` (static reason or missing inputs; not executed), `failed`, `timeout`, `skipped`. Exit code is non-zero only for `failed`/`timeout`.
- Environment precedence: variables already in the process environment win over manifest `[env]` and notebook `env`.
- Run code from the repo root: notebooks import root modules and `notebooks.utils`.

## Duplication (check which copy a caller imports before editing)
- Text cleaning: `clean_text`, `get_wordnet_pos`, `lemmatize_text`, `remove_stopwords`, `tokenizer_transform` exist under the same name in root `utils.py` and `notebooks/utils.py`; `replace_text_components` (root) vs `replace_txt_components` (notebooks) are near-duplicates.
- Hierarchy helpers: `preprocess_functions.py` `build_tree`/`extract_keys`/`map_targets` vs `notebooks/utils.py` `construct_hierarchy`/`flatten_dict_keys`/`translate_src_categories` (inferred equivalents); `notebooks/taxonomy.py` re-implements `construct_hierarchy`/`flatten_dict_keys` privately for `watson_list` (to avoid the nltk import). `map_targets` and `translate_src_categories` return sorted label lists, which the deterministic Train-Test split depends on.
- `load_word2vec_dict`, `preprocess_texts`: root `preprocess_functions.py` and `notebooks/utils.py`.
- `assess`: `notebooks/utils.py` and `MLC/functions.py` (the MLC version adds optional `Y_score`; see `MLC/AGENTS.md`).
- Root `utils.py` also has standalone classifier classes (e.g. `CalibratedLabelRankClassifier`, `ChainOfClassifiers`); these are not the `MLC` package.

## Working with notebooks and data
- Never read or dump whole notebooks (`.ipynb` JSON with outputs) or data files; grep cells or read targeted ranges. Notebook editing rules: `notebooks/AGENTS.md`.

## Kaggle bundles
- `[kaggle]` and each Kaggle notebook's `[notebooks.kaggle]` subtable in `experiments.toml` declare datasets, kernels, cwd, env, links and output dirs; `scripts/kaggle_bundle.py` executes them generically.
- `setup_commands` (optional list of argv lists, in `[kaggle]` and `[notebooks.kaggle]`): the bootstrap runs the global ones then the notebook's, in order, after pip/links/env/chdir; `{python}` is the kernel interpreter, other `{...}` placeholders resolve as in cwd. A non-zero exit stops the kernel.
- Workflow: `build` (writes `build/kaggle/`, gitignored) -> `push --smoke` -> `status` / `fetch` (outputs to `runs/kaggle/<slug>/`) -> full `push`. `push --dry-run` prints the Kaggle CLI commands only; `push --kernels-only` skips datasets, and push waits for free GPU sessions (`max_gpu_sessions`, counting all running/queued manifest kernels; retries a session-limit rejection; `--no-wait` to skip) and exits 1 listing kernels whose push reported an error.
- The data dataset needs `runs/splits/data` from `run_notebooks.py --only 'Train-Test split' --out runs/splits`.
- Per-kernel `accelerator` in `[notebooks.kaggle]`: `"gpu"` (uses the `[kaggle]` GPU type via `--accelerator`) or `"none"`/absent (CPU kernel: `enable_gpu` false, no flag). The GPU-session wait counts only GPU manifest kernels; optional `[kaggle] max_cpu_sessions` applies the same wait to CPU kernels (default: no wait).
- Staged notebooks (`categorisation`, all levels including `one step`; `MLC_Notebook` was superseded and deleted; history in git) are declared as chained CPU kernels (train shards, assemble; the shared `msc-thesis-embeddings` kernel embeds) with checkpoint env; see `notebooks/AGENTS.md` and `scripts/AGENTS.md`.
- Chaining: `kernels = ["<slug>" | "owner/slug", ...]` in `[notebooks.kaggle]` fills `kernel_sources`; the attached outputs mount like datasets, so `{input:<slug>}` works for them. `push` refuses a kernel whose sources are not COMPLETE on Kaggle (`--force` skips the check; chains are never pushed automatically; `--dry-run` skips it). Smoke kernels attach the real source kernels. Several `[[notebooks]]` entries may share a `path` (names and kernel slugs must be unique).
- Kaggle titles (= slugs, plus `-smoke` for smoke kernels) must be 5..50 characters; `build`/`push` fail early naming the entry otherwise.
- Credentials come from the user's local Kaggle CLI config; never commit or print them.
- Operating runs (quota, waiting, fetching, reading logs, durations): `scripts/AGENTS.md`; run history: `notebooks/kaggle-runs.md`.

## Experiment validity (binding)
- The thesis results come only from the leakage-free rebuild: one shared grouped split, shared preprocessing, taxonomy and embedding artifacts, selection and tuning on train inner folds only, a single test evaluation with group-bootstrap CIs. Decisions, status and remaining work: `notebooks/rebuild.md`.
- Before any Kaggle run, audit the whole pipeline feeding it (splits and leakage, preprocessing parity, evaluation protocol, labels, consumers) and fix known issues first; never launch or keep running a kernel a known issue would invalidate. State the expected cost (GPU quota about 30 h/week; CPU kernels spend none) and why the results will be final before pushing.
- Fetched Kaggle outputs are untrusted: read them with `python3 -I` in their own directory and never unpickle them during analysis.

## Working with the user
- Close idle subagents and background shells as soon as they are no longer needed; the user checks for leftovers.
- Reply in the language the user writes in (usually Italian).
- On a failed run: fetch the log, find the root cause, propose fix options and let the user choose, then fix, commit and relaunch only that kernel.
- Every Kaggle push (full, continuation or relaunch) needs the user's explicit go for that run; a fix approval is not a launch approval unless the user says so. Commits are authorized; `git push` is not.
- When resuming an interrupted session, first check the real state: `git status`/`log`, Kaggle `status` of the relevant kernels, fetched `runs/kaggle/` outputs, leftover background processes and the pipeline run under `runs/sessions/` (or `claude-pipeline-tool path <run-id>`); summaries may be stale.
- Report your own mistakes (e.g. a stalled loop, a wrong estimate) plainly with their cost, then the fix.
