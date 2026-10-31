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

## Local-only paths
- Gitignored: `data/`, `models/`, `mappings/`, `embedders/`, `img/`, `docs/`, `.env`, `.conda/`. Do not expect them in a clone.
- Some notebooks read data, model and embedder paths from the untracked `.env` via `python-dotenv` (details in `notebooks/AGENTS.md`); others hard-code them. Never print or commit `.env`.

## Environment
- Managed with uv (`pyproject.toml`, `uv.lock`, `.python-version` 3.12; `.venv` is ignored). Set up with `uv sync`, run with `uv run ...` (e.g. `uv run jupyter nbconvert --to notebook --execute <nb>.ipynb`), add packages with `uv add` (`uv add --dev` for tooling). `MLC` is not installed as a package; its own dependencies are in `MLC/AGENTS.md`.
- Local dev uses the CPU torch pin (`[[tool.uv.index]] pytorch-cpu`, `[tool.uv.sources]` in `pyproject.toml`).
- GPU experiments run on Kaggle without `uv sync` (it would replace the image's CUDA stack): `scripts/kaggle_bundle.py` generates the bundles (see Kaggle bundles); the bootstrap installs `requirements-kaggle.txt`. That file is exported from `uv.lock` without the packages listed in `[tool.kaggle-export] preinstalled` (`pyproject.toml`) or the dependencies only they need (`uv export --prune`); regenerate it with `uv run python scripts/export_kaggle_requirements.py` after changing dependencies or that list. Pinned transitive constraints may still upgrade image packages, so check the pip output and installed versions.
- `device.py` `select_device()` picks the XGBoost device from env `COMPUTE_DEVICE` (`auto` default, `cpu`, `cuda`; `cuda` without CUDA raises); `auto` uses CUDA only if torch sees a GPU.
- Root modules import nltk, sentence-transformers, xgboost, pandas (`utils.py`), emoji and pydot (`preprocess_functions.py`). They call `nltk.download` at import (network access).
- The only tests are the `MLC` unittest suite (`uv run python -m unittest discover -s MLC/tests -t .`, details in `MLC/AGENTS.md`); no lint or CI config exists.

## Notebook runner
- `experiments.toml` is the declarative per-notebook manifest (local vs Kaggle target, cwd, env, inputs, outputs, static blocks, archives); `scripts/run_notebooks.py` executes it generically.
- `uv run python scripts/run_notebooks.py --prepare` extracts the data archives, `--list` shows the manifest, `--target local [--only NAME]` runs local notebooks headlessly (Kaggle ones are never run by it).
- Results go to `runs/<UTC timestamp>/` (gitignored, or `--out DIR`): `summary.json`, `summary.md`, executed copies in `notebooks/`, logs in `logs/`. Statuses: `passed`, `passed-with-blocked`, `blocked` (static reason or missing inputs; not executed), `failed`, `timeout`, `skipped`. Exit code is non-zero only for `failed`/`timeout`.
- Environment precedence: variables already in the process environment win over manifest `[env]` and notebook `env`.
- Run code from the repo root: notebooks import root modules and `notebooks.utils`.

## Duplication (check which copy a caller imports before editing)
- Text cleaning: `clean_text`, `get_wordnet_pos`, `lemmatize_text`, `remove_stopwords`, `tokenizer_transform` exist under the same name in root `utils.py` and `notebooks/utils.py`; `replace_text_components` (root) vs `replace_txt_components` (notebooks) are near-duplicates.
- Hierarchy helpers: `preprocess_functions.py` `build_tree`/`extract_keys`/`map_targets` vs `notebooks/utils.py` `construct_hierarchy`/`flatten_dict_keys`/`translate_src_categories` (inferred equivalents). `map_targets` and `translate_src_categories` (and the local `map_targets` copies in `MLCGeneral`/`MLCCS`) return sorted label lists, which the deterministic Train-Test split depends on.
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
- Credentials come from the user's local Kaggle CLI config; never commit or print them.
- Operating runs (quota, waiting, fetching, reading logs, durations): `scripts/AGENTS.md`; run history: `notebooks/kaggle-runs.md`.

## Working with the user
- Reply in the language the user writes in (usually Italian).
- On a failed run: fetch the log, find the root cause, propose fix options and let the user choose, then fix, commit and relaunch only that kernel.
