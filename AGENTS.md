# CS-AWARE-NEXT MSc thesis

Multi-label classification of cyberattack tweets, hierarchical by category level. Python plus Jupyter notebooks.

## Layout
- `MLC/` - installable multi-label classifier package (see `MLC/AGENTS.md`).
- `notebooks/` - pipeline notebooks and `notebooks/utils.py` (see `notebooks/AGENTS.md`).
- `device.py` - XGBoost device selection; `scripts/export_kaggle_requirements.py` - writes `requirements-kaggle.txt`.
- `utils.py`, `preprocess_functions.py` - root helper modules (which notebooks import them: `notebooks/AGENTS.md`).
- `data/` - datasets (local only).

## Local-only paths
- Gitignored: `data/`, `models/`, `mappings/`, `embedders/`, `img/`, `docs/`, `.env`, `.conda/`. Do not expect them in a clone.
- Some notebooks read data, model and embedder paths from the untracked `.env` via `python-dotenv` (details in `notebooks/AGENTS.md`); others hard-code them. Never print or commit `.env`.

## Environment
- Managed with uv (`pyproject.toml`, `uv.lock`, `.python-version` 3.12; `.venv` is ignored). Set up with `uv sync`, run with `uv run ...` (e.g. `uv run jupyter nbconvert --to notebook --execute <nb>.ipynb`), add packages with `uv add` (`uv add --dev` for tooling). `MLC` is not installed as a package; its own dependencies are in `MLC/AGENTS.md`.
- Local dev uses the CPU torch pin (`[[tool.uv.index]] pytorch-cpu`, `[tool.uv.sources]` in `pyproject.toml`).
- GPU experiments run on Kaggle without `uv sync` (it would replace the image's CUDA stack): upload the repo code, then `pip install -r requirements-kaggle.txt`. That file is exported from `uv.lock` without the packages listed in `[tool.kaggle-export] preinstalled` (`pyproject.toml`) or the dependencies only they need (`uv export --prune`); regenerate it with `uv run python scripts/export_kaggle_requirements.py` after changing dependencies or that list. Pinned transitive constraints may still upgrade image packages, so check the pip output and installed versions.
- `device.py` `select_device()` picks the XGBoost device from env `COMPUTE_DEVICE` (`auto` default, `cpu`, `cuda`; `cuda` without CUDA raises); `auto` uses CUDA only if torch sees a GPU.
- Root modules import nltk, sentence-transformers, xgboost, pandas (`utils.py`), emoji and pydot (`preprocess_functions.py`). They call `nltk.download` at import (network access).
- No tests, lint or CI config exist.
- Run code from the repo root: notebooks import root modules and `notebooks.utils`.

## Duplication (check which copy a caller imports before editing)
- Text cleaning: `clean_text`, `get_wordnet_pos`, `lemmatize_text`, `remove_stopwords`, `tokenizer_transform` exist under the same name in root `utils.py` and `notebooks/utils.py`; `replace_text_components` (root) vs `replace_txt_components` (notebooks) are near-duplicates.
- Hierarchy helpers: `preprocess_functions.py` `build_tree`/`extract_keys`/`map_targets` vs `notebooks/utils.py` `construct_hierarchy`/`flatten_dict_keys`/`translate_src_categories` (inferred equivalents).
- `load_word2vec_dict`, `preprocess_texts`: root `preprocess_functions.py` and `notebooks/utils.py`.
- `assess`: `notebooks/utils.py` and `MLC/functions.py`.
- Root `utils.py` also has standalone classifier classes (e.g. `CalibratedLabelRankClassifier`, `ChainOfClassifiers`); these are not the `MLC` package.

## Working with notebooks and data
- Never read or dump whole notebooks (`.ipynb` JSON with outputs) or data files; grep cells or read targeted ranges. Notebook editing rules: `notebooks/AGENTS.md`.
