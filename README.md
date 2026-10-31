# CS-AWARE-NEXT - MSc Thesis

## Overview

This repository contains the code and LaTeX sources of the MSc thesis *Multi-Label Classification in the CS-Aware-Next Project: Evaluating ML Algorithms for Cybersecurity Filtering and Classification* by Irfan Cela (Computer Science Engineering, Politecnico di Milano, A.Y. 2023-24). Supervised by Prof. Cinzia Cappiello, co-supervised by Camilla Sancricca. The work supports the CS-AWARE-NEXT project, funded by the European Union, which aims to bring AI-based cybersecurity situational awareness from social media into security workflows. The system filters cybersecurity-relevant tweets and classifies them into a hierarchy of categories.

## Approach

1. **Filtering**: a binary classifier keeps only cybersecurity-relevant tweets. Logistic Regression, Naive Bayes, Random Forest and XGBoost are compared.
2. **Hierarchical multi-label classification**: the relevant tweets are classified per category level: the base notebook (`MultiLabelClassification.ipynb`) plus one notebook per sub-category level (Computer, ComputerNetworking, ComputerSecurity, Hardware, OperatingSystems, Software). The `MLC` package implements the multi-label classifiers: BR, CC, CDN, CLR, HOMER, LP, MBR, PSt and RAkEL.
3. **Text representations**: static GloVe embeddings are compared with context-aware encoders (DistilRoBERTa, SBERT and the cybersecurity-specific ATT&CK-BERT).

Pipeline order and notebook details: [notebooks/AGENTS.md](notebooks/AGENTS.md).

## Repository layout

- `MLC/` - multi-label classifier package (not installed by `uv sync`); see [MLC/AGENTS.md](MLC/AGENTS.md).
- `notebooks/` - pipeline notebooks and `notebooks/utils.py`; see [notebooks/AGENTS.md](notebooks/AGENTS.md).
- `scripts/` - notebook runner (`run_notebooks.py`), Kaggle bundle generator (`kaggle_bundle.py`) and Kaggle requirements export (`export_kaggle_requirements.py`); see [scripts/AGENTS.md](scripts/AGENTS.md).
- `thesis/` - LaTeX thesis sources; see [thesis/AGENTS.md](thesis/AGENTS.md).
- `device.py` - XGBoost device selection (`COMPUTE_DEVICE`).
- `utils.py`, `preprocess_functions.py` - root helper modules.
- `experiments.toml` - declarative manifest of notebooks, environment, inputs and Kaggle targets.
- `data/`, `models/`, `embedders/` - local-only data, models and embedders (see below).

Project-wide conventions and navigation: [AGENTS.md](AGENTS.md).

## Getting started

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                   # create .venv from uv.lock (CPU torch pin)
uv run python scripts/run_notebooks.py --prepare          # extract the data archives
uv run python scripts/run_notebooks.py --list             # show the notebook manifest
uv run python scripts/run_notebooks.py --target local     # run local notebooks headlessly
uv run python scripts/run_notebooks.py --target local --only 'Train-Test split'
```

Run commands from the repository root. Results are written to `runs/<UTC timestamp>/` (gitignored). Run statuses and options: [AGENTS.md](AGENTS.md#notebook-runner).

## GPU runs on Kaggle

GPU experiments run on Kaggle, not with `uv sync`. `scripts/kaggle_bundle.py` builds the bundles from `experiments.toml`, and the bootstrap installs `requirements-kaggle.txt`. Operating steps (build, push, status, fetch, quota) are in [scripts/AGENTS.md](scripts/AGENTS.md); run history is in [notebooks/kaggle-runs.md](notebooks/kaggle-runs.md).

## Data and local-only files

The following are gitignored and are not part of a clone: `data/`, `models/`, `mappings/`, `embedders/`, `img/`, `docs/`, `.env` and `.conda/`. Some notebooks read data, model and embedder paths from the untracked `.env` (see [notebooks/AGENTS.md](notebooks/AGENTS.md)). Never commit `.env`.

The Cotov (NistPosts) dataset is unrecoverable. The sections that depend on it print a `BLOCKED:` line and are skipped; the notebook runner reports them as blocked.

## License

Released into the public domain under [The Unlicense](LICENSE).
