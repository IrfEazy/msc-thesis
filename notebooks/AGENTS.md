# Notebooks

## Pipeline order (inferred from names and imports)
- "base" here means `MultiLabelClassification.ipynb`.
- `filter` -> `Train-Test split` -> `MultiLabelClassification<Level>` per hierarchy level (base, Computer, ComputerNetworking, ComputerSecurity, Hardware, OperatingSystems, Software, SMOTE variant) -> `SystemLabelAssignment`.
- Alternative one-step branch: `MLC_Notebook`, `MLCGeneral`, `MLCCS`, built on the `MLC` package.

## Imports
- `filter`, `Train-Test split`: `notebooks.utils`; `MLC_Notebook`, `MLCGeneral`, `MLCCS`: `MLC`, `notebooks.utils` and `notebooks.checkpoints` (checkpoint store; repo root must be on `sys.path`, as the Kaggle bootstrap does).
- `MultiLabelClassification*` and `SystemLabelAssignment`: root `utils` and `preprocess_functions` (SMOTE variant: `utils` only).

## Environment variables
- `filter`, `Train-Test split` and `MLC_Notebook` read env vars (`load_dotenv()`, then `os.chdir(ROOT)`); values are in the local `.env` (see root `AGENTS.md`). `MLCGeneral` and `MLCCS` read only the run controls below (no `.env`, relative paths). The other notebooks use hard-coded or relative paths.
- Names seen: `ROOT`, `RANDOM_STATE`, `TEST_SIZE`, `TWEETS_CSV`, `FILTER_TRAIN_CSV`, `FILTER_TEST_CSV`, `TEMP_CATEGORIES_TRAIN_CSV`, `TEMP_CATEGORIES_TEST_CSV`, `ONE_STEP_CATEGORIES_TRAIN_CSV`, `ONE_STEP_CATEGORIES_TEST_CSV`, `COTOV_CSV`, `CV`, `FILTER_MODELS`, `FILTERS_PERFORMANCES_CSV`, `OPTIMAL_FILTER_PICKLE`, `OVERWRITE`, `MODELS_DIR`, `MAPPINGS_DIR`, embedder paths (`SBERT_PATH`, `ATTACK_BERT_PATH`, `DISTILROBERTA_PATH`, `GLOVE_6B_{50,100,200,300}D_PATH`).
- `filter`: `OVERWRITE` comes from env `OVERWRITE` (default false); the manifest sets it true so the tuned filter is saved before it is reloaded.

## Staged, checkpointed runs (`MLC_Notebook`, `MLCGeneral`, `MLCCS`)
- Stage env: `MLC_STAGE` (`MLC_Notebook`) or `STAGE` (`MLCGeneral`, `MLCCS`) = `train`, `assemble` or `all` (default; `all` trains then assembles in one session). `MLC_Notebook` also has `embed` (embeds train/test texts once into the checkpoint store; `train` fails closed without it). Production runs use no other overrides.
- Units: `MLC_UNITS` (`MLC_Notebook`, train only) = comma-separated `<level>:<embedder>` (levels: general, computer security, operating systems, software, hardware); `EMBEDDERS` (`MLCGeneral`, `MLCCS`, train only) = comma-separated embedder names. Embedder names: `GloVe.6B.50D/100D/200D/300D`, `DistilRoBERTa`, `SBERT`, `ATT&CK-BERT`. Assemble rejects any subset (it must cover the full configuration).
- Checkpoints (`notebooks/checkpoints.py`): `CHECKPOINT_DIR` (default `./checkpoints`, gitignored) is the output store; `CHECKPOINT_INPUTS` (`:`-separated, globs ok) are read-only earlier stores. Identity (config, code and data hashes, module versions incl. the torch build tag) must match across stages, so all stages of a notebook run on the same image and with identical env strings (`ROOT`, `RANDOM_STATE`, `TEST_SIZE`, split CSVs, embedder paths, `MAPPINGS_DIR`, `MODELS_DIR`).
- Identity extras: `MLC_Notebook` adds `cell_sources` (`inspect.getsource` of the training functions defined before the identity cell); `MLCGeneral`/`MLCCS` add `training_cells_sha256` (own cells from `# NOTEBOOK-CELLS-START` up to the identity cell, from the IPython input history `_ih`).
- Provenance digest (`MLC_Notebook`: `RESULT/PROVENANCE`; `MLCGeneral`/`MLCCS`: `PROVENANCE_DIGEST`) covers the assembly code (`MLCGeneral`/`MLCCS`: `_ih` cells from `# ASSEMBLY-CODE-START` to `# ASSEMBLY-CODE-END`; `MLC_Notebook`: the assembly cell), checkpoint sha256s, validation/test digests and the run identity; assembly-only fixes keep training checkpoints.
- Markers: in `MLCGeneral`/`MLCCS`, `# NOTEBOOK-CELLS-START`, `# ASSEMBLY-CODE-START` and `# ASSEMBLY-CODE-END` must each occur exactly once (duplicates fail closed); `MLC_Notebook`'s assembly cell starts with the exact line `# ASSEMBLY-CELL (marker: the provenance digest hashes the text of this cell)`, which is not duplicate-checked.
- `MLC_Notebook`: editing a hashed training function (`HASHED_CODE`), the config cell or the hashed data starts a new identity; old checkpoints fail the identity check and are not reused.
- Re-running a training cell (before the identity cell) in a live kernel silently changes the `MLCGeneral`/`MLCCS` hash, or raises if the cell carries a marker; use batch execution for real runs.
- Budget (train): `CHECKPOINT_BUDGET_HOURS` and `CHECKPOINT_MARGIN_MINUTES` (the margin must cover the longest single fit); optional `CHECKPOINT_SESSION_START`, `CHECKPOINT_FINALISE_MINUTES`. When the budget stops training the status is INCOMPLETE, the notebook still exits 0 and assembly is skipped; rerun (or add a continuation shard, `scripts/AGENTS.md`) to resume.
- Assemble: fails closed if any candidate is missing or failed to fit; `ALLOW_FAILED_CANDIDATES=1` deliberately excludes failed candidates (not part of the identity); `ALLOW_UNRESOLVED_EMBEDDERS=1` (`MLCGeneral`, `MLCCS`) allows unresolved embedder inputs. Candidates impossible by design (e.g. PSt `pruning_value` above the largest labelset count) are stored as 'infeasible' records, printed as "infeasible by design", and do not block assemble.
- Non-production local overrides (tiny checks only; `MLC_Notebook` assemble/all accept the `MLC_*` subset vars only with `MLC_LOCAL_TINY=1`): `MLC_LOCAL_TINY=1`, `MLC_METHODS`, `MLC_ALGORITHMS`, `MLC_EMBEDDERS`, `MLC_SMOKE_ROWS`, `MLC_SMOKE_TEST_ROWS` (`MLC_Notebook`); `MAX_ROWS`, `TEST_MAX_ROWS`, `TEST_EMBEDDER_DIM`, `METHODS_SUBSET`, `CLASSIFIERS_SUBSET`, `LOCAL_EMBEDDERS` (`MLCGeneral`, `MLCCS`). At assemble/all they refuse the default output location: `MODELS_OUT_DIR` (`MLCGeneral`, `MLCCS`) or `MODELS_DIR` and `MAPPINGS_DIR` (`MLC_Notebook`) must point elsewhere. Never set them in the manifest.
- Kaggle: `experiments.toml` declares CPU kernels per stage (`MLC_Notebook`: embed, one train shard per embedder, assemble; `MLCGeneral`: one shard per embedder, assemble; `MLCCS`: GloVe and transformers shards, assemble). Operating them: `scripts/AGENTS.md`.

## Recorded Python versions (`language_info.version`)
- 3.10.14: `MultiLabelClassification*`; 3.11.13: `filter`, `MLC_Notebook`; 3.13.2: `MLCCS`, `MLCGeneral`, `Train-Test split`; 2.7.6: `SystemLabelAssignment` (likely stale metadata).

## Device selection
- `filter` and `MultiLabelClassification*` (except SMOTE) pass `device=select_device()` (root `device.py`, env `COMPUTE_DEVICE`: auto, cpu or cuda) to every active `XGBClassifier`.

## Legacy notebook
- `MultiLabelClassificationSMOTE.ipynb` is legacy and does not run: it imports `MultiLabelClassifier` and SMOTE helpers from `utils`, which were deliberately removed in commit 04ce252. They are recoverable from `04ce252^`, which also needed `bayes_opt`.

## Notebook behaviour
- `filter`: the tuned final model is an `XGBClassifier` built from `hyperopt.space_eval(space, best)` (ints cast), matching the XGBoost search.
- `MLC_Notebook`: `System.predict` embeds each module's input with that module's trained embedder, parsed from the text after the last " - " in the winning model's `name` (e.g. `HOMER[RF] - ATT&CK-BERT`); it raises `ValueError` if a name lacks it or a module's models disagree. It needs the `EMBEDDING_SOURCES` cell to have run.
- Kaggle run history and results: `kaggle-runs.md`.

## Editing
- Kaggle kernels are generated by `scripts/kaggle_bundle.py` (bootstrap cell + unchanged source cells); never edit source notebooks for Kaggle.
- Edit source cells only; do not rewrite outputs or metadata unless asked.
- The Cotov (NistPosts) dataset (`data/anastasia.csv`, by Anastasia Cotov) is not in the repo: the "Performance over Cotov's Dataset" section of `filter.ipynb` and the "Anastasia's Dataset" section of `SystemLabelAssignment.ipynb` print a `BLOCKED:` line and are skipped locally, and `scripts/run_notebooks.py` reports them as blocked. The Kaggle `filter` run reads it from the attached dataset `irfancela/anastasias-twitter-dataset` (`COTOV_CSV` in `experiments.toml`; `filter.ipynb` reads `COTOV_CSV`).
- `experiments.toml` (repo root) declares per-notebook targets (local vs Kaggle) and run settings.
