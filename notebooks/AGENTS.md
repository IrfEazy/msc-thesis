# Notebooks

## Pipeline order (inferred from names and imports)
- "base" here means `MultiLabelClassification.ipynb`.
- `filter` -> `Train-Test split` -> `MultiLabelClassification<Level>` per hierarchy level (base, Computer, ComputerNetworking, ComputerSecurity, Hardware, OperatingSystems, Software, SMOTE variant) -> `SystemLabelAssignment`.
- Alternative one-step branch: `MLC_Notebook`, `MLCGeneral`, `MLCCS`, built on the `MLC` package.

## Imports
- `filter`, `Train-Test split`: `notebooks.utils`; `MLC_Notebook`: `MLC` and `notebooks.utils`; `MLCGeneral`, `MLCCS`: `MLC` only.
- `MultiLabelClassification*` and `SystemLabelAssignment`: root `utils` and `preprocess_functions` (SMOTE variant: `utils` only).

## Environment variables
- Only `filter`, `Train-Test split` and `MLC_Notebook` read env vars (`load_dotenv()`, then `os.chdir(ROOT)`); values are in the local `.env` (see root `AGENTS.md`). The other notebooks use hard-coded or relative paths.
- Names seen: `ROOT`, `RANDOM_STATE`, `TEST_SIZE`, `TWEETS_CSV`, `FILTER_TRAIN_CSV`, `FILTER_TEST_CSV`, `TEMP_CATEGORIES_TRAIN_CSV`, `TEMP_CATEGORIES_TEST_CSV`, `ONE_STEP_CATEGORIES_TRAIN_CSV`, `ONE_STEP_CATEGORIES_TEST_CSV`, `COTOV_CSV`, `CV`, `FILTER_MODELS`, `FILTERS_PERFORMANCES_CSV`, `OPTIMAL_FILTER_PICKLE`, `MODELS_DIR`, `MAPPINGS_DIR`, embedder paths (`SBERT_PATH`, `ATTACK_BERT_PATH`, `DISTILROBERTA_PATH`, `GLOVE_6B_{50,100,200,300}D_PATH`).

## Recorded Python versions (`language_info.version`)
- 3.10.14: `MultiLabelClassification*`; 3.11.13: `filter`, `MLC_Notebook`; 3.13.2: `MLCCS`, `MLCGeneral`, `Train-Test split`; 2.7.6: `SystemLabelAssignment` (likely stale metadata).

## Device selection
- `filter` and `MultiLabelClassification*` (except SMOTE) pass `device=select_device()` (root `device.py`, env `COMPUTE_DEVICE`: auto, cpu or cuda) to every active `XGBClassifier`.

## Legacy notebook
- `MultiLabelClassificationSMOTE.ipynb` is legacy and does not run: it imports `MultiLabelClassifier` and SMOTE helpers from `utils`, which were deliberately removed in commit 16e936a. They are recoverable from `16e936a^`, which also needed `bayes_opt`.

## Editing
- Edit source cells only; do not rewrite outputs or metadata unless asked.
- The Cotov (NistPosts) dataset is unrecoverable; its sections in `filter.ipynb` and `SystemLabelAssignment.ipynb` print a `BLOCKED:` line and are skipped when the file is missing (the notebook runner reports them as blocked).
