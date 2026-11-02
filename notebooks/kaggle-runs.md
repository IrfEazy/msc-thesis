# Kaggle run log

Full runs of the Kaggle notebooks (owner `irfancela`, accelerator T4). Outputs are in `runs/kaggle/<slug>/` (gitignored); durations and metrics are taken from the fetched kernel logs. Add an entry for each new full run.

## 2026-11-01 to 2026-11-02: leakage-free rebuild (CPU kernels)
| Notebook | Kernel | Code | Status | Duration | Outputs |
|---|---|---|---|---|---|
| Embeddings | `msc-thesis-embeddings` | 7c9ae10 | COMPLETE | 3.3 h | `embeddings/` (14 artifacts: corpus 21,368 rows, cotov 22,347 rows, x 7 embedders) |
| filter | `msc-thesis-filter` v4 | 092d31d | COMPLETE | 7.1 h | `models/filter.pkl` + `.json` sidecar (status `complete`), `filters_performances.csv`, `checkpoints/` |

Results (filter):
- Selected RF + SBERT (untuned selection stage: grouped 5-fold CV balanced accuracy 79.43 +- 1.75; tuned 80.05 +- 1.97; both selection diagnostics); tuned 100 evaluations: `n_estimators` 340, `max_depth` 22, `min_samples_leaf` 3, `min_samples_split` 14, `max_features` sqrt.
- Test (4,271 rows, prevalence 0.52, 95% group-bootstrap CIs): balanced accuracy 78.74 [76.86, 80.81], ROC AUC 87.56 [85.93, 89.09]; untuned reference balanced accuracy 78.66 [76.65, 80.62] (tuning gain negligible).
- Cotov (22,347 rows, prevalence 0.73, row bootstrap): balanced accuracy 97.77 [97.57, 97.99], ROC AUC 99.80.
- A stale `filter_models.pkl` from the 2026-10-30 run was moved to `runs/kaggle/_stale-20261101/`.

## 2026-10-29 to 2026-10-31 (code at 197f5b5; superseded, invalid for the thesis: see `rebuild.md`)
| Notebook | Final version | Status | Duration | Outputs |
|---|---|---|---|---|
| filter | v3 | COMPLETE | 0.9 h | `models/filter.pkl`, `filter_models.pkl`, `filters_performances.csv` |
| MLCGeneral | v2 | COMPLETE | 6.8 h | `models/GENERAL.pkl`, `models/general/`, `models/dictionaries/` (7.1 GB) |
| MLCCS | v2 | COMPLETE | 1.5 h | `models/COMPUTER.SECURITY.pkl`, `models/computer security/`, `models/dictionaries/` (1.3 GB) |
| MLC_Notebook | v2 | COMPLETE | 10.5 h | `models/{System,General,Computersecurity,Software,Hardware,Operatingsystems}.pickle`, `mappings/` (35 files; 9.5 GB total) |

Failures fixed on the way:
- filter v1: `KeyError 'min_samples_split'` (final model built as RandomForest from an XGBoost search) -> 422645b. v2: `FileNotFoundError` on the tuned filter (hard-coded `OVERWRITE = False`) -> 83b61f4.
- MLCGeneral/MLCCS: push rejected at the 2-session GPU limit -> 911bc34; v1 `LookupError` for NLTK `averaged_perceptron_tagger_eng` -> a5bb9ff (`setup_commands`).
- MLC_Notebook v1: ran 10.4 h, then `ValueError` (feature count) in `System.predict` because embedders were hard-coded -> 197f5b5.

Results:
- filter: tuned XGBoost test accuracy 78.95%, below some untuned combinations in `filters_performances.csv` (the comparison may not be like for like).
- MLC_Notebook per-module winners: General ATT&CK-BERT, Computer security DistilRoBERTa, Operating systems SBERT, Software GloVe.6B.300D, Hardware SBERT.
- MLC_Notebook `System` on 4078 threat tweets: accuracy 59.27%, Hamming loss 0.078, micro P/R/F1 0.73/0.58/0.65, macro 0.84/0.52/0.61. Precision is high but recall is low on several labels (network security 0.29, computer components 0.29, networking 0.32, hardware 0.37); `other` has recall 0.96 and precision 0.57.
