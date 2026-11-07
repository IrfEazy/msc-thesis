# Kaggle run log

Full runs of the Kaggle notebooks (owner `irfancela`; T4 GPU for the 2026-10 runs, CPU kernels since the rebuild). Outputs are in `runs/kaggle/<slug>/` (gitignored); durations and metrics are taken from the fetched kernel logs. Add an entry for each new full run.

## 2026-11-03 to 2026-11-07: Two-Step categorisation (CPU kernels)
Code 914067e (notebook and config; 60f32a7 changed only `scripts/` and the computer security transformers slug, so the checkpoint identity is the same). Durations from the kernel logs; `done/total` = candidate-fold checkpoints.

| Level | Kernel | Status | Duration | Checkpoints |
|---|---|---|---|---|
| general | `msc-thesis-cat-general-train-glove{50,100,200,300}` | COMPLETE | 0.9 / 1.7 / 2.3 / 3.9 h | 195/195 each |
| general | `msc-thesis-cat-general-train-{distilroberta,sbert,attackbert}` | INCOMPLETE (budget) | 9.6 / 9.7 / 9.7 h | 180 / 179 / 177 of 195, 0 failed |
| general | `msc-thesis-cat-general-train2-{distilroberta,sbert,attackbert}` (2026-11-07, continuations, commit 9d3522c, same code) | COMPLETE | 0.6 / 0.8 / 0.9 h | +15 / +16 / +18, 195/195 each, 0 failed |
| general | `msc-thesis-cat-general-assemble` | ERROR (missing checkpoints, refused to publish) | 0.2 h | - |
| general | `msc-thesis-cat-general-assemble` v2 (2026-11-07, attaches the `train2-*` continuations) | COMPLETE (published, sidecar `complete`) | 1.2 h | 1365 |
| computer security | `-train-glove` / `-train-trans` / `-assemble` | COMPLETE | 1.5 / 4.7 / 0.2 h | 780 + 585 |
| hardware | `-train-all` / `-assemble` | COMPLETE | 2.3 / 0.1 h | 1365 |
| operating systems | `-train-all` / `-assemble` | COMPLETE | 0.7 / 0.1 h | 1365 |
| software | `-train-all` / `-assemble` | COMPLETE | 2.5 / 0.1 h | 1365 |

Results (test, all eligible rows, 95% group-bootstrap CIs; outputs in `runs/kaggle/<level assemble slug>/models/`):

| Level | Selected embedder | n | micro F1 | macro F1 |
|---|---|---|---|---|
| general | ATT&CK-BERT | 2220 (thesis-comparable 2216) | 0.666 [0.637, 0.695] | 0.574 [0.537, 0.610] |
| computer security | GloVe.6B.200D | 1163 | 0.882 [0.861, 0.901] | 0.800 [0.648, 0.871] |
| hardware | SBERT | 442 (thesis-comparable 424) | 0.812 [0.775, 0.846] | 0.689 [0.595, 0.748] |
| operating systems | DistilRoBERTa | 217 | 0.719 [0.629, 0.808] | 0.663 [0.549, 0.780] |
| software | DistilRoBERTa | 806 | 0.787 [0.746, 0.824] | 0.755 [0.706, 0.795] |

Problems on the way:
- The general transformer shards were estimated at about 8 h from one-fold local timings (`notebooks/rebuild.md`) but used the whole 10 h effective budget; HOMER[LR/DT/RF] and some RAkEL[RF] folds are missing. Continued by the `train2-*` shards on 2026-11-07 (each under 1 h: the missing folds were far cheaper than the budget stop suggested); the general assemble v2 then published (outputs fetched to `runs/kaggle/general-assemble-v2/msc-thesis-cat-general-assemble/`, kept apart from the failed v1 output in `runs/kaggle/msc-thesis-cat-general-assemble/`). General OOF composed macro F1 per embedder (selection scores, optimistic): GloVe 50/100/200/300 0.549/0.534/0.568/0.582, DistilRoBERTa 0.597, SBERT 0.598, ATT&CK-BERT 0.607.
- The general assemble was pushed because Kaggle reported its sources COMPLETE although their checkpoint state was INCOMPLETE (see `scripts/AGENTS.md`).
- Push rejections: 400 Bad Request for a 51-character title (slug shortened, `build` now checks), `Maximum batch CPU session count of 5 reached`, and API rate limiting misreported as "not COMPLETE" (fixed in 60f32a7).
- A lead push loop stopped at the first failed push and left the queue idle for about 8 h overnight; one fetch was partial (818 of 2731 files) and was repeated.
- Session artifacts (pipeline run, loop logs, timing scripts): `runs/sessions/2026-11-02-categorisation/`.

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
