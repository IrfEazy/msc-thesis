# Kaggle run log

Full runs of the Kaggle notebooks (owner `irfancela`, accelerator T4). Outputs are in `runs/kaggle/<slug>/` (gitignored); durations and metrics are taken from the fetched kernel logs. Add an entry for each new full run.

## 2026-10-29 to 2026-10-31 (code at 6ce51e1)
| Notebook | Final version | Status | Duration | Outputs |
|---|---|---|---|---|
| filter | v3 | COMPLETE | 0.9 h | `models/filter.pkl`, `filter_models.pkl`, `filters_performances.csv` |
| MLCGeneral | v2 | COMPLETE | 6.8 h | `models/GENERAL.pkl`, `models/general/`, `models/dictionaries/` (7.1 GB) |
| MLCCS | v2 | COMPLETE | 1.5 h | `models/COMPUTER.SECURITY.pkl`, `models/computer security/`, `models/dictionaries/` (1.3 GB) |
| MLC_Notebook | v2 | COMPLETE | 10.5 h | `models/{System,General,Computersecurity,Software,Hardware,Operatingsystems}.pickle`, `mappings/` (35 files; 9.5 GB total) |

Failures fixed on the way:
- filter v1: `KeyError 'min_samples_split'` (final model built as RandomForest from an XGBoost search) -> 28221d3. v2: `FileNotFoundError` on the tuned filter (hard-coded `OVERWRITE = False`) -> ec0f3ed.
- MLCGeneral/MLCCS: push rejected at the 2-session GPU limit -> 5ccfa20; v1 `LookupError` for NLTK `averaged_perceptron_tagger_eng` -> c88b939 (`setup_commands`).
- MLC_Notebook v1: ran 10.4 h, then `ValueError` (feature count) in `System.predict` because embedders were hard-coded -> 6ce51e1.

Results:
- filter: tuned XGBoost test accuracy 78.95%, below some untuned combinations in `filters_performances.csv` (the comparison may not be like for like).
- MLC_Notebook per-module winners: General ATT&CK-BERT, Computer security DistilRoBERTa, Operating systems SBERT, Software GloVe.6B.300D, Hardware SBERT.
- MLC_Notebook `System` on 4078 threat tweets: accuracy 59.27%, Hamming loss 0.078, micro P/R/F1 0.73/0.58/0.65, macro 0.84/0.52/0.61. Precision is high but recall is low on several labels (network security 0.29, computer components 0.29, networking 0.32, hardware 0.37); `other` has recall 0.96 and precision 0.57.
