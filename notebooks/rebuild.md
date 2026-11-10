# Leakage-free rebuild: decisions, status, remaining work

Goal set by the human (2026-11-01): recreate the system exactly as the thesis methodology describes, with correct code, so that the best system and its statistics are valid. Every split, preprocessing, leakage and evaluation problem is fixed in code before anything runs. Keep this file current: update the status when a stage lands and drop items once done.

## Decisions fixed with the human (do not re-litigate)
- One shared grouped split for every stage (approved 2026-11-01): copies of the same tweet (same `group_key`, truncated copies merged by prefix) are always on the same side. Seed 42 and test size 0.2 were fixed before any result; `min_prefix` 40. Manifest sha256 `9f8b62852c14bccca294ea4c3bf1b5c554bb5272e2b066ba3d103d77041d76a0` (21,368 rows, 14,147 groups, train 17,097 / test 4,271; zero group overlap with Cotov).
- Selection, tuning and any threshold use only the train inner folds of the manifest. The test set is evaluated once per final model; report prevalence and 95% bootstrap CIs (resampling manifest groups on the test set; rows on Cotov, which has no groups).
- PSt `pruning_value` (p) is a hyperparameter to tune, not a constant. MBR uses the robust out-of-fold variant (`cv=5`).
- Staged notebooks run as chained, checkpointed CPU kernels. Re-running everything is fine; new outputs overwrite the old ones.
- All outputs produced before this rebuild (old filter, MLCGeneral, MLCCS, MLC_Notebook runs in `kaggle-runs.md`) are invalid for the thesis: independent per-notebook splits leaked duplicates (8,917 rows, 41.7%, belong to multi-row groups), the filter was tuned on test, Cotov was scored with the wrong model, preprocessing differed between stages.
- The original group-3 models and inputs (`AGENTS.md`, lost-inputs bullet) are not part of the rebuild; the rebuilt Two-Step and One-Step systems replace them.

## Lessons for staged Kaggle runs
- One-fold local timings underestimate the Kaggle runtime (Two-Step general transformer: about 1.6 h per fold locally, so about 8 h per shard, but the shards needed more than 9.7 h). Treat any shard estimated above about 60% of the 10 h effective budget (`CHECKPOINT_BUDGET_HOURS` 11 minus 60 min margin) as needing a split or a continuation shard (same env, previous shard in `kernels` and `CHECKPOINT_INPUTS`; the assemble lists both).

## Done
- Shared library (`textprep`, `taxonomy`, `splits`; commits 63f83b9, 38d758f) and the split notebook.
- Shared embedding artifacts (`embeddings.py`, `Embeddings.ipynb`; 7c9ae10): Kaggle kernel `msc-thesis-embeddings`, 14 production artifacts (corpus and cotov x 7 embedders).
- Relevance filter (`filtering.py`, `filter.ipynb`; 092d31d): Kaggle `msc-thesis-filter` v4 COMPLETE (results in `kaggle-runs.md`).
- Two-Step categorisation (`categorisation.ipynb`, one parametrised notebook for the five levels general, computer security, hardware, operating systems, software; replaces `MLCGeneral`/`MLCCS`): all five levels COMPLETE and published on Kaggle; general needed continuation shards `train2-{distilroberta,sbert,attackbert}` and assemble v2 (2026-11-07). Results in `kaggle-runs.md`.
- One-Step categorisation (level `one step` of `categorisation`; commits 18afe65, 16cfaa3; the Two-Step sub-level categorisers are reused, as their population and labels are identical; `MLC_Notebook.ipynb` deleted, history in git): COMPLETE and published on Kaggle 2026-11-10 (transformer shards needed continuations `train2-*` and `train3-*`). Implicit relevance is evaluated against gold relevance (One-Step `other` also covers relevant posts without a mapped category, so never derive relevance from `other` alone). Results in `kaggle-runs.md`.
- System (`SystemLabelAssignment`: `system.py`, `system_eval.py`, `system_pipeline.py`, `system.toml`; details `notebooks/AGENTS.md`): Kaggle `msc-thesis-system` COMPLETE 2026-11-10 (code e1f34f8), locked Two-Step vs One-Step evaluation with paired group-bootstrap CIs. Results in `kaggle-runs.md`.
- Kaggle data dataset (2026-11-10): no longer carries the legacy `filter_*`/`categories_*` files (older Kaggle versions keep them; local `data/tweets-dataset` untouched).
- Thesis rewritten from the valid runs only (methodology, results with selection diagnostics in Appendix A, introduction, conclusions, abstracts; commits 73d7879, 2a24b13, 4f7211a).

## Remaining, in order (each: work package, verify, review, commit, smoke, cost statement, full run)
