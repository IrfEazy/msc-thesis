# MLC package

Scikit-learn style multi-label classifiers. `setup.py` declares numpy, scikit-learn and scipy (scipy is imported by HOMER; tqdm and typing_extensions are not imported anywhere).

## Classifiers
- `__init__.__all__` exports 9: BR, CLR, CC, LP, PSt, CDN, MBR, RAkEL, HOMER (each `<Name>Classifier`, file `<Name>Classifier.py`).
- BR, CC, CLR, LP, RAkEL follow Zhang & Zhou 2014 (pseudocode); PSt, MBR/2BR, HOMER, CDN follow the descriptions in Bogatinovski et al. 2022 (PSt: Read et al. 2008; HOMER: Tsoumakas et al. 2008; CDN: Guo & Gu 2011).
- BR: one binary model per label; `t_criterion=False` (True gives an empty prediction its top-scored label).
- CC: `order` is `None` (natural), a permutation, or `"random"` (drawn from `random_state` at every fit); training uses true earlier labels, prediction the earlier binary predictions; outputs stay in original column order.
- CLR: pairwise models on rows where exactly one of the two labels is relevant, plus q label-vs-virtual-label models; predict j iff zeta*(j) > zeta*(V). A pair with no differing row has no model and casts no vote.
- LP: one class per distinct labelset; single labelset means no model is fitted (constant prediction).
- PSt: `pruning_value=2` (p), `max_reintroduced=1` (b). Equals LP when every labelset count >= p, which holds on the thesis data (labelsets below 20 are dropped). Raises `ValueError` if no row has a frequent labelset.
- MBR: stage 1 BR, stage 2 BR on `[X, F]`; `cv=5` uses out-of-fold F (cost `cv + 1` fits per label for stage 1), `cv=None` is the legacy in-sample variant; `meta_estimator` is the stage-2 estimator.
- RAkEL (RAkELo): `k=3` clipped to q (warning); `n_estimators=None` means m = min(2q, C(q,k)); distinct labelsets, every label covered first (uncovered labels are never predicted, with a warning, if m is too small); label j predicted when mu_j / tau_j > threshold.
- HOMER: balanced k-means over label columns (`n_clusters >= 2`; `n_iter=10` k-means iterations, a constructor parameter; `random_state`); top-down traversal descends while the meta-label score reaches `threshold`.
- CDN: one model per label on `[X, other true labels]`; Gibbs sampling where each sweep visits every label once in a fresh random order, `n_iterations=100`, `burn_in=50` (`0 <= burn_in < n_iterations` is validated), marginals averaged after burn-in.

## Interface convention (no abstract base class)
- Subclass `BaseEstimator` and `ClassifierMixin`; `base_estimator` is the first positional argument and defaults to `None`, which builds a fresh `LogisticRegression(max_iter=1000)` (`_base.resolve_estimator`); no default instance is shared.
- `fit(X, Y)` decorated with `@check_same_rows("X", "Y")` and `@check_binary_matrices("Y")`; returns self. Fitted attributes (trailing underscore, e.g. `estimators_`, `n_labels_`) exist only after `fit`.
- `predict` returns an int 0/1 matrix `(n_samples, n_labels)` in label order. `predict_proba` returns continuous scores of the same shape, not calibrated probabilities in general; CLR, RAkEL (vote fractions), CDN (Gibbs marginals) and HOMER (pruned top-down scores, 0 for pruned labels) are scores of that kind.
- `evaluate(X, Y)` returns `functions.assess(Y, predict(X), predict_proba(X))`, so AUC comes from the scores.
- Threshold: CDN and RAkEL `predict` take a `threshold` argument (CDN `evaluate` too); HOMER takes one in its constructor. BR, CC and MBR threshold per-label scores at 0.5; CLR predicts j when zeta*(j) > zeta*(V); LP predicts the most probable labelset (PSt via its internal LP).
- Pickles produced before the corrections to this package are incompatible (attributes and defaults changed); refit.

## Shared pieces
- `_base.py`: `resolve_estimator`; `ConstantBinaryClassifier` (stands in when a target has one class); `fit_binary` (clone and fit, or constant); `positive_score` (class-1 `predict_proba`, else `decision_function`, else `predict`); `binary_predict` (score > 0.5, else `predict == 1`).
- `preconditions.py`: decorators `check_binary_matrices`, `check_same_columns`, `check_same_rows`; each takes parameter names and raises `ValueError`.
- `functions.assess(Y, Y_pred, Y_score=None)` returns a dict with keys `accuracy` (subset accuracy), `hamming_loss`, `report`. `report` is `classification_report` output (`zero_division=0`) plus an `auc` entry per average and per label. AUC uses `Y_score` (falls back to the hard `Y_pred`, which is degenerate). A label with a single true class has AUC NaN; macro and weighted AUC average over the labels with a defined AUC.

## Tests
- `MLC/tests/` (unittest; `common.py` has `make_data`, `RecordingEstimator`, `check_contract`). Run from the repo root: `uv run python -m unittest discover -s MLC/tests -t .`. Use `OMP_NUM_THREADS=1` on a loaded machine.

## Adding a classifier
- Create `<Name>Classifier.py` following the interface above, using the `_base.py` helpers.
- Import and add it to `__init__.py` and `__all__`, and add `tests/test_<name>.py`.
- Keep dependencies within numpy, scikit-learn and scipy; declare any new one in `setup.py`.
