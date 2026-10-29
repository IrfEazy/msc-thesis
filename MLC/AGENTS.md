# MLC package

Scikit-learn style multi-label classifiers. `setup.py` declares numpy, scikit-learn, tqdm; tqdm is unused in the code, and `CCClassifier` imports `typing_extensions` (undeclared).

## Classifiers
- `__init__.__all__` exports 9: BR, CLR, CC, LP, PSt, CDN, MBR, RAkEL, HOMER (each `<Name>Classifier`, file `<Name>Classifier.py`).

## Interface convention (no abstract base class)
- Subclass `BaseEstimator` and `ClassifierMixin`; `__init__(base_estimator=LogisticRegression(...), ...)`.
- `fit(X, Y)` decorated with `@check_same_rows("X", "Y")` and `@check_binary_matrices("Y")`; returns self.
- `predict`, `predict_proba`, and `evaluate(X, Y)` which returns `functions.assess(Y, predict(X))`.
- Threshold: CDN and RAkEL `predict` take a `threshold` argument (CDN `evaluate` too); HOMER takes one in its constructor.

## Shared pieces
- `preconditions.py`: decorators `check_binary_matrices`, `check_same_columns`, `check_same_rows`; each takes parameter names and raises `ValueError`.
- `functions.assess(Y, Y_pred)` returns a dict with keys `accuracy`, `hamming_loss`, `report`. `report` is `classification_report` output plus an `auc` entry per average and per label.

## Adding a classifier
- Create `<Name>Classifier.py` following the interface above.
- Import and add it to `__init__.py` and `__all__`.
- Keep dependencies within numpy and scikit-learn; declare any new one in `setup.py`.

## Known caveat
- Default `base_estimator` is a single `LogisticRegression` instance created at definition time and shared across instances. Pass an explicit estimator; do not mutate the default. Defaults differ: `max_iter=1000` in LP, PSt, MBR, RAkEL, HOMER only.
