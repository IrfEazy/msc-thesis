import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold

from ._base import fit_binary, positive_score, resolve_estimator
from .functions import assess
from .preconditions import check_same_rows, check_binary_matrices


def _binary_predict(est, X) -> numpy.ndarray:
    """0/1 prediction of a fitted binary estimator (score > 0.5 when it has
    ``predict_proba``, else its own ``predict``)."""
    if hasattr(est, "predict_proba"):
        return (positive_score(est, X) > 0.5).astype(int)
    return (numpy.asarray(est.predict(X)) == 1).astype(int)


class MBRClassifier(BaseEstimator, ClassifierMixin):
    """Meta Binary Relevance / 2BR (Bogatinovski et al. 2022).

    Stage 1 is BR on X. Stage 2 is BR on ``[X, F]`` where F are the stage-1
    positive-class scores. For training, F is out-of-fold (``cv`` folds, per
    label) so the meta features are not fitted in-sample; stage 1 is then
    refit on all data for prediction.

    Parameters
    ----------
    base_estimator : estimator, default None
        Stage-1 binary classifier. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    meta_estimator : estimator, default None
        Stage-2 binary classifier. None means a fresh
        ``LogisticRegression(max_iter=1000)``.
    cv : int or None, default 5
        Number of folds for the out-of-fold stage-1 scores (shuffled
        StratifiedKFold when the label has at least ``cv`` samples in each
        class, else KFold). None uses legacy in-sample stage-1 scores.
    random_state : int or None, default None
        Seed of the fold shuffling.

    Attributes
    ----------
    first_stage_estimators_ : list of q classifiers fitted on all data
    meta_estimators_ : list of q classifiers fitted on ``[X, F]``
    n_labels_ : int

    Notes
    -----
    ``predict_proba`` returns the stage-2 positive-class scores.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        meta_estimator: ClassifierMixin | None = None,
        cv: int | None = 5,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.random_state = random_state

    def _oof_scores(self, template, X, y):
        n_splits = self.cv
        counts = numpy.bincount(y.astype(int), minlength=2)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits, shuffle=True, random_state=self.random_state)
        scores = numpy.zeros(X.shape[0])
        for train, val in splitter.split(X, y):
            est = fit_binary(template, X[train], y[train])
            scores[val] = positive_score(est, X[val])
        return scores

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def fit(self, X: ArrayLike, Y: ArrayLike) -> "MBRClassifier":
        """
        Fit stage 1 (BR on X) and stage 2 (BR on X plus stage-1 scores).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
        Y : ArrayLike of shape (n_samples, n_labels)

        Returns
        -------
        self : "MBRClassifier"
        """
        X = numpy.asarray(X)
        Y = numpy.asarray(Y)
        q = Y.shape[1]
        base = resolve_estimator(self.base_estimator)
        meta = resolve_estimator(self.meta_estimator)

        first = [fit_binary(base, X, Y[:, i]) for i in range(q)]
        if self.cv is None:
            F = numpy.column_stack([positive_score(e, X) for e in first])
        else:
            F = numpy.column_stack([self._oof_scores(base, X, Y[:, i]) for i in range(q)])
        X_meta = numpy.hstack((X, F))
        self.first_stage_estimators_ = first
        self.meta_estimators_ = [fit_binary(meta, X_meta, Y[:, i]) for i in range(q)]
        self.n_labels_ = q
        return self

    def _meta_features(self, X):
        X = numpy.asarray(X)
        F = numpy.column_stack([positive_score(e, X) for e in self.first_stage_estimators_])
        return numpy.hstack((X, F))

    def predict_proba(self, X: ArrayLike) -> numpy.ndarray:
        """
        Stage-2 positive-class score of each label, shape (n_samples, n_labels).
        """
        X_meta = self._meta_features(X)
        return numpy.column_stack([positive_score(e, X_meta) for e in self.meta_estimators_])

    def predict(self, X: ArrayLike) -> numpy.ndarray:
        """
        Predict labels with the stage-2 models (scores above 0.5 for
        estimators with ``predict_proba``, else their own ``predict``).

        Returns
        -------
        Y_pred : ndarray of int, shape (n_samples, n_labels)
        """
        X_meta = self._meta_features(X)
        return numpy.column_stack([_binary_predict(e, X_meta) for e in self.meta_estimators_])

    @check_same_rows("X", "Y")
    @check_binary_matrices("Y")
    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> dict[str, float]:
        """
        Evaluate on the given data; see ``functions.assess`` (AUC uses
        ``predict_proba`` scores).
        """
        return assess(Y, self.predict(X), self.predict_proba(X))
