"""Shared helpers for the MLC classifiers.

Reused by the problem-transformation classifiers (BR, CC, CLR, MBR, LP, PSt,
RAkEL, HOMER, CDN) so that they share the same default estimator, the same
handling of single-class training targets and the same notion of a continuous
positive-class score.
"""

import numpy
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


def resolve_estimator(est):
    """Return ``est`` unchanged, or a fresh ``LogisticRegression(max_iter=1000)``
    when ``est`` is None. A new instance is built on every call, so no default
    estimator is ever shared between classifiers."""
    if est is None:
        return LogisticRegression(max_iter=1000)
    return est


class ConstantBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier that always predicts ``value`` (0 or 1).

    Stands in for a base estimator when a training target has a single class
    (many scikit-learn estimators refuse to fit on one). ``classes_`` is
    always ``[0, 1]`` and ``predict_proba`` returns ``[[1 - value, value]]``
    per sample, so it can be used wherever a fitted binary classifier is.

    Parameters
    ----------
    value : int, default 0
        The constant class, 0 or 1.
    """

    def __init__(self, value=0):
        self.value = value

    def fit(self, X=None, y=None):
        self.classes_ = numpy.array([0, 1])
        return self

    def predict(self, X):
        return numpy.full(numpy.asarray(X).shape[0], int(self.value), dtype=int)

    def predict_proba(self, X):
        n = numpy.asarray(X).shape[0]
        v = float(self.value)
        return numpy.tile(numpy.array([1.0 - v, v]), (n, 1))


def fit_binary(template, X: ArrayLike, y: ArrayLike):
    """Fit a clone of ``template`` on a binary target ``y``.

    Returns the fitted clone, or a fitted ``ConstantBinaryClassifier`` (with
    the single observed class) when ``y`` contains only one class. ``template``
    itself is never fitted.
    """
    y = numpy.asarray(y).ravel()
    classes = numpy.unique(y)
    if classes.size < 2:
        value = int(classes[0]) if classes.size else 0
        return ConstantBinaryClassifier(value).fit(X, y)
    return clone(template).fit(X, y)


def positive_score(est, X: ArrayLike) -> numpy.ndarray:
    """Continuous score for class 1 of a fitted binary estimator, shape (n,).

    Order of preference: the ``predict_proba`` column whose class is 1 (a
    constant 0/1 vector if the estimator never saw class 1); else
    ``decision_function``; else ``predict`` cast to float.
    """
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        classes = list(getattr(est, "classes_", [0, 1]))
        if 1 in classes:
            return numpy.asarray(proba)[:, classes.index(1)].astype(float)
        return numpy.zeros(numpy.asarray(proba).shape[0])
    if hasattr(est, "decision_function"):
        return numpy.asarray(est.decision_function(X), dtype=float).ravel()
    return numpy.asarray(est.predict(X), dtype=float)


def binary_predict(est, X: ArrayLike) -> numpy.ndarray:
    """0/1 prediction of a fitted binary estimator, shape (n,).

    Score > 0.5 (see ``positive_score``) when it has ``predict_proba``, else
    ``predict(X) == 1``.
    """
    if hasattr(est, "predict_proba"):
        return (positive_score(est, X) > 0.5).astype(int)
    return (numpy.asarray(est.predict(X)) == 1).astype(int)
