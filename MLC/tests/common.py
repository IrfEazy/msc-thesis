"""Shared test utilities for the MLC classifiers."""

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression


def make_data(seed=0, n_samples=60, n_features=6, n_classes=4):
    """Small deterministic multi-label dataset: (X float, Y int 0/1)."""
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=2,
        random_state=seed,
    )
    return X.astype(float), Y.astype(int)


class RecordingEstimator(ClassifierMixin, BaseEstimator):
    """Wraps a binary estimator (default LogisticRegression) and records every
    fit call in the class-level list ``RecordingEstimator.calls`` as
    ``(X copy, y copy)``. The record is class-level so it survives sklearn
    ``clone``; call ``RecordingEstimator.reset()`` before a test."""

    calls = []

    def __init__(self, estimator=None):
        self.estimator = estimator

    @classmethod
    def reset(cls):
        cls.calls = []

    def fit(self, X, y):
        X = numpy.asarray(X)
        y = numpy.asarray(y)
        type(self).calls.append((X.copy(), y.copy()))
        base = self.estimator if self.estimator is not None else LogisticRegression(max_iter=1000)
        self.estimator_ = clone(base).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def check_contract(test, factory):
    """Assert the common MLC classifier contract.

    ``factory()`` returns a new unfitted classifier; ``test`` is a TestCase.
    Checks: no fitted attributes before fit; fit returns self; predict gives
    an (n, q) int 0/1 array; sklearn clone works; refit equals a fresh fit;
    a constant label column (all 0 and all 1) can be fitted.
    """
    X, Y = make_data(0)
    clf = factory()
    test.assertEqual([k for k in vars(clf) if k.endswith("_")], [])
    test.assertIs(clf.fit(X, Y), clf)
    P = clf.predict(X)
    test.assertEqual(P.shape, Y.shape)
    test.assertTrue(numpy.issubdtype(P.dtype, numpy.integer))
    test.assertTrue(numpy.isin(P, [0, 1]).all())
    clone(clf)

    X2, Y2 = make_data(1)
    clf.fit(X2, Y2)
    numpy.testing.assert_array_equal(clf.predict(X), factory().fit(X2, Y2).predict(X))

    for const in (0, 1):
        Yc = Y.copy()
        Yc[:, 0] = const
        Pc = factory().fit(X, Yc).predict(X)
        test.assertEqual(Pc.shape, Yc.shape)
        test.assertTrue((Pc[:, 0] == const).all())
