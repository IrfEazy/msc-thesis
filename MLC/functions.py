import numpy
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    roc_auc_score,
)
from .preconditions import check_binary_matrices, check_same_columns, check_same_rows


def _auc(y_true, y_score) -> float:
    """ROC AUC, NaN when y_true has a single class."""
    if numpy.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@check_same_rows("Y", "Y_pred")
@check_binary_matrices("Y", "Y_pred")
@check_same_columns("Y", "Y_pred")
def assess(
    Y: ArrayLike, Y_pred: ArrayLike, Y_score: ArrayLike | None = None
) -> dict[str, float]:
    """
    Evaluate multi-label predictions against the true labels.

    Parameters
    ----------
    Y : ArrayLike of shape (n_samples, n_labels)
        The true binary label matrix.
    Y_pred : ArrayLike of shape (n_samples, n_labels)
        The predicted binary label matrix.
    Y_score : ArrayLike of shape (n_samples, n_labels), optional
        Continuous scores for the positive class of each label. AUC is
        computed from them. When omitted, AUC is computed from the hard
        ``Y_pred`` instead, which is degenerate (a single operating point).

    Returns
    -------
    metrics : dict
        ``accuracy``: subset accuracy (a sample counts only if all its labels
        are correct); ``hamming_loss``: fraction of wrong label cells;
        ``report``: ``classification_report`` output dict (``zero_division=0``)
        with an extra ``auc`` entry for every label key ("0", "1", ...) and
        every average key ("micro avg", "macro avg", "weighted avg",
        "samples avg").

    Notes
    -----
    A label whose true column has a single class has AUC NaN. Macro and
    weighted AUC average only the labels with a defined AUC (NaN if none);
    weighted uses the label supports. Micro AUC uses all cells; samples AUC
    averages over rows containing both classes (NaN if none). Never raises
    for single-class columns.
    """
    Y = numpy.asarray(Y)
    Y_pred = numpy.asarray(Y_pred)
    if Y_score is None:
        S = Y_pred.astype(float)
    else:
        S = numpy.asarray(Y_score, dtype=float)
        if S.shape != Y.shape:
            raise ValueError("Y_score must have the same shape as Y.")

    n_labels = Y.shape[1]
    auc_labels = numpy.array([_auc(Y[:, i], S[:, i]) for i in range(n_labels)])
    ok = ~numpy.isnan(auc_labels)
    support = Y.sum(axis=0).astype(float)

    auc_micro = _auc(Y.ravel(), S.ravel())
    auc_macro = float(auc_labels[ok].mean()) if ok.any() else float("nan")
    auc_weighted = (
        float(numpy.average(auc_labels[ok], weights=support[ok]))
        if ok.any()
        else float("nan")
    )
    rows = [
        _auc(Y[r], S[r]) for r in range(Y.shape[0]) if numpy.unique(Y[r]).size == 2
    ]
    auc_samples = float(numpy.mean(rows)) if rows else float("nan")

    report = classification_report(Y, Y_pred, output_dict=True, zero_division=0.0)
    for key, value in (
        ("micro avg", auc_micro),
        ("macro avg", auc_macro),
        ("weighted avg", auc_weighted),
        ("samples avg", auc_samples),
    ):
        report.setdefault(key, {})["auc"] = value
    for i in range(n_labels):
        report.setdefault(str(i), {})["auc"] = float(auc_labels[i])

    return {
        "accuracy": accuracy_score(Y, Y_pred),
        "hamming_loss": hamming_loss(Y, Y_pred),
        "report": report,
    }
