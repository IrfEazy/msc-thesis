from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    roc_auc_score,
)
from .preconditions import check_binary_matrices, check_same_columns, check_same_rows


@check_same_rows("Y", "Y_pred")
@check_binary_matrices("Y", "Y_pred")
@check_same_columns("Y", "Y_pred")
def assess(Y: ArrayLike, Y_pred: ArrayLike) -> dict[str, float]:
    """
    Evaluate the model on the given data.

    Parameters
    ----------
    Y : ArrayLike of shape (n_samples, n_labels)
        The input features.
    Y_pred : ArrayLike of shape (n_samples, n_labels)
        The true binary label matrix.

    Returns
    -------
    metrics : dict[str, float]
        Dictionary containing accuracy, micro F1 score, and hamming loss.
    """
    accuracy = accuracy_score(Y, Y_pred)

    auc_score_micro = roc_auc_score(Y, Y_pred, average="micro")
    auc_score_macro = roc_auc_score(Y, Y_pred, average="macro")
    auc_score_weighted = roc_auc_score(Y, Y_pred, average="weighted")
    auc_score_samples = roc_auc_score(Y, Y_pred, average="samples")
    auc_per_label = roc_auc_score(Y, Y_pred, average=None)

    report = classification_report(Y, Y_pred, output_dict=True, zero_division=0.0)
    report["micro avg"]["auc"] = auc_score_micro
    report["macro avg"]["auc"] = auc_score_macro
    report["weighted avg"]["auc"] = auc_score_weighted
    report["samples avg"]["auc"] = auc_score_samples

    n_classes = Y.shape[1]
    class_names = [f"{i}" for i in range(n_classes)]
    for i, target in enumerate(class_names):
        if target in report:
            report[target]["auc"] = auc_per_label[i]
        else:
            # In case labels are not printed per class, you can store them separately
            report[target] = {"auc": auc_per_label[i]}

    hamming = hamming_loss(Y, Y_pred)
    return {"accuracy": accuracy, "hamming_loss": hamming, "report": report}
