from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(y_true, y_pred, num_classes: int) -> dict:
    """UA = unweighted accuracy (mean per-class recall), WA = plain accuracy."""
    labels = list(range(num_classes))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = np.isin(labels, np.unique(y_true))
    return {
        "ua": float(recalls[present].mean()),
        "wa": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_eval_outputs(outputs: List[dict], num_classes: int) -> dict:
    y_true = np.concatenate([o["labels"] for o in outputs])
    y_pred = np.concatenate([o["preds"] for o in outputs])
    return {"loss": float(np.mean([o["loss"] for o in outputs])), **compute_metrics(y_true, y_pred, num_classes)}
