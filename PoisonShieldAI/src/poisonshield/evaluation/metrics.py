"""Separate victim-performance and poison-filtering metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           n_classes: int) -> dict:
    if not len(y_true):
        raise ValueError("No test records")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1), "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def filter_metrics(keep: np.ndarray, is_poison: np.ndarray) -> dict:
    """Poison origin is provided by the *evaluation harness*, never the defender."""
    if len(keep) != len(is_poison):
        raise ValueError("Mask / origin length mismatch")
    removed = ~keep
    tp = int(np.count_nonzero(removed & is_poison))
    fp = int(np.count_nonzero(removed & ~is_poison))
    fn = int(np.count_nonzero(keep & is_poison))
    tn = int(np.count_nonzero(keep & ~is_poison))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    return {
        "poison_detection_precision": precision,
        "poison_detection_recall": recall,
        "poison_detection_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
        "clean_retention": tn/(tn+fp) if tn+fp else 0.0,
        "poison_removal": tp/(tp+fn) if tp+fn else 0.0,
        "filter_false_positive_rate": fp/(fp+tn) if fp+tn else 0.0,
        "filtered_rows": int(np.count_nonzero(keep)),
        "removed_clean": fp, "removed_poison": tp, "retained_poison": fn,
    }
