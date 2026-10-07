"""Evaluation metrics including ordinal and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
)


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    O = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    hist_true = np.bincount(y_true, minlength=n_classes).astype(float)
    hist_pred = np.bincount(y_pred, minlength=n_classes).astype(float)
    E = np.outer(hist_true, hist_pred)
    E = E / E.sum() * O.sum()
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / den)


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def severe_misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 2) -> float:
    return float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) >= threshold))


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 4) -> float:
    y = y_true.astype(int)
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "class_recall": recall_score(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "severe_misclassification_rate": severe_misclassification_rate(y_true, y_pred),
    }
    if probs is not None:
        out["brier"] = multiclass_brier(y_true, probs)
        out["ece"] = expected_calibration_error(y_true, probs)
    return out
