"""Ordinal classification helpers (Slice 6 / M8).

Frank–Hall binary decomposition and median-of-distribution decoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


def median_of_distribution(proba: np.ndarray) -> np.ndarray:
    """Decode grade as the median of the predictive categorical distribution.

    For each row, find the smallest grade ``g`` with cumulative mass ≥ 0.5.
    """
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    cdf = np.cumsum(p, axis=1)
    return np.argmax(cdf >= 0.5, axis=1).astype(int)


def argmax_grade(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        return np.array([int(np.argmax(p))], dtype=int)
    return np.argmax(p, axis=1).astype(int)


class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Frank–Hall ordinal reduction: K−1 binary classifiers for thresholds 0|1..K-1|K.

    Probability of class k is reconstructed from threshold probabilities
    P(Y>k). Base estimator should support ``predict_proba``.
    """

    def __init__(self, base_estimator: Any, n_classes: int = 4):
        self.base_estimator = base_estimator
        self.n_classes = int(n_classes)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.arange(self.n_classes)
        self.estimators_: list[Any] = []
        for thr in range(self.n_classes - 1):
            # Binary: 1 if y > thr
            y_bin = (y > thr).astype(int)
            est = clone(self.base_estimator)
            # Degenerate fold: all one class
            if len(np.unique(y_bin)) < 2:
                self.estimators_.append(("constant", float(y_bin.mean())))
            else:
                est.fit(X, y_bin)
                self.estimators_.append(("model", est))
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        # p_gt[k] = P(Y > k)
        p_gt = np.zeros((n, self.n_classes - 1), dtype=float)
        for k, (kind, obj) in enumerate(self.estimators_):
            if kind == "constant":
                p_gt[:, k] = float(obj)
            else:
                proba = obj.predict_proba(X)
                # column for class 1
                if proba.shape[1] == 1:
                    p_gt[:, k] = proba[:, 0]
                else:
                    # classes may be [0] only or [0,1]
                    classes = list(getattr(obj, "classes_", [0, 1]))
                    if 1 in classes:
                        p_gt[:, k] = proba[:, classes.index(1)]
                    else:
                        p_gt[:, k] = 0.0
        # Enforce monotone nonincreasing P(Y>k)
        for k in range(1, self.n_classes - 1):
            p_gt[:, k] = np.minimum(p_gt[:, k], p_gt[:, k - 1])
        # P(Y=0)=1-P(Y>0); P(Y=k)=P(Y>k-1)-P(Y>k); P(Y=K-1)=P(Y>K-2)
        out = np.zeros((n, self.n_classes), dtype=float)
        out[:, 0] = 1.0 - p_gt[:, 0]
        for k in range(1, self.n_classes - 1):
            out[:, k] = p_gt[:, k - 1] - p_gt[:, k]
        out[:, -1] = p_gt[:, -1]
        out = np.clip(out, 1e-12, None)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def predict(self, X) -> np.ndarray:
        return median_of_distribution(self.predict_proba(X))
