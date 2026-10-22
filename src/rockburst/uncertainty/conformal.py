"""Split / Mondrian / CV+ conformal calibration (group-aware)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative split-conformal quantile: level ⌈(n+1)(1−α)⌉ / n."""
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = len(s)
    if n == 0:
        return 1.0
    # Index is 1-based rank k = ceil((n+1)(1-alpha)), then take s[k-1]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


@dataclass
class ConformalState:
    """Fitted nonconformity thresholds (Mondrian, marginal, or CV+)."""

    alpha: float
    class_thresholds: np.ndarray  # shape (n_classes,)
    method: str = "marginal_split"
    n_calib: int = 0
    n_fit: int = 0
    class_calib_counts: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "class_thresholds": self.class_thresholds.tolist(),
            "method": self.method,
            "n_calib": self.n_calib,
            "n_fit": self.n_fit,
            "class_calib_counts": list(self.class_calib_counts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConformalState":
        return cls(
            alpha=float(d["alpha"]),
            class_thresholds=np.asarray(d["class_thresholds"], dtype=float),
            method=str(d.get("method", "marginal_split")),
            n_calib=int(d.get("n_calib", 0)),
            n_fit=int(d.get("n_fit", 0)),
            class_calib_counts=[int(x) for x in d.get("class_calib_counts", [])],
        )


def _nonconformity(y: np.ndarray, P: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] < n_classes:
        pad = np.zeros((len(P), n_classes))
        pad[:, : P.shape[1]] = P
        P = pad
    return 1.0 - P[np.arange(len(y)), np.clip(y, 0, n_classes - 1)]


def fit_marginal_thresholds(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    *,
    alpha: float = 0.1,
    n_classes: int = 4,
) -> ConformalState:
    """Single pooled threshold shared by all classes (marginal split conformal)."""
    y = np.asarray(y_calib, dtype=int)
    scores = _nonconformity(y, proba_calib, n_classes)
    q = finite_sample_quantile(scores, alpha)
    counts = [int((y == c).sum()) for c in range(n_classes)]
    return ConformalState(
        alpha=alpha,
        class_thresholds=np.full(n_classes, q, dtype=float),
        method="marginal_split",
        n_calib=int(len(y)),
        class_calib_counts=counts,
    )


def fit_mondrian_thresholds(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    *,
    alpha: float = 0.1,
    n_classes: int = 4,
    min_class_count: int = 5,
) -> ConformalState:
    """Per-class quantiles of ``1 - p_y``; pooled fallback when class is sparse."""
    y = np.asarray(y_calib, dtype=int)
    scores = _nonconformity(y, proba_calib, n_classes)
    pooled_q = finite_sample_quantile(scores, alpha)
    thresholds = np.full(n_classes, pooled_q, dtype=float)
    counts: list[int] = []
    for c in range(n_classes):
        mask = y == c
        counts.append(int(mask.sum()))
        if counts[-1] >= min_class_count:
            thresholds[c] = finite_sample_quantile(scores[mask], alpha)
    return ConformalState(
        alpha=alpha,
        class_thresholds=thresholds,
        method="mondrian_split",
        n_calib=int(len(y)),
        class_calib_counts=counts,
    )


def fit_cv_plus_thresholds(
    y_scores: np.ndarray,
    *,
    alpha: float = 0.1,
    n_classes: int = 4,
) -> ConformalState:
    """CV+ style: thresholds from pooled OOF nonconformity scores (grouped folds)."""
    scores = np.asarray(y_scores, dtype=float).ravel()
    q = finite_sample_quantile(scores, alpha)
    return ConformalState(
        alpha=alpha,
        class_thresholds=np.full(n_classes, q, dtype=float),
        method="cv_plus_grouped",
        n_calib=int(len(scores)),
        class_calib_counts=[],
    )


def make_contiguous_ordinal_set(grades: list[int]) -> list[int]:
    """Fill gaps so ordinal prediction sets are contiguous (Slice 6 / M11)."""
    if not grades:
        return []
    lo, hi = int(min(grades)), int(max(grades))
    return list(range(lo, hi + 1))


def conformal_prediction_set(
    proba: np.ndarray,
    *,
    alpha: float = 0.1,
    class_thresholds: np.ndarray | None = None,
    contiguous: bool = True,
) -> list[int]:
    """Prediction set from fitted thresholds, else APS fallback.

    When ``contiguous`` (default), fill holes so sets are ordinal intervals —
    required for ordered rockburst grades (M11).
    """
    p = np.asarray(proba, dtype=float).ravel()
    if class_thresholds is not None:
        thr = np.asarray(class_thresholds, dtype=float).ravel()
        if len(thr) < len(p):
            thr = np.pad(
                thr,
                (0, len(p) - len(thr)),
                constant_values=thr[-1] if len(thr) else 1.0,
            )
        chosen = [int(c) for c in range(len(p)) if (1.0 - p[c]) <= thr[c]]
    else:
        order = np.argsort(-p)
        s = 0.0
        chosen = []
        for idx in order:
            chosen.append(int(idx))
            s += p[idx]
            if s >= 1.0 - alpha:
                break
        chosen = sorted(chosen)
    if contiguous:
        return make_contiguous_ordinal_set(chosen)
    return sorted(chosen)


def coverage_and_size(
    y_true: np.ndarray,
    sets: list[list[int]],
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    covered = np.array([int(y[i]) in sets[i] for i in range(len(y))], dtype=float)
    sizes = np.array([len(s) for s in sets], dtype=float)
    per_class: dict[str, Any] = {}
    for c in sorted(set(y.tolist())):
        m = y == c
        per_class[str(c)] = {
            "coverage": float(covered[m].mean()) if m.any() else None,
            "mean_size": float(sizes[m].mean()) if m.any() else None,
            "n": int(m.sum()),
        }
    return {
        "coverage": float(covered.mean()) if len(covered) else None,
        "mean_size": float(sizes.mean()) if len(sizes) else None,
        "per_class": per_class,
        "n": int(len(y)),
    }


def compare_conformal_methods(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    *,
    alpha: float = 0.1,
    cv_plus_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score Mondrian vs marginal (and optional CV+) on the same test fold."""
    methods: dict[str, ConformalState] = {
        "mondrian_split": fit_mondrian_thresholds(y_calib, proba_calib, alpha=alpha),
        "marginal_split": fit_marginal_thresholds(y_calib, proba_calib, alpha=alpha),
    }
    if cv_plus_scores is not None and len(cv_plus_scores):
        methods["cv_plus_grouped"] = fit_cv_plus_thresholds(cv_plus_scores, alpha=alpha)

    out: dict[str, Any] = {"class_calib_counts": methods["mondrian_split"].class_calib_counts}
    for name, conf in methods.items():
        sets = [
            conformal_prediction_set(
                proba_test[i], alpha=alpha, class_thresholds=conf.class_thresholds
            )
            for i in range(len(y_test))
        ]
        cov = coverage_and_size(y_test, sets)
        out[name] = {
            **cov,
            "thresholds": conf.class_thresholds.tolist(),
            "n_calib": conf.n_calib,
        }
    return out


def pick_defensible_method(
    comparison: dict[str, Any],
    *,
    alpha: float = 0.1,
    prefer_mondrian_grade3: bool = True,
    min_grade3_calib: int = 5,
) -> str:
    """Select conformal method for production (Slice 6 / M11).

    Prefer Mondrian when grade-3 (strong) has enough calibration mass so
    class-conditional coverage is meaningful; otherwise fall back to
    coverage-closest-to-nominal with smaller mean set size as tie-break.
    Resolves prior contradiction of hard-coding marginal while documenting Mondrian.
    """
    counts = comparison.get("class_calib_counts") or []
    mondrian = comparison.get("mondrian_split") or {}
    grade3_n = int(counts[3]) if len(counts) > 3 else 0
    if prefer_mondrian_grade3 and mondrian.get("coverage") is not None:
        per = (mondrian.get("per_class") or {}).get("3") or {}
        if grade3_n >= min_grade3_calib and per.get("coverage") is not None:
            return "mondrian_split"

    nominal = 1.0 - alpha
    candidates = []
    for name in ("mondrian_split", "marginal_split", "cv_plus_grouped"):
        block = comparison.get(name)
        if not block or block.get("coverage") is None:
            continue
        # Slight preference for Mondrian when coverages are equally close
        mondrian_tie = 0 if name == "mondrian_split" else 1
        candidates.append(
            (
                abs(float(block["coverage"]) - nominal),
                float(block.get("mean_size") or 99),
                mondrian_tie,
                name,
            )
        )
    if not candidates:
        return "mondrian_split"
    candidates.sort()
    return candidates[0][3]
