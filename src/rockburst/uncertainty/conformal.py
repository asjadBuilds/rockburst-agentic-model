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


def ordinal_nonconformity(
    y: np.ndarray, P: np.ndarray, *, n_classes: int = 4
) -> np.ndarray:
    """Ordinal nonconformity: expected absolute grade error under P (RM8).

    Lower is more conforming. Encourages contiguous mass around the true grade.
    """
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] < n_classes:
        pad = np.zeros((len(P), n_classes))
        pad[:, : min(P.shape[1], n_classes)] = P[:, :n_classes]
        P = pad
    grades = np.arange(n_classes, dtype=float)
    # E[|G - y|] under predictive categorical
    return np.array(
        [float(np.dot(P[i], np.abs(grades - float(y[i])))) for i in range(len(y))],
        dtype=float,
    )


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


def fit_ordinal_thresholds(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    *,
    alpha: float = 0.1,
    n_classes: int = 4,
) -> ConformalState:
    """Split conformal with ordinal |y − median(P)| nonconformity (RM8).

    ``class_thresholds`` stores the integer radius ``q`` in every slot (hack for
    the shared ConformalState schema). Use ``ordinal_prediction_set`` at serve
    time; ``compare_conformal_methods`` routes ordinal_split through that helper.
    """
    y = np.asarray(y_calib, dtype=int)
    P = np.asarray(proba_calib, dtype=float)
    from rockburst.ml.ordinal import median_of_distribution

    med = median_of_distribution(P)
    scores = np.abs(y.astype(float) - med.astype(float))
    q = finite_sample_quantile(scores, alpha)
    counts = [int((y == c).sum()) for c in range(n_classes)]
    return ConformalState(
        alpha=alpha,
        class_thresholds=np.full(n_classes, q, dtype=float),
        method="ordinal_split",
        n_calib=int(len(y)),
        class_calib_counts=counts,
    )


def ordinal_prediction_set(
    proba: np.ndarray,
    *,
    radius: float,
    contiguous: bool = True,
) -> list[int]:
    """Contiguous grades within ``radius`` of the predictive median (RM8)."""
    from rockburst.ml.ordinal import median_of_distribution

    p = np.asarray(proba, dtype=float).ravel()
    m = int(median_of_distribution(p)[0])
    r = int(np.ceil(float(radius)))
    chosen = [g for g in range(len(p)) if abs(g - m) <= r]
    if contiguous:
        return make_contiguous_ordinal_set(chosen)
    return sorted(chosen)


def fit_class_conditional_alpha(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    *,
    alpha: float = 0.1,
    n_classes: int = 4,
    focus_class: int = 3,
    focus_alpha: float | None = None,
    min_class_count: int = 5,
) -> ConformalState:
    """Pooled conformal with a distinct class-specific threshold for grade 3."""
    focus_alpha = float(alpha if focus_alpha is None else focus_alpha)
    y = np.asarray(y_calib, dtype=int)
    scores = _nonconformity(y, proba_calib, n_classes)
    pooled_q = finite_sample_quantile(scores, alpha)
    thresholds = np.full(n_classes, pooled_q, dtype=float)
    counts: list[int] = []
    for c in range(n_classes):
        mask = y == c
        counts.append(int(mask.sum()))
        # This variant differs from Mondrian: non-focus classes deliberately
        # retain the pooled threshold; only grade 3 is class-conditioned.
        if c == focus_class and counts[-1] >= min_class_count:
            thresholds[c] = finite_sample_quantile(scores[mask], focus_alpha)
    return ConformalState(
        alpha=alpha,
        class_thresholds=thresholds,
        method="class_conditional_grade3",
        n_calib=int(len(y)),
        class_calib_counts=counts,
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


def set_size_distribution(
    y_true: np.ndarray,
    sets: list[list[int]],
    *,
    fold: int | None = None,
    n_classes: int = 4,
    large_set_size: int = 3,
) -> dict[str, Any]:
    """Set-size histogram overall / per class / optional fold tag (RM8 / F12)."""
    y = np.asarray(y_true, dtype=int)
    sizes = np.array([len(s) for s in sets], dtype=int)
    max_size = max(int(sizes.max()) if len(sizes) else 0, n_classes)

    def _hist(sz: np.ndarray) -> dict[str, int]:
        return {str(k): int((sz == k).sum()) for k in range(0, max_size + 1)}

    per_class: dict[str, Any] = {}
    for c in range(n_classes):
        m = y == c
        if not m.any():
            per_class[str(c)] = {"n": 0, "size_hist": {}, "mean_size": None}
            continue
        per_class[str(c)] = {
            "n": int(m.sum()),
            "size_hist": _hist(sizes[m]),
            "mean_size": float(sizes[m].mean()),
        }
    abstain_proxy = sizes >= int(large_set_size)
    return {
        "fold": fold,
        "n": int(len(y)),
        "size_hist": _hist(sizes),
        "mean_size": float(sizes.mean()) if len(sizes) else None,
        "per_class": per_class,
        "frac_size_ge_large": float(abstain_proxy.mean()) if len(sizes) else None,
        "large_set_size": int(large_set_size),
    }


def coverage_size_abstention_tradeoff(
    method_blocks: dict[str, dict[str, Any]],
    *,
    alpha: float = 0.1,
    large_set_size: int = 3,
) -> dict[str, Any]:
    """Summarise coverage–size–abstention trade-off across conformal methods (RM8)."""
    rows: list[dict[str, Any]] = []
    for name, block in method_blocks.items():
        if not isinstance(block, dict) or block.get("coverage") is None:
            continue
        mean_size = block.get("mean_size")
        frac_large = block.get("frac_size_ge_large")
        if frac_large is None and block.get("size_hist"):
            hist = block["size_hist"]
            n = sum(int(v) for v in hist.values()) or 1
            frac_large = sum(
                int(hist.get(str(k), 0)) for k in range(large_set_size, 20)
            ) / n
        rows.append(
            {
                "method": name,
                "coverage": block.get("coverage"),
                "mean_size": mean_size,
                "frac_size_ge_large": frac_large,
                "coverage_gap_vs_nominal": (
                    None
                    if block.get("coverage") is None
                    else float(block["coverage"]) - (1.0 - alpha)
                ),
                "per_class": block.get("per_class"),
            }
        )
    return {
        "alpha": alpha,
        "large_set_size": large_set_size,
        "methods": rows,
        "note": (
            "Trade-off after RM2: |set|≥large_set_size is not auto-severe by "
            "duplicate rule; abstention proxy = fraction of sets that wide."
        ),
    }


def compare_conformal_methods(
    y_calib: np.ndarray,
    proba_calib: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    *,
    alpha: float = 0.1,
    cv_plus_scores: np.ndarray | None = None,
    include_ordinal: bool = True,
    include_class_conditional: bool = True,
    large_set_size: int = 3,
) -> dict[str, Any]:
    """Score Mondrian vs marginal (+ optional CV+/ordinal/class-cond) on one fold."""
    methods: dict[str, ConformalState] = {
        "mondrian_split": fit_mondrian_thresholds(y_calib, proba_calib, alpha=alpha),
        "marginal_split": fit_marginal_thresholds(y_calib, proba_calib, alpha=alpha),
    }
    if cv_plus_scores is not None and len(cv_plus_scores):
        methods["cv_plus_grouped"] = fit_cv_plus_thresholds(cv_plus_scores, alpha=alpha)
    if include_ordinal:
        methods["ordinal_split"] = fit_ordinal_thresholds(
            y_calib, proba_calib, alpha=alpha
        )
    if include_class_conditional:
        methods["class_conditional_grade3"] = fit_class_conditional_alpha(
            y_calib, proba_calib, alpha=alpha, focus_class=3
        )

    out: dict[str, Any] = {"class_calib_counts": methods["mondrian_split"].class_calib_counts}
    for name, conf in methods.items():
        if name == "ordinal_split":
            radius = float(conf.class_thresholds[0]) if len(conf.class_thresholds) else 1.0
            sets = [
                ordinal_prediction_set(proba_test[i], radius=radius, contiguous=True)
                for i in range(len(y_test))
            ]
        else:
            sets = [
                conformal_prediction_set(
                    proba_test[i], alpha=alpha, class_thresholds=conf.class_thresholds
                )
                for i in range(len(y_test))
            ]
        cov = coverage_and_size(y_test, sets)
        dist = set_size_distribution(
            y_test, sets, large_set_size=large_set_size
        )
        out[name] = {
            **cov,
            "thresholds": conf.class_thresholds.tolist(),
            "n_calib": conf.n_calib,
            "size_hist": dist["size_hist"],
            "frac_size_ge_large": dist["frac_size_ge_large"],
            "set_size_distribution": dist,
        }
    out["tradeoff"] = coverage_size_abstention_tradeoff(
        {k: out[k] for k in methods}, alpha=alpha, large_set_size=large_set_size
    )
    return out


def pick_defensible_method(
    comparison: dict[str, Any],
    *,
    alpha: float = 0.1,
    prefer_mondrian_grade3: bool = True,
    min_grade3_calib: int = 5,
) -> str:
    """Choose the smallest-set method that meets overall and grade-3 coverage."""
    del prefer_mondrian_grade3  # retained as a compatibility argument
    counts = comparison.get("class_calib_counts") or []
    grade3_n = int(counts[3]) if len(counts) > 3 else 0
    nominal = 1.0 - alpha
    candidates = []
    fallbacks = []
    for name in (
        "class_conditional_grade3",
        "mondrian_split",
        "marginal_split",
        "cv_plus_grouped",
    ):
        block = comparison.get(name)
        if not block or block.get("coverage") is None:
            continue
        coverage = float(block["coverage"])
        grade3 = ((block.get("per_class") or {}).get("3") or {}).get("coverage")
        abstention = block.get("frac_size_ge_large")
        mean_size = float(block.get("mean_size") or 99)
        efficient = (
            float(abstention) if abstention is not None else 1.0,
            mean_size,
            abs(coverage - nominal),
            name,
        )
        fallbacks.append((abs(coverage - nominal), mean_size, name))
        overall_ok = coverage >= nominal - 0.02
        grade3_ok = (
            grade3_n < min_grade3_calib
            or grade3 is None
            or float(grade3) >= nominal - 0.03
        )
        if overall_ok and grade3_ok:
            candidates.append(efficient)
    if not candidates:
        return min(fallbacks)[2] if fallbacks else "mondrian_split"
    candidates.sort()
    return candidates[0][3]
