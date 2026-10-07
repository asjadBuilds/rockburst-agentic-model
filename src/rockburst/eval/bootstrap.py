"""Group-level bootstrap CIs for repeated grouped CV metrics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def grouped_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 42,
    agg: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float | None]:
    """Resample unique groups with replacement; CI on the aggregate of values.

    ``values`` and ``groups`` are aligned at the case (row) level.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if agg is None:
        agg = lambda x: float(np.mean(x))  # noqa: E731
    if len(values) == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "method": "grouped_bootstrap"}
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    point = agg(values)
    if len(uniq) < 2:
        return {
            "mean": point,
            "std": 0.0,
            "ci95_low": point,
            "ci95_high": point,
            "method": "grouped_bootstrap",
        }
    boots = []
    for _ in range(n_boot):
        sample_g = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sample_g)
        if not mask.any():
            continue
        boots.append(agg(values[mask]))
    arr = np.asarray(boots, dtype=float)
    return {
        "mean": point,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_low": float(np.percentile(arr, 2.5)) if len(arr) else point,
        "ci95_high": float(np.percentile(arr, 97.5)) if len(arr) else point,
        "method": "grouped_bootstrap",
        "n_boot": n_boot,
    }


def grouped_bootstrap_metric(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 500,
    seed: int = 42,
    min_n: int = 5,
) -> dict[str, float | None]:
    """Grouped bootstrap CI for a paired metric ``metric(a[mask], b[mask])``."""
    a = np.asarray(a)
    b = np.asarray(b)
    groups = np.asarray(groups)
    if len(a) == 0:
        return {
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "method": "grouped_bootstrap",
            "n_boot": 0,
        }
    try:
        point_raw = metric(a, b)
        point = float(point_raw) if point_raw is not None else float("nan")
    except (ValueError, ZeroDivisionError):
        point = float("nan")
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    if len(uniq) < 2:
        ok = point if np.isfinite(point) else None
        return {
            "mean": ok,
            "std": 0.0 if ok is not None else None,
            "ci95_low": ok,
            "ci95_high": ok,
            "method": "grouped_bootstrap",
            "n_boot": 0,
        }
    for _ in range(n_boot):
        sample_g = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sample_g)
        if int(mask.sum()) < min_n:
            continue
        try:
            v = metric(a[mask], b[mask])
        except (ValueError, ZeroDivisionError):
            continue
        if v is None:
            continue
        fv = float(v)
        if not np.isfinite(fv):
            continue
        boots.append(fv)
    arr = np.asarray(boots, dtype=float)
    ok = point if np.isfinite(point) else None
    return {
        "mean": ok,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else (0.0 if len(arr) else None),
        "ci95_low": float(np.percentile(arr, 2.5)) if len(arr) else ok,
        "ci95_high": float(np.percentile(arr, 97.5)) if len(arr) else ok,
        "method": "grouped_bootstrap",
        "n_boot": int(len(arr)),
    }


def fold_list_ci(values: list[float]) -> dict[str, Any]:
    """Legacy normal CI over fold means (too narrow; kept for comparison)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "method": "normal_fold"}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / max(np.sqrt(len(arr)), 1.0)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "method": "normal_fold",
    }
