"""Shared reliability feature path for fusion train and serve."""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.fusion.stacker import FEATURE_NAMES, pack_r_vector
from rockburst.physics.relations import FEATURE_COLS


def compute_r_model(proba: Any) -> float:
    """Single train/serve model-confidence: raw max class probability."""
    return float(np.asarray(proba, dtype=float).ravel().max())


def _is_observed(case: dict[str, Any], col: str) -> bool:
    v = case.get(col)
    if v is None:
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def missingness_counts(
    case: dict[str, Any] | None = None,
    *,
    observed_mask: dict[str, bool] | None = None,
) -> tuple[float, float]:
    """Return (n_missing, d_observed) for stacker features."""
    if observed_mask is not None:
        n_miss = float(sum(1 for c in FEATURE_COLS if not observed_mask.get(c, False)))
        d_obs = float(1.0 if observed_mask.get("d", False) else 0.0)
        return n_miss, d_obs
    case = case or {}
    n_miss = float(sum(1 for c in FEATURE_COLS if not _is_observed(case, c)))
    d_obs = float(1.0 if _is_observed(case, "d") else 0.0)
    return n_miss, d_obs


def compute_reliability_features(
    data_q: Any,
    phys: Any,
    pred: Any,
    unc: Any,
    case: dict[str, Any] | None = None,
) -> np.ndarray:
    """Single train/serve code path for the fusion stacker input vector."""
    obs = getattr(pred, "observed_mask", None) if pred is not None else None
    n_miss, d_obs = missingness_counts(case, observed_mask=obs)
    return pack_r_vector(
        float(data_q.data_reliability) if data_q is not None else 0.5,
        float(phys.physics_score) if phys is not None else 0.5,
        float(pred.r_model) if pred is not None else 0.5,
        float(unc.r_uncertainty) if unc is not None else 0.5,
        float(unc.stability_score) if unc is not None else 0.5,
        n_miss,
        d_obs,
    )


def reliability_feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def r_vector_dict(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float).ravel()
    return {n: float(r[i]) for i, n in enumerate(FEATURE_NAMES) if i < len(r)}
