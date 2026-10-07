"""Shared reliability feature path for fusion train and serve."""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.fusion.stacker import FEATURE_NAMES, pack_r_vector


def compute_reliability_features(
    data_q: Any,
    phys: Any,
    pred: Any,
    unc: Any,
) -> np.ndarray:
    """Single train/serve code path for the fusion stacker input vector."""
    return pack_r_vector(
        float(data_q.data_reliability) if data_q is not None else 0.5,
        float(phys.physics_score) if phys is not None else 0.5,
        float(pred.r_model) if pred is not None else 0.5,
        float(unc.r_uncertainty) if unc is not None else 0.5,
        float(unc.stability_score) if unc is not None else 0.5,
    )


def reliability_feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def r_vector_dict(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float).ravel()
    return {n: float(r[i]) for i, n in enumerate(FEATURE_NAMES)}
