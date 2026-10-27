"""Post-hoc calibration diagnostics for base multiclass probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to already-normalized probabilities."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    logits = np.log(p) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def _nll(proba: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Fit one temperature on held-out/OOF predictions only."""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) < 5:
        return 1.0
    result = minimize_scalar(
        lambda log_t: _nll(apply_temperature(p, np.exp(log_t)), y),
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    return float(np.exp(result.x)) if result.success else 1.0


def temperature_calibration_report(proba: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Report whether OOF-fitted temperature scaling improves multiclass NLL."""
    temperature = fit_temperature(proba, y)
    calibrated = apply_temperature(proba, temperature)
    before = _nll(np.asarray(proba, dtype=float), np.asarray(y, dtype=int))
    after = _nll(calibrated, np.asarray(y, dtype=int))
    return {
        "method": "temperature_scaling",
        "fit_protocol": "held_out_or_oof_only",
        "temperature": temperature,
        "nll_before": before,
        "nll_after": after,
        "improves_nll": bool(after <= before + 1e-12),
    }
