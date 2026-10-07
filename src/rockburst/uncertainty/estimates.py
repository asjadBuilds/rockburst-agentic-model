"""Uncertainty: entropy, disagreement, stability (conformal in conformal.py)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.uncertainty.conformal import conformal_prediction_set


def predictive_entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def disagreement(probas: list[np.ndarray]) -> float:
    if not probas:
        return 0.0
    preds = [int(p.argmax()) for p in probas]
    _vals, counts = np.unique(preds, return_counts=True)
    maj = counts.max() / len(preds)
    return float(1.0 - maj)


def perturb_case(
    case: dict[str, Any],
    features: list[str],
    eps: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out = dict(case)
    for f in features:
        v = out.get(f)
        if v is None or not np.isfinite(float(v)):
            continue
        noise = rng.uniform(-eps, eps)
        out[f] = float(v) * (1.0 + noise)
    return out


def stability_from_preds(
    base_pred: int,
    preds: list[int],
    base_proba: np.ndarray,
    probas: list[np.ndarray],
) -> dict[str, Any]:
    flip_rate = float(np.mean([p != base_pred for p in preds])) if preds else 0.0
    if probas:
        deltas = [float(np.mean(np.abs(base_proba - p))) for p in probas]
        mean_delta = float(np.mean(deltas))
    else:
        mean_delta = 0.0
    r_stability = float(
        np.clip(1.0 - 0.7 * flip_rate - 0.3 * min(mean_delta * 2, 1.0), 0, 1)
    )
    label = "high" if r_stability >= 0.75 else "moderate" if r_stability >= 0.5 else "low"
    return {
        "flip_rate": flip_rate,
        "mean_proba_delta": mean_delta,
        "stability_score": r_stability,
        "prediction_stability": label,
    }


def uncertainty_bundle(
    proba: np.ndarray,
    ensemble_probas: list[np.ndarray] | None = None,
    *,
    alpha: float = 0.1,
    class_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    ens = ensemble_probas or []
    dis = disagreement(ens) if ens else 0.0
    ent = predictive_entropy(proba)
    r_unc = float(np.clip(1.0 - (0.6 * (ent / np.log(4)) + 0.4 * dis), 0, 1))
    pset = conformal_prediction_set(
        proba, alpha=alpha, class_thresholds=class_thresholds
    )
    return {
        "uncertainty_score": 1.0 - r_unc,
        "r_uncertainty": r_unc,
        "entropy": ent,
        "model_disagreement": dis,
        "prediction_set": pset,
        "probabilities": proba.tolist(),
        "predicted_grade": int(proba.argmax()),
    }
