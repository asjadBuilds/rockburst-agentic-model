"""Uncertainty: entropy, disagreement, stability (conformal in conformal.py)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.uncertainty.conformal import conformal_prediction_set


def predictive_entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    n = max(len(p), len(q))
    pp, qq = np.zeros(n), np.zeros(n)
    pp[: len(p)] = p
    qq[: len(q)] = q
    return float(0.5 * np.abs(pp - qq).sum())


def mean_pairwise_tv(probas: list[np.ndarray]) -> float:
    """Continuous ensemble disagreement: mean pairwise total variation distance."""
    if len(probas) < 2:
        return 0.0
    tvs: list[float] = []
    for i in range(len(probas)):
        for j in range(i + 1, len(probas)):
            tvs.append(total_variation(probas[i], probas[j]))
    return float(np.mean(tvs)) if tvs else 0.0


def disagreement(probas: list[np.ndarray]) -> float:
    """Ensemble disagreement (Slice 4 / M5): mean pairwise TV; 0 if <2 members."""
    return mean_pairwise_tv(probas)


def perturb_case(
    case: dict[str, Any],
    features: list[str],
    eps: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Perturb primary measured inputs only; recompute SCF/B1/B2 (M3).

    Derived features in ``features`` are ignored. After noise on
    σθ, σc, σt, Wet, D, identities are refreshed so perturbations stay
    physically consistent.
    """
    from rockburst.physics.relations import (
        PRIMARY_FEATURES,
        compute_b1,
        compute_b2,
        compute_scf,
    )

    out = dict(case)
    primary = [f for f in features if f in PRIMARY_FEATURES]
    if not primary:
        primary = list(PRIMARY_FEATURES)
    for f in primary:
        v = out.get(f)
        if v is None or not np.isfinite(float(v)):
            continue
        noise = rng.uniform(-eps, eps)
        out[f] = float(v) * (1.0 + noise)
    st, sc, stens = out.get("sigma_theta"), out.get("sigma_c"), out.get("sigma_t")
    if (
        st is not None
        and sc is not None
        and stens is not None
        and np.isfinite(float(st))
        and np.isfinite(float(sc))
        and np.isfinite(float(stens))
    ):
        out["scf"] = compute_scf(float(st), float(sc))
        out["b1"] = compute_b1(float(sc), float(stens))
        out["b2"] = compute_b2(float(sc), float(stens))
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
