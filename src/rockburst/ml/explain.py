"""Local / fold-wise explainability hooks (RM9 / F15)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Domain prior: higher SCF / Wet / σθ should not push severity downward.
_SEVERITY_POSITIVE_FEATURES = ("scf", "wet", "sigma_theta")


def local_permutation_attributions(
    model: Any,
    X_row: pd.DataFrame | np.ndarray,
    *,
    feature_names: list[str],
    predicted_grade: int,
    n_repeats: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Per-case local permutation: drop in P(ŷ) when feature is shuffled.

    Positive value ⇒ feature supports the predicted grade (domain-aligned when
    the grade is high and the feature is a severity driver).
    """
    rng = np.random.default_rng(seed)
    if isinstance(X_row, pd.DataFrame):
        base = X_row.copy()
        names = list(feature_names) if feature_names else list(base.columns)
        x0 = base.iloc[[0]].copy()
    else:
        names = list(feature_names)
        x0 = pd.DataFrame(np.asarray(X_row, dtype=float).reshape(1, -1), columns=names)

    def _p_hat(frame: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(frame), dtype=float)
            if proba.ndim == 1:
                return float(proba[int(np.clip(predicted_grade, 0, len(proba) - 1))])
            g = int(np.clip(predicted_grade, 0, proba.shape[1] - 1))
            return float(proba[0, g])
        pred = int(model.predict(frame)[0])
        return 1.0 if pred == int(predicted_grade) else 0.0

    base_p = _p_hat(x0)
    out: dict[str, float] = {}
    for feat in names:
        if feat not in x0.columns:
            continue
        drops: list[float] = []
        for _ in range(n_repeats):
            xp = x0.copy()
            # Local noise: replace with a draw from N(x, |x|*0.15 + 0.01)
            val = float(xp[feat].iloc[0]) if np.isfinite(float(xp[feat].iloc[0])) else 0.0
            xp.iloc[0, xp.columns.get_loc(feat)] = val + float(
                rng.normal(0.0, abs(val) * 0.15 + 0.01)
            )
            drops.append(base_p - _p_hat(xp))
        out[feat] = float(np.mean(drops))
    return out


def check_implausible_explanation_directions(
    attributions: dict[str, float],
    predicted_grade: int,
    *,
    min_abs: float = 0.02,
) -> list[str]:
    """Flag attributions that contradict severity-increasing domain priors (RM9).

    If the model predicts a severe grade (≥2) but SCF/Wet/σθ have a clearly
    negative local attribution (feature supports lower grade), fire a signal.
    Conversely, grade 0 with large positive severity-driver attribution.
    """
    flags: list[str] = []
    g = int(predicted_grade)
    for feat in _SEVERITY_POSITIVE_FEATURES:
        if feat not in attributions:
            continue
        a = float(attributions[feat])
        if abs(a) < min_abs:
            continue
        if g >= 2 and a < -min_abs:
            flags.append(
                f"implausible_explanation:{feat}:neg_for_grade_{g}={a:.4f}"
            )
        if g == 0 and a > min_abs:
            flags.append(
                f"implausible_explanation:{feat}:pos_for_grade_0={a:.4f}"
            )
    return flags


def dependence_frame(
    df: pd.DataFrame,
    *,
    features: tuple[str, ...] = ("scf", "wet", "sigma_theta"),
    target: str = "rockburst_grade",
) -> dict[str, Any]:
    """Lightweight dependence data for F15 (Phase 6 plots)."""
    out: dict[str, Any] = {}
    for feat in features:
        if feat not in df.columns or target not in df.columns:
            continue
        mask = df[feat].notna() & df[target].notna()
        x = df.loc[mask, feat].astype(float).to_numpy()
        y = df.loc[mask, target].astype(int).to_numpy()
        if len(x) < 5:
            continue
        # Spearman-like rank correlation (no scipy dependency)
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        corr = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 else None
        out[feat] = {
            "n": int(len(x)),
            "spearman_approx": corr,
            "mean_by_grade": {
                str(g): float(x[y == g].mean()) if (y == g).any() else None
                for g in range(4)
            },
        }
    return out


def aggregate_fold_importances(
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine per-fold permutation importances → mean ± variability (RM9)."""
    if not fold_rows:
        return {
            "features": [],
            "importances_mean": [],
            "importances_std": [],
            "ranked": [],
            "n_folds": 0,
        }
    feat_names = list(fold_rows[0].get("features") or [])
    mat = []
    for row in fold_rows:
        means = row.get("importances_mean")
        if means is None and row.get("ranked"):
            # rebuild aligned vector
            lookup = {
                r["feature"]: float(r["importance_mean"]) for r in row["ranked"]
            }
            means = [lookup.get(f, 0.0) for f in feat_names]
        mat.append(np.asarray(means, dtype=float))
    M = np.vstack(mat)
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean)
    order = np.argsort(-mean)
    ranked = [
        {
            "feature": feat_names[i],
            "importance_mean": float(mean[i]),
            "importance_std": float(std[i]),
            "n_folds": int(M.shape[0]),
        }
        for i in order
    ]
    return {
        "features": feat_names,
        "importances_mean": mean.tolist(),
        "importances_std": std.tolist(),
        "ranked": ranked,
        "top_features": [r["feature"] for r in ranked[:5]],
        "n_folds": int(M.shape[0]),
        "fold_rows": fold_rows,
        "protocol": (
            "permutation_importance on each outer-fold held-out test set; "
            "pipeline fit on fold train/fit only; report mean±std across folds"
        ),
    }
