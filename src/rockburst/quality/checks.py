"""Data quality / anomaly scoring (fold-aware when reference is train-only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.near_dup import (
    DEFAULT_REL_TOL,
    RAW_DUP_COLS,
    rows_near_duplicate,
)


# Typical underground rock-mechanics MPa / ratio ranges (sanity, not hard physics)
_SCALE_BOUNDS: dict[str, tuple[float, float]] = {
    "sigma_theta": (0.1, 500.0),
    "sigma_c": (0.5, 400.0),
    "sigma_t": (0.01, 50.0),
    "scf": (0.0, 5.0),
    "b1": (0.5, 200.0),
    "b2": (0.0, 50.0),
    "wet": (0.1, 80.0),
    "d": (10.0, 5000.0),
}


def _fill_nan_with_median(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_med = np.nanmedian(X, axis=0)
    out = X.copy()
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(col_med, inds[1])
    return out, col_med


def _mahalanobis_ood_score(X: np.ndarray, x_case: np.ndarray) -> tuple[float, bool]:
    """Return (reliability in [0,1], is_ood). Higher reliability = more in-distribution."""
    if len(X) < 10:
        return 1.0, False
    mu = X.mean(axis=0)
    xc = x_case - mu
    try:
        cov = np.cov(X, rowvar=False)
        # Shrink diagonal for stability
        cov = cov + 1e-3 * np.eye(cov.shape[0])
        inv = np.linalg.pinv(cov)
        d2 = float(xc @ inv @ xc)
    except Exception:  # noqa: BLE001
        return 1.0, False
    # Chi-square-ish: p≈0.99 for 8-df is ~20; map distance to reliability
    r = float(np.exp(-0.5 * max(d2, 0.0) / max(X.shape[1], 1)))
    return r, d2 > 25.0


def _knn_ood_score(X: np.ndarray, x_case: np.ndarray, k: int = 5) -> tuple[float, bool]:
    if len(X) < k + 1:
        return 1.0, False
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(x_case.reshape(1, -1))
    mean_d = float(dists.mean())
    # Reference pairwise scale
    sample = X if len(X) <= 200 else X[np.random.default_rng(0).choice(len(X), 200, replace=False)]
    ref_nn = NearestNeighbors(n_neighbors=min(k, len(sample) - 1), metric="euclidean")
    ref_nn.fit(sample)
    ref_d, _ = ref_nn.kneighbors(sample)
    ref_mean = float(np.median(ref_d[:, -1]))
    ratio = mean_d / max(ref_mean, 1e-9)
    r = float(1.0 / (1.0 + max(ratio - 1.0, 0.0)))
    return r, ratio > 2.5


def combination_scale_issues(case: dict[str, Any], *, rel_tol: float = 0.05) -> list[str]:
    """M12: combination (identity) and magnitude/scale sanity flags."""
    issues: list[str] = []
    st = case.get("sigma_theta")
    sc = case.get("sigma_c")
    scf = case.get("scf")
    if (
        st is not None
        and sc is not None
        and scf is not None
        and all(np.isfinite(float(v)) for v in (st, sc, scf))
        and abs(float(sc)) > 1e-9
    ):
        expected = float(st) / float(sc)
        denom = max(abs(expected), abs(float(scf)), 1e-12)
        if abs(expected - float(scf)) / denom > rel_tol:
            issues.append(
                f"combo_scf_mismatch=listed={float(scf):.4g},sigma_theta/sigma_c={expected:.4g}"
            )
    b1 = case.get("b1")
    stens = case.get("sigma_t")
    if (
        b1 is not None
        and sc is not None
        and stens is not None
        and all(np.isfinite(float(v)) for v in (b1, sc, stens))
        and abs(float(stens)) > 1e-9
    ):
        expected_b1 = float(sc) / float(stens)
        denom = max(abs(expected_b1), abs(float(b1)), 1e-12)
        if abs(expected_b1 - float(b1)) / denom > rel_tol:
            issues.append(
                f"combo_b1_mismatch=listed={float(b1):.4g},sigma_c/sigma_t={expected_b1:.4g}"
            )
    for feat, (lo, hi) in _SCALE_BOUNDS.items():
        val = case.get(feat)
        if val is None or not np.isfinite(float(val)):
            continue
        v = float(val)
        if v < lo or v > hi:
            issues.append(f"scale_out_of_range={feat}:{v:g} not in [{lo:g},{hi:g}]")
    return issues


def _iqr_bounds(col: np.ndarray, factor: float) -> tuple[float, float]:
    q1, q3 = np.nanpercentile(col, [25, 75])
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def score_data_quality(
    case: dict[str, Any],
    reference: pd.DataFrame,
    *,
    iqr_factor: float = 3.0,
    scf_ood_threshold: float = 2.0,
    near_dup_rel_tol: float = DEFAULT_REL_TOL,
    near_dup_rtol: float | None = None,  # legacy alias ignored if rel_tol set
    near_dup_atol: float | None = None,  # legacy unused
    group_col: str = "dup_group_id",
) -> dict[str, Any]:
    """Score one case against a reference set.

    Caller must pass a training-fold reference that already excludes the case
    and its ``dup_group_id`` (see ``build_fold_reference``). This function also
    drops matching ``no`` / group rows defensively.

    Near-duplicate detection uses the same raw-feature relative tolerance as
    ``assign_dup_groups``.
    """
    del near_dup_rtol, near_dup_atol  # legacy kwargs accepted for call-site compat
    ref = reference.copy()
    case_no = case.get("no")
    if case_no is not None and "no" in ref.columns:
        ref = ref[ref["no"].astype(int) != int(case_no)]
    case_gid = case.get(group_col)
    if case_gid is not None and group_col in ref.columns:
        ref = ref[ref[group_col] != case_gid]

    issues: list[str] = []
    missing = [
        c
        for c in FEATURE_COLS
        if case.get(c) is None or not np.isfinite(float(case[c]))
    ]
    if missing:
        issues.append(f"missing_features={missing}")

    outlier_feats = []
    for c in FEATURE_COLS:
        val = case.get(c)
        if val is None or not np.isfinite(float(val)):
            continue
        col = ref[c].to_numpy(dtype=float)
        if len(col) == 0 or not np.isfinite(col).any():
            continue
        lo, hi = _iqr_bounds(col, iqr_factor)
        if float(val) < lo or float(val) > hi:
            outlier_feats.append(c)
    if outlier_feats:
        issues.append(f"iqr_outliers={outlier_feats}")

    scf = case.get("scf")
    ood = bool(
        scf is not None and np.isfinite(float(scf)) and float(scf) > scf_ood_threshold
    )
    if ood:
        issues.append(f"scf_ood={scf}")

    near_dups: list[int] = []
    conflicting_near_dup_labels = False
    vec_raw = np.array([float(case.get(c, np.nan)) for c in RAW_DUP_COLS], dtype=float)
    vec_all = np.array([float(case.get(c, np.nan)) for c in FEATURE_COLS], dtype=float)
    if len(ref) > 0:
        ref_mat = ref[list(RAW_DUP_COLS)].to_numpy(dtype=float)
        nos = ref["no"].to_numpy() if "no" in ref.columns else np.arange(len(ref))
        for i, row in enumerate(ref_mat):
            if rows_near_duplicate(vec_raw, row, rel_tol=near_dup_rel_tol):
                near_dups.append(int(nos[i]))
    if near_dups:
        issues.append(f"near_duplicates={near_dups[:10]}")
        # M5: near-duplicates with conflicting rockburst grades (label noise signal)
        if "rockburst_grade" in ref.columns and "no" in ref.columns:
            grades: list[int] = []
            case_g = case.get("rockburst_grade")
            if case_g is not None and np.isfinite(float(case_g)):
                grades.append(int(case_g))
            for no in near_dups:
                hit = ref.loc[ref["no"].astype(int) == int(no), "rockburst_grade"]
                if len(hit):
                    grades.append(int(hit.iloc[0]))
            if len(set(grades)) > 1:
                conflicting_near_dup_labels = True
                issues.append("conflicting_near_dup_labels")

    for iss in combination_scale_issues(case, rel_tol=near_dup_rel_tol):
        issues.append(iss)

    r_iforest = 1.0
    r_mahal = 1.0
    r_knn = 1.0
    ood_if = ood_mahal = ood_knn = False
    try:
        if len(ref) >= 10:
            X_raw = ref[FEATURE_COLS].to_numpy(dtype=float)
            X, col_med = _fill_nan_with_median(X_raw)
            x_case = vec_all.copy()
            miss = ~np.isfinite(x_case)
            x_case[miss] = col_med[miss]

            clf = IsolationForest(random_state=42, contamination=0.05)
            clf.fit(X)
            score = float(clf.decision_function(x_case.reshape(1, -1))[0])
            r_iforest = float(1 / (1 + np.exp(-5 * score)))
            ood_if = score < 0
            if ood_if:
                issues.append("isolation_forest_outlier")

            r_mahal, ood_mahal = _mahalanobis_ood_score(X, x_case)
            if ood_mahal:
                issues.append("mahalanobis_ood")

            r_knn, ood_knn = _knn_ood_score(X, x_case)
            if ood_knn:
                issues.append("knn_ood")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"ood_detectors_skipped={exc}")

    # Ensemble OOD reliability: mean of IF / Mahalanobis / kNN (M12 comparison)
    r_ood = float(np.mean([r_iforest, r_mahal, r_knn]))
    conflict_pen = 0.1 if conflicting_near_dup_labels else 0.0
    combo_pen = 0.08 * sum(1 for i in issues if i.startswith("combo_"))
    scale_pen = 0.05 * sum(1 for i in issues if i.startswith("scale_out_of_range"))
    miss_pen = 0.08 * len(missing)
    out_pen = 0.05 * len(outlier_feats)
    ood_pen = 0.2 if ood else 0.0
    dup_pen = 0.05 if near_dups else 0.0
    r_data = float(
        np.clip(
            0.5 * r_ood
            + 0.5
            * (1 - miss_pen - out_pen - ood_pen - dup_pen - conflict_pen - combo_pen - scale_pen),
            0,
            1,
        )
    )

    return {
        "data_reliability": r_data,
        "ood_flag": bool(ood or ood_if or ood_mahal or ood_knn),
        "ood_scores": {
            "isolation_forest": r_iforest,
            "mahalanobis": r_mahal,
            "knn": r_knn,
        },
        "missing_features": missing,
        "near_duplicates": near_dups[:20],
        "conflicting_near_dup_labels": conflicting_near_dup_labels,
        "issues": issues,
    }
