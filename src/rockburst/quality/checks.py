"""Data quality / anomaly scoring (fold-aware when reference is train-only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.near_dup import (
    DEFAULT_REL_TOL,
    RAW_DUP_COLS,
    rows_near_duplicate,
)


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

    r_iforest = 1.0
    try:
        if len(ref) >= 10:
            X = ref[FEATURE_COLS].to_numpy(dtype=float)
            col_med = np.nanmedian(X, axis=0)
            inds = np.where(~np.isfinite(X))
            X = X.copy()
            X[inds] = np.take(col_med, inds[1])
            clf = IsolationForest(random_state=42, contamination=0.05)
            clf.fit(X)
            x_case = vec_all.copy()
            miss = ~np.isfinite(x_case)
            x_case[miss] = col_med[miss]
            score = float(clf.decision_function(x_case.reshape(1, -1))[0])
            r_iforest = float(1 / (1 + np.exp(-5 * score)))
            if score < 0:
                issues.append("isolation_forest_outlier")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"isolation_forest_skipped={exc}")

    miss_pen = 0.08 * len(missing)
    out_pen = 0.05 * len(outlier_feats)
    ood_pen = 0.2 if ood else 0.0
    dup_pen = 0.05 if near_dups else 0.0
    r_data = float(
        np.clip(0.5 * r_iforest + 0.5 * (1 - miss_pen - out_pen - ood_pen - dup_pen), 0, 1)
    )

    return {
        "data_reliability": r_data,
        "ood_flag": ood or ("isolation_forest_outlier" in issues),
        "missing_features": missing,
        "near_duplicates": near_dups[:20],
        "issues": issues,
    }
