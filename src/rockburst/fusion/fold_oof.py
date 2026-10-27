"""Fold-local ReliabilityFusion fit on cross-fitted OOF reliability vectors.

Used by ablation / missing-data eval so the fold stacker + calibrator never see
in-sample (fit_idx) reliability vectors. Production ``train.py`` already builds
outer-OOF R then cross-fits; this helper mirrors that discipline inside one
outer-train portion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.schemas import PredictiveOutput
from rockburst.agents.tools import run_data_quality, run_physics, run_uncertainty
from rockburst.fusion.features import compute_r_model, compute_reliability_features
from rockburst.fusion.stacker import (
    FEATURE_NAMES,
    ReliabilityFusion,
    select_decision_thresholds,
)
from rockburst.ml.pipelines import build_preprocess
from rockburst.ml.train import (
    _predict_proba_safe,
    _split_fit_calib,
    fit_fold_ensemble,
)
from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
)


def _fit_conf(method: str, y_calib: np.ndarray, p_cal: np.ndarray, alpha: float):
    if method == "mondrian_split":
        return fit_mondrian_thresholds(y_calib, p_cal, alpha=alpha)
    return fit_marginal_thresholds(y_calib, p_cal, alpha=alpha)


def collect_oof_reliability_in_train(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    *,
    base_est: Any,
    imputer: str,
    ranked: list[tuple[str, dict[str, Any]]],
    cfg: dict[str, Any],
    seed: int,
    fold: int,
    conf_method: str,
    alpha: float,
    n_inner: int = 3,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit reliability vectors on the outer-train indices ``tr`` only.

    Returns ``(R, y_correct, row_indices)`` with one OOF vector per scored row.
    Never scores a row with a pipe that trained on that row. Does not load
    full-data ``component_calibrators.joblib`` (cfg calibrators must be explicit).
    """
    tr = np.asarray(tr, dtype=int)
    if len(tr) == 0:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=float),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )

    y_tr, g_tr = y[tr], groups[tr]
    n_unique = len(np.unique(g_tr))
    n_sp = min(max(2, n_inner), max(2, n_unique))
    live_cfg = {
        **cfg,
        "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
        # Explicit: no fold component calibrators unless caller injected them.
        "_component_calibrators": cfg.get("_component_calibrators"),
    }
    # Drop auto-load sentinel: features.py only uses explicit calibrators.
    if live_cfg.get("_component_calibrators") is None:
        live_cfg.pop("_component_calibrators", None)

    R_by: dict[int, np.ndarray] = {}
    ycorr_by: dict[int, int] = {}

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=seed + fold)
        splits = list(sgkf.split(np.zeros(len(tr)), y_tr, g_tr))
    except ValueError:
        # Degenerate groups: hold out ~20% as a single OOF block.
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(len(tr))
        cut = max(1, int(0.2 * len(tr)))
        splits = [(perm[cut:], perm[:cut])]

    for inner_i, (fit_rel, te_rel) in enumerate(splits):
        inner_tr = tr[fit_rel]
        inner_te = tr[te_rel]
        fit_idx, calib_idx = _split_fit_calib(
            y, groups, inner_tr, seed=seed + fold * 17 + inner_i
        )
        pipe = Pipeline(
            [("prep", build_preprocess(imputer)), ("model", clone(base_est))]
        )
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        conf = None
        if len(calib_idx):
            conf = _fit_conf(
                conf_method,
                y[calib_idx],
                _predict_proba_safe(pipe, X.iloc[calib_idx]),
                alpha,
            )
            conf.n_fit = int(len(fit_idx))
        fold_ensemble = fit_fold_ensemble(
            X, y, fit_idx, ranked, seed=seed + fold, fold=fold, top_k=3
        )
        for j in inner_te:
            j = int(j)
            if j in R_by:
                continue
            case = df.iloc[j].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
            obs = {
                c: bool(np.isfinite(float(case[c]))) if case.get(c) is not None else False
                for c in FEATURE_COLS
            }
            pred = PredictiveOutput(
                predicted_grade=int(proba.argmax()),
                probabilities=proba.tolist(),
                r_model=compute_r_model(proba),
                observed_mask=obs,
                model_name="fold_oof",
            )
            dq = run_data_quality(case, ref, live_cfg)
            phys = run_physics(case, live_cfg)
            unc = run_uncertainty(
                case,
                pred,
                live_cfg,
                pipeline=pipe,
                ensemble=fold_ensemble,
                conformal=conf,
            )
            # No cfg auto-load: pass cfg only for explicit calibrators.
            R_by[j] = compute_reliability_features(
                dq, phys, pred, unc, case=case, cfg=live_cfg
            )
            ycorr_by[j] = int(pred.predicted_grade == int(y[j]))

    row_indices = np.array(sorted(R_by.keys()), dtype=int)
    if max_rows is not None and len(row_indices) > max_rows:
        rng = np.random.default_rng(seed + fold)
        row_indices = np.sort(rng.choice(row_indices, size=max_rows, replace=False))
    if len(row_indices) == 0:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=float),
            np.zeros(0, dtype=int),
            row_indices,
        )
    R = np.vstack([R_by[int(j)] for j in row_indices])
    y_correct = np.array([ycorr_by[int(j)] for j in row_indices], dtype=int)
    return R, y_correct, row_indices


def fit_fold_fusion_oof(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    *,
    base_est: Any,
    imputer: str,
    ranked: list[tuple[str, dict[str, Any]]],
    cfg: dict[str, Any],
    seed: int,
    fold: int,
    conf_method: str,
    alpha: float,
    quick: bool = False,
) -> ReliabilityFusion:
    """Fit fold stacker + calibrator + thresholds on train-portion OOF vectors only."""
    n_inner = 2 if quick else 3
    max_rows = 24 if quick else 80
    R, y_correct, row_idx = collect_oof_reliability_in_train(
        df,
        X,
        y,
        groups,
        tr,
        base_est=base_est,
        imputer=imputer,
        ranked=ranked,
        cfg=cfg,
        seed=seed,
        fold=fold,
        conf_method=conf_method,
        alpha=alpha,
        n_inner=n_inner,
        max_rows=max_rows,
    )
    fusion = ReliabilityFusion()
    fusion.conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
    if len(R) == 0:
        return fusion

    # RC3: fit stacker on all train-portion OOF; calibrate on all those raw scores
    # (not a tiny inner holdout). Nested split only if we have enough rows for
    # a cheap cross-fit raw collection.
    fusion.fit(R, y_correct)
    if not fusion.is_fit:
        return fusion

    g_rows = groups[row_idx]
    n_local = len(row_idx)
    if n_local >= 20 and len(np.unique(g_rows)) >= 3:
        # Nested grouped folds over OOF rows → cross-fitted raw for calibrator.
        fold_of = np.full(n_local, -1, dtype=int)
        try:
            n_sp = min(3, len(np.unique(g_rows)))
            sgkf = StratifiedGroupKFold(
                n_splits=n_sp, shuffle=True, random_state=seed + 200 + fold
            )
            for fi, (_tr, te) in enumerate(
                sgkf.split(np.zeros(n_local), y_correct, g_rows)
            ):
                fold_of[te] = fi
        except ValueError:
            fold_of[:] = 0
        from rockburst.fusion.stacker import fit_calibrator_on_crossfit_raw

        fit_calibrator_on_crossfit_raw(
            fusion,
            R,
            y_correct,
            fold_of,
            feature_mask=fusion.feature_mask,
            conf_eps=fusion.conf_eps,
            method=(cfg.get("fusion") or {}).get("calibration_method"),
        )
        # Re-fit stacker on all local OOF after attaching calibrator.
        fusion.fit(R, y_correct)
        r_vals = np.array([fusion.r_final(r) for r in R])
    else:
        r_raw = np.array([fusion._raw_score(r) for r in R])
        fusion.fit_calibrator(
            r_raw,
            y_correct,
            method=(cfg.get("fusion") or {}).get("calibration_method"),
        )
        r_vals = np.array([fusion.r_final(r) for r in R])
    thr = select_decision_thresholds(r_vals, y_correct)
    fusion.set_decision_thresholds(thr)
    return fusion
