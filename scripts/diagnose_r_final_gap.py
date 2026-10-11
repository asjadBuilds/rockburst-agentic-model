#!/usr/bin/env python3
"""Diagnose R_final vs max-prob (Spearman, top ties, fold coefs). Prints <=15 lines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import run_data_quality, run_physics, run_predict, run_uncertainty
from rockburst.config import load_config, resolve_path
from rockburst.fusion.features import compute_r_model, compute_reliability_features
from rockburst.fusion.stacker import FEATURE_NAMES, ReliabilityFusion
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.agents.schemas import PredictiveOutput
from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.fold_ref import build_fold_reference


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    m1 = resolve_path(cfg, "baselines") / "m1_results.json"
    import json

    meta = json.loads(m1.read_text()) if m1.exists() else {}
    best = meta.get("nested_unbiased", {}).get("production_model") or meta.get("best_model") or "catboost__knn"
    name, imp = best.split("__", 1)
    params = meta.get("best_params") or {}
    seed = int(cfg.get("seed", 42))
    try:
        est = estimator_from_params(name, params, seed)
    except Exception:
        from rockburst.ml.pipelines import get_model_zoo

        est = get_model_zoo(quick=True)[name]

    spearman_folds, coefs, top_ties = [], [], []
    r_all, m_all = [], []
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        fit_idx, _ = _split_fit_calib(y, groups, tr, seed=seed + fold)
        pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        R_tr, ycorr = [], []
        for j in fit_idx[:40]:
            case = df.iloc[int(j)].to_dict()
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
            )
            dq = run_data_quality(case, ref, cfg)
            phys = run_physics(case, cfg)
            unc = run_uncertainty(case, pred, cfg, pipeline=pipe)
            R_tr.append(compute_reliability_features(dq, phys, pred, unc, case=case))
            ycorr.append(int(pred.predicted_grade == int(y[j])))
        fusion = ReliabilityFusion().fit(np.asarray(R_tr), np.asarray(ycorr))
        if fusion.is_fit and fusion.coef_ is not None:
            coefs.append(fusion.coef_)
        rf, mp = [], []
        for j in te:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            pred = run_predict(case, cfg, pipeline=pipe)
            dq = run_data_quality(case, ref, cfg)
            phys = run_physics(case, cfg)
            unc = run_uncertainty(case, pred, cfg, pipeline=pipe)
            rvec = compute_reliability_features(dq, phys, pred, unc, case=case)
            rf.append(fusion.r_final(rvec) if fusion.is_fit else float("nan"))
            mp.append(float(max(pred.probabilities)))
        rf_a, mp_a = np.asarray(rf), np.asarray(mp)
        spearman_folds.append(float(spearmanr(rf_a, mp_a).correlation))
        top = max(int(0.1 * len(rf_a)), 1)
        thr = np.partition(rf_a, -top)[-top]
        top_ties.append(float(np.mean(rf_a >= thr - 1e-12)))
        r_all.extend(rf)
        m_all.extend(mp)

    print("spearman_per_fold:", [round(x, 4) for x in spearman_folds])
    print("spearman_mean:", round(float(np.mean(spearman_folds)), 4))
    print("top10pct_tie_share_mean:", round(float(np.mean(top_ties)), 4))
    if coefs:
        mean_c = np.mean(coefs, axis=0)
        print("coef_names:", FEATURE_NAMES)
        print("coef_mean:", [round(float(x), 4) for x in mean_c])
    r_all, m_all = np.asarray(r_all), np.asarray(m_all)
    print("global_spearman:", round(float(spearmanr(r_all, m_all).correlation), 4))
    print("n_heldout:", len(r_all))


if __name__ == "__main__":
    main()
