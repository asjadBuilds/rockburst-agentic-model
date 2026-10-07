#!/usr/bin/env python3
"""Item 2: refit constrained fusion from saved OOF R; missing-D confidence drop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import run_data_quality, run_physics, run_predict, run_uncertainty
from rockburst.agents.schemas import PredictiveOutput
from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import expected_calibration_error
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import (
    FEATURE_NAMES,
    ReliabilityFusion,
    bootstrap_coef_cis,
    choose_feature_mask,
    select_decision_thresholds,
)
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference


def main() -> None:
    cfg = load_config()
    models_dir = resolve_path(cfg, "models")
    baselines = resolve_path(cfg, "baselines")
    R = np.load(models_dir / "fusion_R_oof.npy")
    y_correct = np.load(models_dir / "fusion_y_correct.npy")
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    assert len(R) == len(y)

    m1 = json.loads((baselines / "m1_results.json").read_text())
    before = {
        "fusion_coef": m1.get("fusion_coef"),
        "fusion_r_final_ece": m1.get("fusion_r_final_ece"),
        "fusion_mean_r_final": m1.get("fusion_mean_r_final"),
        "fusion_oof_accuracy_of_base": m1.get("fusion_oof_accuracy_of_base"),
        "decision_thresholds": m1.get("decision_thresholds"),
    }

    _inner_fit, inner_val = _split_fit_calib(
        y, groups, np.arange(len(y)), seed=seed + 99
    )
    mask, feat_report = choose_feature_mask(R[_inner_fit], y_correct[_inner_fit])
    fusion = ReliabilityFusion().fit(R, y_correct, feature_mask=mask, select_features=False)
    fusion.feature_report = feat_report
    r_raw = np.array([fusion._raw_score(R[i]) for i in range(len(y))])
    fusion.fit_calibrator(r_raw[inner_val], y_correct[inner_val])
    r_final = np.array([fusion.r_final(R[i]) for i in range(len(y))])
    thr = select_decision_thresholds(r_final[inner_val], y_correct[inner_val])
    fusion.set_decision_thresholds(thr)
    fusion.save(models_dir / "fusion.joblib")

    boot = bootstrap_coef_cis(
        R[_inner_fit],
        y_correct[_inner_fit],
        n_boot=200,
        seed=seed,
        feature_mask=mask,
    )
    after = {
        "fusion_coef": fusion.coef_.tolist(),
        "fusion_intercept": float(fusion.intercept_),
        "fusion_feature_report": feat_report,
        "fusion_zero_weight_flags": feat_report.get("zero_weight_flags", []),
        "fusion_coef_bootstrap": boot,
        "decision_thresholds": thr,
        "fusion_r_final_ece": expected_calibration_error(
            y_correct, np.column_stack([1 - r_final, r_final])
        ),
        "fusion_r_final_ece_inner_val": expected_calibration_error(
            y_correct[inner_val],
            np.column_stack([1 - r_final[inner_val], r_final[inner_val]]),
        ),
        "fusion_brier": float(brier_score_loss(y_correct, r_final)),
        "fusion_mean_r_final": float(r_final.mean()),
        "fusion_oof_accuracy_of_base": float(y_correct.mean()),
        "fusion_calibrated": fusion.calibrator is not None,
        "feature_names": FEATURE_NAMES,
    }

    # Missing-D drop on held-out folds: cases with D observed
    best_name = (
        m1.get("nested_unbiased", {}).get("production_model")
        or m1.get("best_model")
        or "catboost__knn"
    )
    name, imp = best_name.split("__", 1)
    params = m1.get("best_params") or {}
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:
        from rockburst.ml.pipelines import get_model_zoo

        zoo = get_model_zoo(quick=True)
        base_est = zoo.get(name, next(iter(zoo.values())))

    drops = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + fold)
        pipe = Pipeline(
            [("prep", build_preprocess(imp)), ("model", clone(base_est))]
        )
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        # Fit fold fusion on train live R (subset)
        R_tr, ycorr = [], []
        for j in fit_idx[:80]:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
            pred = PredictiveOutput(
                predicted_grade=int(proba.argmax()),
                probabilities=proba.tolist(),
                r_model=float(__import__("numpy").asarray(proba).max()),  # raw max-prob; missingness is a stacker feature
                model_name="fold",
            )
            dq = run_data_quality(case, ref, cfg)
            phys = run_physics(case, cfg)
            unc = run_uncertainty(case, pred, cfg, pipeline=pipe, ensemble=None)
            R_tr.append(compute_reliability_features(dq, phys, pred, unc))
            ycorr.append(int(pred.predicted_grade == int(y[j])))
        fold_fusion = ReliabilityFusion().fit(np.asarray(R_tr), np.asarray(ycorr))
        if not fold_fusion.is_fit:
            continue
        for j in te:
            case = df.iloc[int(j)].to_dict()
            d_val = case.get("d")
            if d_val is None or (
                isinstance(d_val, float) and not np.isfinite(float(d_val))
            ):
                continue
            ref = build_fold_reference(df, fit_idx, case)
            proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
            pred = PredictiveOutput(
                predicted_grade=int(proba.argmax()),
                probabilities=proba.tolist(),
                r_model=float(proba.max()),
                model_name="fold",
            )
            pred = run_predict(case, cfg, pipeline=pipe, model_name="fold")
            dq = run_data_quality(case, ref, cfg)
            phys = run_physics(case, cfg)
            unc = run_uncertainty(case, pred, cfg, pipeline=pipe, ensemble=None)
            r_obs = float(fold_fusion.r_final(compute_reliability_features(dq, phys, pred, unc)))
            case_m = dict(case)
            case_m["d"] = float("nan")
            pred_m = run_predict(case_m, cfg, pipeline=pipe, model_name="fold")
            dq_m = run_data_quality(case_m, ref, cfg)
            phys_m = run_physics(case_m, cfg)
            unc_m = run_uncertainty(case_m, pred_m, cfg, pipeline=pipe, ensemble=None)
            r_masked = float(
                fold_fusion.r_final(compute_reliability_features(dq_m, phys_m, pred_m, unc_m))
            )
            drops.append(
                {
                    "no": int(case["no"]),
                    "fold": fold,
                    "r_obs": r_obs,
                    "r_masked": r_masked,
                    "drop": r_obs - r_masked,
                }
            )

    drop_arr = np.array([d["drop"] for d in drops], dtype=float) if drops else np.array([])
    miss_report = {
        "n_observed_d_cases": len(drops),
        "mean_r_obs": float(np.mean([d["r_obs"] for d in drops])) if drops else None,
        "mean_r_masked": float(np.mean([d["r_masked"] for d in drops])) if drops else None,
        "mean_drop": float(drop_arr.mean()) if len(drop_arr) else None,
        "frac_drop_positive": float((drop_arr > 0).mean()) if len(drop_arr) else None,
        "note": (
            "Positive drop means R_final falls when D is masked. "
            "If mean_drop≈0, R_data/R_physics may have zero fusion weight."
        ),
    }

    # Reliability diagram
    bins = np.linspace(0, 1, 11)
    diagram = []
    for i in range(10):
        m = (r_final > bins[i]) & (r_final <= bins[i + 1])
        if m.sum() == 0:
            continue
        diagram.append(
            {
                "bin": [float(bins[i]), float(bins[i + 1])],
                "mean_r_final": float(r_final[m].mean()),
                "empirical_correct": float(y_correct[m].mean()),
                "n": int(m.sum()),
            }
        )
    after["fusion_reliability_diagram"] = diagram

    # Update m1_results.json fusion fields
    m1.update(after)
    (baselines / "m1_results.json").write_text(json.dumps(m1, indent=2, default=str))

    out = {
        "before": before,
        "after": after,
        "missing_d_drop": miss_report,
    }
    (baselines / "fusion_item2_refit.json").write_text(json.dumps(out, indent=2, default=str))
    (baselines / "fusion_missing_d_drop.json").write_text(
        json.dumps({"summary": miss_report, "cases": drops[:50]}, indent=2)
    )
    print(json.dumps({"before": before, "after_coef": after["fusion_coef"],
                      "after_ece": after["fusion_r_final_ece"],
                      "after_mean_R": after["fusion_mean_r_final"],
                      "zero_flags": after["fusion_zero_weight_flags"],
                      "collinearity": feat_report.get("collinearity_action"),
                      "missing_d": miss_report}, indent=2))


if __name__ == "__main__":
    main()
