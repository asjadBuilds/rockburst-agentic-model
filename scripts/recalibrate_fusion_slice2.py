#!/usr/bin/env python3
"""Slice 2 (C3/C4): recalibrate fusion from saved OOF R; write risk-coverage note.

Does NOT recompute live R (that needs full train with fold ensembles).
Uses artifacts/models/fusion_R_oof.npy + fusion_y_correct.npy when present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import expected_calibration_error
from rockburst.fusion.stacker import (
    ReliabilityFusion,
    locked_feature_mask,
    locked_feature_report,
    select_decision_thresholds,
)
from rockburst.ml.pipelines import prepare_xy
from rockburst.ml.train import _risk_coverage_curve, _split_fit_calib


def main() -> None:
    cfg = load_config()
    models_dir = resolve_path(cfg, "models")
    baselines = resolve_path(cfg, "baselines")
    r_path = models_dir / "fusion_R_oof.npy"
    y_path = models_dir / "fusion_y_correct.npy"
    if not r_path.exists() or not y_path.exists():
        raise SystemExit(
            "Missing fusion_R_oof.npy / fusion_y_correct.npy — run full train first."
        )

    R = np.load(r_path)
    y_correct = np.load(y_path)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    assert len(R) == len(y) == len(y_correct)

    m1_path = baselines / "m1_results.json"
    m1 = json.loads(m1_path.read_text()) if m1_path.exists() else {}
    before = {
        "fusion_r_final_ece": m1.get("fusion_r_final_ece"),
        "fusion_mean_r_final": m1.get("fusion_mean_r_final"),
        "fusion_brier": m1.get("fusion_brier"),
        "decision_thresholds": m1.get("decision_thresholds"),
        "fusion_calibration_method": m1.get("fusion_calibration_method"),
        "fusion_calibration_n": m1.get("fusion_calibration_n"),
    }

    n_outer = int(cfg["cv"]["n_outer"])
    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(y), -1, dtype=int)
    for fold, (_tr, te) in enumerate(sgkf.split(X, y, groups)):
        fold_of_row[te] = fold

    mask = locked_feature_mask()
    feat_report = locked_feature_report()
    conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
    r_final_oof = np.zeros(len(y), dtype=float)
    cal_ns: list[int] = []
    cal_methods: list[str | None] = []

    for fold in range(n_outer):
        train_mask = (fold_of_row >= 0) & (fold_of_row != fold)
        te_mask = fold_of_row == fold
        if not train_mask.any() or not te_mask.any():
            continue
        idx_fit = np.where(train_mask)[0]
        idx_te = np.where(te_mask)[0]
        _inner_fit, inner_val = _split_fit_calib(
            y, groups, idx_fit, seed=seed + 100 + fold
        )
        fold_fusion = ReliabilityFusion()
        fold_fusion.conf_eps = conf_eps
        fold_fusion.fit(
            R[idx_fit], y_correct[idx_fit], feature_mask=mask, select_features=False
        )
        fold_fusion.feature_report = feat_report
        if not fold_fusion.is_fit:
            continue
        r_raw_fit = np.array([fold_fusion._raw_score(R[i]) for i in idx_fit])
        val_pos = np.isin(idx_fit, inner_val)
        if val_pos.any():
            fold_fusion.fit_calibrator(
                r_raw_fit[val_pos], y_correct[idx_fit][val_pos]
            )
            cal_ns.append(int(fold_fusion.calibration_n))
            cal_methods.append(fold_fusion.calibration_method)
            r_val = np.array(
                [fold_fusion.r_final(R[i]) for i in idx_fit[val_pos]]
            )
            thr = select_decision_thresholds(r_val, y_correct[idx_fit][val_pos])
            fold_fusion.set_decision_thresholds(thr)
        for j in idx_te:
            r_final_oof[j] = fold_fusion.r_final(R[j])

    # Production serving blob on all OOF
    _inner_fit, inner_val = _split_fit_calib(
        y, groups, np.arange(len(y)), seed=seed + 99
    )
    fusion = ReliabilityFusion()
    fusion.conf_eps = conf_eps
    fusion.fit(R, y_correct, feature_mask=mask, select_features=False)
    fusion.feature_report = feat_report
    if fusion.is_fit:
        r_raw_all = np.array([fusion._raw_score(R[i]) for i in range(len(y))])
        fusion.fit_calibrator(r_raw_all[inner_val], y_correct[inner_val])
        thr_prod = select_decision_thresholds(
            np.array([fusion.r_final(R[i]) for i in inner_val]),
            y_correct[inner_val],
        )
        fusion.set_decision_thresholds(thr_prod)
    fusion.save(models_dir / "fusion.joblib")
    np.save(models_dir / "fusion_r_final_crossfit.npy", r_final_oof)

    held = fold_of_row >= 0
    # Approximate max-prob via r_model channel (locked set A)
    i_m = 2  # FEATURE_NAMES index of r_model
    max_proxy = R[held, i_m]
    rc_max = _risk_coverage_curve(max_proxy, y_correct[held])
    rc_rf = _risk_coverage_curve(r_final_oof[held], y_correct[held])

    after = {
        "fusion_coef": fusion.coef_.tolist() if fusion.is_fit else None,
        "fusion_intercept": float(fusion.intercept_) if fusion.is_fit else None,
        "decision_thresholds": fusion.decision_thresholds,
        "fusion_calibration_method": fusion.calibration_method,
        "fusion_calibration_n": int(fusion.calibration_n),
        "fusion_crossfit_calibration_n_mean": float(np.mean(cal_ns)) if cal_ns else None,
        "fusion_crossfit_calibration_methods": cal_methods,
        "fusion_r_final_ece": expected_calibration_error(
            y_correct[held],
            np.column_stack([1 - r_final_oof[held], r_final_oof[held]]),
        ),
        "fusion_brier": float(brier_score_loss(y_correct[held], r_final_oof[held])),
        "fusion_mean_r_final": float(r_final_oof[held].mean()),
        "fusion_oof_accuracy_of_base": float(y_correct[held].mean()),
        "risk_coverage_max_prob": rc_max,
        "risk_coverage_r_final": rc_rf,
        "risk_coverage_comparison": {
            "max_prob_area": rc_max.get("area"),
            "r_final_area": rc_rf.get("area"),
            "r_final_minus_max_prob": (
                None
                if rc_max.get("area") is None or rc_rf.get("area") is None
                else float(rc_rf["area"] - rc_max["area"])
            ),
            "note": (
                "r_model channel used as max-prob proxy (set A). "
                "Full max-prob needs retrain. Near-zero Δ = ranking collapse."
            ),
        },
        "zero_weight_flags": feat_report.get("zero_weight_flags", []),
        "chosen_fusion_set": feat_report.get("chosen_set"),
        "protocol": (
            "slice2 recalibrate from saved OOF R; fold-ensemble R rebuild "
            "requires `rockburst train`"
        ),
    }

    out = {"before": before, "after": after}
    out_path = baselines / "fusion_slice2_recalibrate.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    # Patch key fields into m1_results for serving consistency
    if m1_path.exists():
        m1.update(
            {
                "fusion_r_final_ece": after["fusion_r_final_ece"],
                "fusion_brier": after["fusion_brier"],
                "fusion_mean_r_final": after["fusion_mean_r_final"],
                "decision_thresholds": after["decision_thresholds"],
                "fusion_calibration_method": after["fusion_calibration_method"],
                "fusion_calibration_n": after["fusion_calibration_n"],
                "risk_coverage_max_prob": after["risk_coverage_max_prob"],
                "risk_coverage_r_final": after["risk_coverage_r_final"],
                "risk_coverage_comparison": after["risk_coverage_comparison"],
                "fusion_slice2_recalibrate": str(out_path),
            }
        )
        m1_path.write_text(json.dumps(m1, indent=2, default=str))

    print(json.dumps({"wrote": str(out_path), **after["risk_coverage_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
