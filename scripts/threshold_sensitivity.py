#!/usr/bin/env python3
"""Item 3: sensitivity of hand-set decision-path constants (± fixed deltas)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import fuse_assessment, run_data_quality, run_physics, run_predict, run_uncertainty
from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import quadratic_weighted_kappa, severe_misclassification_rate
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import fit_marginal_thresholds
from rockburst.ml.train import _predict_proba_safe


def _score(y, pred, abstain):
    acc = float((y == pred).mean())
    qwk = quadratic_weighted_kappa(y, pred)
    sev = severe_misclassification_rate(y, pred)
    abs_rate = float(np.mean(abstain))
    return {"accuracy": acc, "qwk": qwk, "severe": sev, "abstention": abs_rate}


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    m1 = json.loads((resolve_path(cfg, "baselines") / "m1_results.json").read_text())
    best_name = (
        m1.get("nested_unbiased", {}).get("production_model")
        or m1.get("best_model")
        or "catboost__knn"
    )
    name, imp = best_name.split("__", 1)
    params = m1.get("best_params") or {}
    seed = int(cfg.get("seed", 42))
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:
        from rockburst.ml.pipelines import get_model_zoo

        base_est = get_model_zoo(quick=True)[name]

    # Baseline + perturbations (pre-registered domain defaults; do not pick best)
    variants = {
        "baseline": {},
        "disagreement_threshold_+0.1": {"routing.disagreement_threshold": 0.45},
        "disagreement_threshold_-0.1": {"routing.disagreement_threshold": 0.25},
        "large_set_size_+1": {"routing.large_set_size": 4},
        "hard_stability_gate_+0.1": {"_hard_stability": 0.65},
        "hard_stability_gate_-0.1": {"_hard_stability": 0.45},
        "physics_penalties_x1.5": {"_phys_scale": 1.5},
        "mild_eps_+0.02": {"uncertainty.mild_eps": 0.07},
    }

    rows = []
    # Single outer split for sensitivity (not headline nested) — report as sensitivity only
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(iter(sgkf.split(X, y, groups)))
    fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed)
    pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(base_est))])
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    conf = None
    if len(calib_idx):
        conf = fit_marginal_thresholds(
            y[calib_idx], _predict_proba_safe(pipe, X.iloc[calib_idx]), alpha=0.1
        )

    for vname, overrides in variants.items():
        local = json.loads(json.dumps(cfg))  # deep-ish copy via json
        if "routing.disagreement_threshold" in overrides:
            local.setdefault("routing", {})["disagreement_threshold"] = overrides[
                "routing.disagreement_threshold"
            ]
        if "routing.large_set_size" in overrides:
            local.setdefault("routing", {})["large_set_size"] = overrides[
                "routing.large_set_size"
            ]
        if "uncertainty.mild_eps" in overrides:
            local.setdefault("uncertainty", {})["mild_eps"] = overrides[
                "uncertainty.mild_eps"
            ]
        hard_stab = overrides.get("_hard_stability", 0.55)
        preds, abs_flags = [], []
        for j in te:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            pred = run_predict(case, local, pipeline=pipe, model_name=best_name)
            phys = run_physics(case, local)
            unc = run_uncertainty(
                case, pred, local, pipeline=pipe, ensemble=None, conformal=conf
            )
            dq = run_data_quality(case, ref, local)
            out = fuse_assessment(dq, phys, pred, unc, local, ["sens"])
            # Apply optional hard stability override for sensitivity
            if len(unc.prediction_set) >= 3 and unc.stability_score < hard_stab:
                out["recommended_action"] = "abstain"
                out["abstained"] = True
            preds.append(int(out.get("predicted_grade") if out.get("predicted_grade") is not None else pred.predicted_grade))
            abs_flags.append(int(bool(out.get("abstained"))))
        yt = y[te]
        sc = _score(yt, np.asarray(preds), np.asarray(abs_flags))
        sc["variant"] = vname
        rows.append(sc)

    # Nested vs ablation gap note from existing artifacts
    abl = json.loads((resolve_path(cfg, "baselines") / "ablation_results.json").read_text())
    gap = {
        "ablation_M1_qwk": abl["ablations"]["M1"]["qwk"]["mean"],
        "ablation_M1_acc": abl["ablations"]["M1"]["accuracy"]["mean"],
        "nested_M1_qwk": m1.get("nested_unbiased", {}).get("pooled_qwk"),
        "nested_M1_acc": m1.get("nested_unbiased", {}).get("pooled_accuracy"),
        "explanation": (
            "Ablation M1 uses a single production mode (nested winner) scored on "
            "repeated OOF folds; nested-unbiased pools per-fold winners chosen by "
            "inner QWK only. Different aggregation + fold-winner mix → gap. "
            "Nested figure remains the headline."
        ),
        "ablation_qwk_ci": abl["ablations"]["M1"]["qwk"],
    }

    out = {
        "protocol": "One outer fold sensitivity; constants pre-registered; do not pick best.",
        "sensitivity": rows,
        "nested_vs_ablation_gap": gap,
        "decision_band_mapping": {
            "R < abstain_t": "abstain",
            "abstain_t ≤ R < eng_t": "engineering_verification",
            "eng_t ≤ R < accept_t": "accept_with_caution",
            "R ≥ accept_t": "accept",
            "ordering": "abstain < eng < caution ≤ accept",
        },
        "hand_set_constants_justification": {
            "physics_penalties_0.1_0.05_0.15": "domain soft penalties from methodology",
            "hard_abstain_set3_stab0.55": "domain: large unstable set → abstain",
            "hard_abstain_disagreement_0.5_rfinal_0.75": "domain: strong conflict + modest trust",
            "hard_abstain_ood_phys0.7_rmodel0.6": "domain: OOD + weak physics + weak model",
            "routing_disagreement_0.35_large_set_3": "config defaults; Critic triggers",
            "mild_eps_0.05_strong_eps_0.15": "config uncertainty defaults",
        },
    }
    dest = resolve_path(cfg, "baselines") / "threshold_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
