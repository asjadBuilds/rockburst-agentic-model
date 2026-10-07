#!/usr/bin/env python3
"""Generate evidence package examples/traces.

Production-model cases are labeled as such (seen in full-data refit).
Additionally writes ≥3 held-out fold-model examples from an outer test fold.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.config import load_config, resolve_path
from rockburst.graph.runner import assess_case, run_graph
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import fit_marginal_thresholds


# Production-model showcase cases (full-data refit — labels include disclaimer)
PROD_CASES = {
    "abstain": 1,
    "standard_a": 15,
    "physics_b1": 20,
    "standard_b": 67,
    "ood": 146,
    "physics_b1_244": 244,
    "near_dup": 180,
    "failure": 329,
}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def generate_production_examples(cfg: dict) -> None:
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    for label, no in PROD_CASES.items():
        try:
            result = assess_case(cfg, no, use_llm=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip prod {no}: {exc}")
            continue
        result["example_protocol"] = (
            "production model, case seen in training "
            "(full-data refit of best_baseline.joblib)"
        )
        _write(ex / f"case_{no}.json", result)
        if label == "abstain":
            shutil.copy(ex / f"case_{no}.json", ex / "abstain_case.json")
        if label == "ood":
            shutil.copy(ex / f"case_{no}.json", ex / "case_corrupt_ood.json")
        if label == "failure":
            df = pd.read_csv(resolve_path(cfg, "processed_csv"))
            true_g = int(df.loc[df["no"] == no, "rockburst_grade"].iloc[0])
            result["failure_analysis"] = {
                "true_grade": true_g,
                "predicted_grade": result.get("predicted_grade"),
                "confident": (result.get("final_system_confidence") or 0) > 0.7,
                "diagnosis": (
                    "Wrong confident prediction"
                    if result.get("predicted_grade") != true_g
                    else "Prediction matched label"
                ),
            }
            _write(ex / "failure_case.json", result)
        print("prod", label, no, result.get("recommended_action"))


def generate_heldout_examples(cfg: dict, n: int = 3) -> None:
    """Assess cases with a fold-trained pipeline never fit on those cases."""
    ex = resolve_path(cfg, "examples")
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    meta_path = resolve_path(cfg, "baselines") / "m1_results.json"
    if not meta_path.exists():
        print("skip held-out examples: m1_results.json missing")
        return
    meta = json.loads(meta_path.read_text())
    best = meta.get("best_model") or "logistic__median"
    name, imp = best.split("__", 1)
    params = meta.get("best_params") or {}
    est = estimator_from_params(name, params, seed)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X, y, groups))
    fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed)
    pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    conf = None
    if len(calib_idx):
        conf = fit_marginal_thresholds(
            y[calib_idx], _predict_proba_safe(pipe, X.iloc[calib_idx]), alpha=0.1
        )
        conf.n_fit = int(len(fit_idx))
    fold_cfg = {
        **cfg,
        "_fold_pipeline": pipe,
        "_conformal_state": conf,
        "_fold_model_name": best,
    }
    # Prefer diverse grades in the test fold
    te_df = df.iloc[te].copy()
    chosen = []
    for g in range(4):
        sub = te_df.loc[te_df["rockburst_grade"] == g, "no"].astype(int).tolist()
        if sub:
            chosen.append(sub[0])
        if len(chosen) >= n:
            break
    while len(chosen) < n and len(te_df):
        x = int(te_df["no"].iloc[len(chosen)])
        if x not in chosen:
            chosen.append(x)

    for i, no in enumerate(chosen[:n]):
        case = df.loc[df["no"] == no].iloc[0].to_dict()
        ref = build_fold_reference(df, fit_idx, case)
        result = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
        result["example_protocol"] = (
            "held-out fold model: case was in outer test fold; "
            "pipeline fit only on fold train/fit indices"
        )
        result["fold_model"] = best
        result["true_grade"] = int(case["rockburst_grade"])
        _write(ex / f"heldout_case_{no}.json", result)
        print("heldout", no, result.get("recommended_action"), result.get("predicted_grade"))


def main() -> None:
    cfg = load_config()
    generate_production_examples(cfg)
    generate_heldout_examples(cfg, n=3)


if __name__ == "__main__":
    main()
