#!/usr/bin/env python3
"""Held-out routing evidence: missingness / OOD / disagreement / Critic / abstain."""

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

from rockburst.config import load_config, resolve_path
from rockburst.graph.runner import run_graph
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import fit_marginal_thresholds


def main() -> None:
    cfg = load_config()
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    # Relabel stale production showcase files
    for p in ex.glob("case_*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("example_protocol", "").startswith("production"):
            d["stale_note"] = (
                "production showcase; case may have been seen in full-data refit"
            )
            p.write_text(json.dumps(d, indent=2, default=str))

    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    meta = json.loads((resolve_path(cfg, "baselines") / "m1_results.json").read_text())
    best = (
        meta.get("nested_unbiased", {}).get("production_model")
        or meta.get("best_model")
        or "catboost__knn"
    )
    name, imp = best.split("__", 1)
    params = meta.get("best_params") or {}
    seed = int(cfg.get("seed", 42))
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

    wanted = {
        "missingness": None,
        "ood": None,
        "disagreement": None,
        "critic": None,
        "abstain": None,
        "failure": None,
    }
    heldout = []
    for j in te:
        case = df.iloc[int(j)].to_dict()
        ref = build_fold_reference(df, fit_idx, case)
        m7 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
        m6 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=False)
        trace = " ".join(m7.get("route_trace") or [])
        payload = {
            **m7,
            "example_protocol": "held-out fold model",
            "true_grade": int(case["rockburst_grade"]),
            "m6_route_trace": m6.get("route_trace"),
        }
        if len(heldout) < 3:
            heldout.append(payload)
            (ex / f"heldout_case_{int(case['no'])}.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        used = {id(v) for v in wanted.values() if v is not None}
        d_miss = case.get("d") is None or (
            isinstance(case.get("d"), float) and not np.isfinite(float(case.get("d")))
        )
        if (
            wanted["missingness"] is None
            and d_miss
            and "extra_validation" in trace
        ):
            wanted["missingness"] = payload
        elif wanted["ood"] is None and m7.get("ood_status") == "Yes":
            wanted["ood"] = payload
        elif wanted["disagreement"] is None and (
            m7.get("agent_disagreement") == "high"
            or float(m7.get("model_disagreement") or 0) >= float(
                fold_cfg.get("routing", {}).get("disagreement_threshold", 0.35)
            )
        ):
            wanted["disagreement"] = payload
        elif (
            wanted["critic"] is None
            and "critic" in trace
            and m7.get("critic_endorse") is False
        ):
            wanted["critic"] = payload
        elif wanted["abstain"] is None and m7.get("abstained"):
            wanted["abstain"] = payload
        elif wanted["failure"] is None and m7.get("predicted_grade") != int(
            case["rockburst_grade"]
        ):
            wanted["failure"] = {
                **payload,
                "failure_analysis": {
                    "true_grade": int(case["rockburst_grade"]),
                    "predicted_grade": m7.get("predicted_grade"),
                    "diagnosis": "Wrong prediction on held-out fold model",
                },
            }
        _ = used
        if all(v is not None for v in wanted.values()) and len(heldout) >= 3:
            break

    for k, v in wanted.items():
        if v is None:
            continue
        (ex / f"trace_{k}.json").write_text(json.dumps(v, indent=2, default=str))
        print("wrote", k, "no", v.get("no") or v.get("true_grade"))
    print("heldout_n", len(heldout))


if __name__ == "__main__":
    main()
