#!/usr/bin/env python3
"""Recompute item5 QWK CIs + item4 Critic AUROC CIs from saved case vectors.

If npz vectors are missing, rebuild them via fold OOF (no Optuna / no full pipeline).
Does not rewrite unrelated JSON fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import (
    run_critic,
    run_data_quality,
    run_physics,
    run_predict,
    run_uncertainty,
)
from rockburst.config import load_config, resolve_path
from rockburst.eval.bootstrap import grouped_bootstrap_metric
from rockburst.eval.metrics import quadratic_weighted_kappa
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import fit_marginal_thresholds


def _auroc_point(e, f):
    f = np.asarray(f)
    e = np.asarray(e)
    if f.sum() == 0 or f.sum() == len(f):
        return float("nan")
    try:
        return float(roc_auc_score(e, f))
    except ValueError:
        return float("nan")


def _ensure_item5_oof(cfg, m1) -> Path:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "item5_dup_label_noise", ROOT / "scripts" / "item5_dup_label_noise.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    bas = resolve_path(cfg, "baselines")
    path = bas / "item5_last_oof.npz"
    raw = pd.read_csv(resolve_path(cfg, "raw_csv"))
    tols = [0.005, 0.01, 0.02, 0.05]
    save_kw: dict = {}
    item5_path = bas / "item5_dup_label_noise.json"
    out = json.loads(item5_path.read_text())
    by_tol = {float(h["tol"]): h for h in out["headlines_by_tol"]}
    for t in tols:
        h = mod._headline_at_tol(raw, cfg, t, m1)
        blob = h.pop("_last_oof")
        save_kw[f"y_{t}"] = np.asarray(blob["y"])
        save_kw[f"pred_{t}"] = np.asarray(blob["pred"])
        save_kw[f"groups_{t}"] = np.asarray(blob["groups"])
        if blob.get("covered") is not None:
            save_kw[f"covered_{t}"] = np.asarray(blob["covered"])
        # patch CI fields only
        tgt = by_tol[float(t)]
        for k in ("M1_qwk_ci", "M1_accuracy_ci", "conformal_coverage_ci", "note"):
            if k in h:
                tgt[k] = h[k]
    np.savez_compressed(path, **save_kw)
    item5_path.write_text(json.dumps(out, indent=2, default=str))
    return path


def _patch_item5_from_npz(cfg) -> None:
    bas = resolve_path(cfg, "baselines")
    path = bas / "item5_last_oof.npz"
    item5_path = bas / "item5_dup_label_noise.json"
    data = np.load(path)
    out = json.loads(item5_path.read_text())
    seed = int(cfg.get("seed", 42))
    for h in out["headlines_by_tol"]:
        t = h["tol"]
        key = f"y_{t}"
        if key not in data:
            continue
        y = data[key]
        pred = data[f"pred_{t}"]
        g = data[f"groups_{t}"]
        h["M1_qwk_ci"] = grouped_bootstrap_metric(
            y, pred, g, quadratic_weighted_kappa, n_boot=200, seed=seed
        )
        h["note"] = (
            "Fixed catboost__knn, no re-tune; >=5 grouped-CV repeats; "
            "QWK CI = grouped bootstrap of QWK per resample; "
            "acc/coverage CIs = grouped bootstrap of case vectors."
        )
    item5_path.write_text(json.dumps(out, indent=2, default=str))


def _ensure_item4_vectors(cfg, m1) -> Path:
    """Fold OOF + Critic rules only (no LangGraph) to rebuild case vectors."""
    bas = resolve_path(cfg, "baselines")
    path = bas / "item4_case_vectors.npz"
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
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

    err_l, flag_l, g_l = [], [], []
    rules = {k: [] for k in ("physics", "ood", "disagreement", "large_set")}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + fold)
        pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(base_est))])
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        conf = None
        if len(calib_idx):
            conf = fit_marginal_thresholds(
                y[calib_idx], _predict_proba_safe(pipe, X.iloc[calib_idx]), alpha=0.1
            )
        fold_cfg = {
            **cfg,
            "_fold_pipeline": pipe,
            "_fold_model_name": best_name,
            "_conformal_state": conf,
            "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
        }
        for j in te:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            pred = run_predict(case, fold_cfg, pipeline=pipe, model_name=best_name)
            phys = run_physics(case, fold_cfg)
            unc = run_uncertainty(case, pred, fold_cfg, pipeline=pipe, conformal=conf)
            dq = run_data_quality(case, ref, fold_cfg)
            yt = int(y[j])
            err_l.append(int(pred.predicted_grade != yt))
            crit = run_critic(dq, phys, pred, unc, fold_cfg)
            flag_l.append(int(not crit.endorse))
            rules["physics"].append(
                int(phys.physics_score < 0.6 and pred.r_model > 0.8)
            )
            rules["ood"].append(int(bool(dq.ood_flag) and pred.r_model > 0.75))
            rules["disagreement"].append(
                int(
                    unc.model_disagreement
                    >= fold_cfg["routing"].get("disagreement_threshold", 0.35)
                )
            )
            rules["large_set"].append(
                int(len(unc.prediction_set) >= fold_cfg["routing"].get("large_set_size", 3))
            )
            g_l.append(groups[j])
    np.savez_compressed(
        path,
        err=np.asarray(err_l, dtype=int),
        flag=np.asarray(flag_l, dtype=int),
        groups=np.asarray(g_l),
        physics=np.asarray(rules["physics"], dtype=int),
        ood=np.asarray(rules["ood"], dtype=int),
        disagreement=np.asarray(rules["disagreement"], dtype=int),
        large_set=np.asarray(rules["large_set"], dtype=int),
    )
    return path


def _patch_item4_from_npz(cfg) -> None:
    bas = resolve_path(cfg, "baselines")
    data = np.load(bas / "item4_case_vectors.npz")
    out = json.loads((bas / "item4_metrics.json").read_text())
    seed = int(cfg.get("seed", 42))
    err = data["err"]
    g = data["groups"]

    def _ci(f):
        return grouped_bootstrap_metric(err, f, g, _auroc_point, n_boot=200, seed=seed)

    ced = out["critic_error_detection"]
    ced["overall_flag"]["auroc_ci"] = _ci(data["flag"])
    # keep point estimate if present; refresh from vectors if missing
    if ced["overall_flag"].get("auroc") is None:
        v = _auroc_point(err, data["flag"])
        ced["overall_flag"]["auroc"] = None if not np.isfinite(v) else float(v)
    for rule in ("physics", "ood", "disagreement", "large_set"):
        block = ced["per_rule"][rule]
        block["auroc_ci"] = _ci(data[rule])
        if block.get("auroc") is None:
            v = _auroc_point(err, data[rule])
            block["auroc"] = None if not np.isfinite(v) else float(v)
    (bas / "item4_metrics.json").write_text(json.dumps(out, indent=2, default=str))


def main() -> None:
    cfg = load_config()
    bas = resolve_path(cfg, "baselines")
    m1 = json.loads((bas / "m1_results.json").read_text())
    # item5
    if not (bas / "item5_last_oof.npz").exists():
        print("building item5_last_oof.npz + patching QWK CIs...")
        _ensure_item5_oof(cfg, m1)
    else:
        print("patching item5 QWK CIs from npz...")
        _patch_item5_from_npz(cfg)
    # item4
    if not (bas / "item4_case_vectors.npz").exists():
        print("building item4_case_vectors.npz (critic-only OOF)...")
        _ensure_item4_vectors(cfg, m1)
    print("patching item4 Critic AUROC CIs from npz...")
    _patch_item4_from_npz(cfg)
    # one-line summaries
    i5 = json.loads((bas / "item5_dup_label_noise.json").read_text())
    h05 = next(h for h in i5["headlines_by_tol"] if float(h["tol"]) == 0.05)
    i4 = json.loads((bas / "item4_metrics.json").read_text())
    ov = i4["critic_error_detection"]["overall_flag"]
    print(
        "item5 tol0.05 QWK_CI",
        round(h05["M1_qwk_ci"]["ci95_low"], 4),
        round(h05["M1_qwk_ci"]["ci95_high"], 4),
        "mean",
        round(h05["M1_qwk_ci"]["mean"], 4),
    )
    print(
        "critic AUROC_CI",
        ov.get("auroc"),
        ov["auroc_ci"].get("ci95_low"),
        ov["auroc_ci"].get("ci95_high"),
    )


if __name__ == "__main__":
    main()
