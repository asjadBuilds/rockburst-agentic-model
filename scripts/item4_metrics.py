#!/usr/bin/env python3
"""Item 4: Critic AUROC, risk-coverage, per-class, confusion — one full OOF pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import (
    fuse_assessment,
    run_critic,
    run_data_quality,
    run_physics,
    run_predict,
    run_uncertainty,
)
from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import (
    classification_report_dict,
    quadratic_weighted_kappa,
    severe_misclassification_rate,
)
from rockburst.eval.ablation import _risk_coverage_curve
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, select_decision_thresholds
from rockburst.graph.runner import run_graph
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import fit_marginal_thresholds, coverage_and_size
from rockburst.uncertainty.estimates import predictive_entropy


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

    # Collectors
    y_all, pred_m1, pred_m7 = [], [], []
    proba_m1, proba_m7 = [], []
    conf_rfinal, conf_maxp, conf_ent = [], [], []
    correct_m1 = []
    critic_flag, rule_flags = [], {k: [] for k in ("physics", "ood", "disagreement", "large_set")}
    sets_m1 = []
    abs_m7 = []

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
        # Fold fusion
        R_tr, ycorr = [], []
        for j in fit_idx[:50]:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
            from rockburst.agents.schemas import PredictiveOutput

            pred = PredictiveOutput(
                predicted_grade=int(proba.argmax()),
                probabilities=proba.tolist(),
                r_model=float(proba.max()),
            )
            dq = run_data_quality(case, ref, cfg)
            phys = run_physics(case, cfg)
            unc = run_uncertainty(case, pred, cfg, pipeline=pipe, conformal=conf)
            R_tr.append(compute_reliability_features(dq, phys, pred, unc))
            ycorr.append(int(pred.predicted_grade == int(y[j])))
        fusion = ReliabilityFusion().fit(np.asarray(R_tr), np.asarray(ycorr))
        if fusion.is_fit:
            r_cal = []
            y_cal = []
            for j in calib_idx[:40]:
                case = df.iloc[int(j)].to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                pred = run_predict(case, cfg, pipeline=pipe)
                dq = run_data_quality(case, ref, cfg)
                phys = run_physics(case, cfg)
                unc = run_uncertainty(case, pred, cfg, pipeline=pipe, conformal=conf)
                r_cal.append(fusion.r_final(compute_reliability_features(dq, phys, pred, unc)))
                y_cal.append(int(pred.predicted_grade == int(y[j])))
            if r_cal:
                fusion.set_decision_thresholds(
                    select_decision_thresholds(np.asarray(r_cal), np.asarray(y_cal))
                )
        fpath = resolve_path(cfg, "models") / "fusion.joblib"
        backup = fpath.read_bytes() if fpath.exists() else None
        fusion.save(fpath)
        fold_cfg = {
            **cfg,
            "_fold_pipeline": pipe,
            "_fold_model_name": best_name,
            "_fold_X": X.iloc[fit_idx],
            "_fold_y": y[fit_idx],
            "_fold_alt_pipes": {},
            "_fold_cross_pipes": {},
            "_conformal_state": conf,
            "routing": {**cfg.get("routing", {}), **fusion.decision_thresholds},
            "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
        }
        for j in te:
            case = df.iloc[int(j)].to_dict()
            ref = build_fold_reference(df, fit_idx, case)
            pred = run_predict(case, fold_cfg, pipeline=pipe, model_name=best_name)
            phys = run_physics(case, fold_cfg)
            unc = run_uncertainty(
                case, pred, fold_cfg, pipeline=pipe, conformal=conf
            )
            dq = run_data_quality(case, ref, fold_cfg)
            yt = int(y[j])
            err = int(pred.predicted_grade != yt)
            # Critic rules (first-pass, before graph)
            crit = run_critic(dq, phys, pred, unc, fold_cfg)
            fired = not crit.endorse
            critic_flag.append(int(fired))
            rule_flags["physics"].append(
                int(phys.physics_score < 0.6 and pred.r_model > 0.8)
            )
            rule_flags["ood"].append(int(bool(dq.ood_flag) and pred.r_model > 0.75))
            rule_flags["disagreement"].append(
                int(unc.model_disagreement >= fold_cfg["routing"].get("disagreement_threshold", 0.35))
            )
            rule_flags["large_set"].append(
                int(len(unc.prediction_set) >= fold_cfg["routing"].get("large_set_size", 3))
            )
            m7 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
            y_all.append(yt)
            pred_m1.append(pred.predicted_grade)
            pred_m7.append(int(m7.get("predicted_grade") if m7.get("predicted_grade") is not None else pred.predicted_grade))
            proba_m1.append(pred.probabilities)
            proba_m7.append(m7.get("probabilities") or pred.probabilities)
            rvec = m7.get("r_vector") or {}
            conf_rfinal.append(float(rvec.get("r_final") or m7.get("final_system_confidence") or 0))
            conf_maxp.append(float(max(pred.probabilities)))
            conf_ent.append(float(1.0 - predictive_entropy(np.asarray(pred.probabilities)) / np.log(4)))
            correct_m1.append(1 - err)
            sets_m1.append(list(unc.prediction_set))
            abs_m7.append(int(bool(m7.get("abstained"))))
        if backup is not None:
            fpath.write_bytes(backup)

    y_all = np.asarray(y_all)
    pred_m1 = np.asarray(pred_m1)
    pred_m7 = np.asarray(pred_m7)
    correct = np.asarray(correct_m1)
    flag = np.asarray(critic_flag)
    err = 1 - correct

    def _flag_metrics(f):
        f = np.asarray(f)
        if f.sum() == 0 or f.sum() == len(f):
            auroc = None
        else:
            try:
                auroc = float(roc_auc_score(err, f))
            except ValueError:
                auroc = None
        prec, rec, _, _ = precision_recall_fscore_support(
            err, f, average="binary", zero_division=0
        )
        return {
            "error_rate_flagged": float(err[f == 1].mean()) if f.sum() else None,
            "error_rate_not_flagged": float(err[f == 0].mean()) if (f == 0).sum() else None,
            "auroc": auroc,
            "precision": float(prec),
            "recall": float(rec),
            "flag_rate": float(f.mean()),
        }

    rc_r = _risk_coverage_curve(np.asarray(conf_rfinal), correct.astype(float))
    rc_m = _risk_coverage_curve(np.asarray(conf_maxp), correct.astype(float))
    rc_e = _risk_coverage_curve(np.asarray(conf_ent), correct.astype(float))

    cm_m1 = classification_report_dict(y_all, pred_m1, np.asarray(proba_m1))
    cm_m7 = classification_report_dict(y_all, pred_m7, np.asarray(proba_m7))
    # row-normalised CM
    def _rownorm(cm):
        cm = np.asarray(cm, dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        return (cm / row).tolist()

    cov = coverage_and_size(y_all, sets_m1)
    # per-class conformal coverage
    per_class = {}
    for c in range(4):
        m = y_all == c
        if not m.any():
            continue
        covered = [int(y_all[i] in sets_m1[i]) for i in np.where(m)[0]]
        per_class[str(c)] = {
            "coverage": float(np.mean(covered)),
            "n": int(m.sum()),
            "mean_size": float(np.mean([len(sets_m1[i]) for i in np.where(m)[0]])),
        }

    severe_per = {}
    for c in range(4):
        m = y_all == c
        if m.any():
            severe_per[str(c)] = float(
                (np.abs(y_all[m] - pred_m1[m]) >= 2).mean()
            )

    out = {
        "protocol": "5-fold grouped OOF, seed 42, production mode, no --quick; Item 4 metrics",
        "critic_error_detection": {
            "overall_flag": _flag_metrics(flag),
            "per_rule": {k: _flag_metrics(v) for k, v in rule_flags.items()},
        },
        "risk_coverage": {
            "R_final": rc_r,
            "max_probability": rc_m,
            "ensemble_entropy_inv": rc_e,
            "R_final_beats_maxprob_area": (
                rc_r["area"] is not None
                and rc_m["area"] is not None
                and rc_r["area"] > rc_m["area"]
            ),
        },
        "per_class": {
            "M1_recall": cm_m1.get("class_recall"),
            "M7_recall": cm_m7.get("class_recall"),
            "conformal_coverage_per_class": per_class,
            "severe_error_overall_M1": severe_misclassification_rate(y_all, pred_m1),
            "severe_error_per_class_M1": severe_per,
            "M1_confusion_counts": cm_m1.get("confusion_matrix"),
            "M1_confusion_row_norm": _rownorm(cm_m1.get("confusion_matrix")),
            "M7_confusion_counts": cm_m7.get("confusion_matrix"),
            "M7_confusion_row_norm": _rownorm(cm_m7.get("confusion_matrix")),
            "M1_qwk": quadratic_weighted_kappa(y_all, pred_m1),
            "M7_qwk": quadratic_weighted_kappa(y_all, pred_m7),
            "M7_abstention": float(np.mean(abs_m7)),
        },
        "conformal_overall": cov,
    }
    dest = resolve_path(cfg, "baselines") / "item4_metrics.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "critic_auroc": out["critic_error_detection"]["overall_flag"]["auroc"],
        "rc_areas": {
            "R_final": rc_r["area"],
            "max_p": rc_m["area"],
            "beats": out["risk_coverage"]["R_final_beats_maxprob_area"],
        },
        "severe_overall": out["per_class"]["severe_error_overall_M1"],
        "M7_abs": out["per_class"]["M7_abstention"],
    }, indent=2))


if __name__ == "__main__":
    main()
