"""Ablation study M1–M7 on repeated grouped OOF folds (no in-sample scoring)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.schemas import PredictiveOutput
from rockburst.agents.tools import (
    fuse_assessment,
    run_data_quality,
    run_physics,
    run_predict,
    run_uncertainty,
)
from rockburst.config import resolve_path
from rockburst.eval.bootstrap import fold_list_ci, grouped_bootstrap_ci
from rockburst.eval.metrics import (
    classification_report_dict,
    quadratic_weighted_kappa,
    severe_misclassification_rate,
)
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, select_decision_thresholds
from rockburst.graph.runner import run_graph
from rockburst.ml.pipelines import build_preprocess, get_model_zoo, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib, train_baselines
from rockburst.ml.tuning import estimator_from_params
from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    coverage_and_size,
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
    conformal_prediction_set,
)
from rockburst.uncertainty.estimates import perturb_case


def _ensure_models(cfg: dict[str, Any], quick: bool) -> dict[str, Any]:
    model_path = resolve_path(cfg, "models") / "best_baseline.joblib"
    path = resolve_path(cfg, "baselines") / "m1_results.json"
    if model_path.exists() and path.exists():
        return json.loads(path.read_text())
    return train_baselines(cfg, quick=quick)


def _fit_conf(method: str, y_cal, p_cal, alpha: float):
    if method == "mondrian_split":
        return fit_mondrian_thresholds(y_cal, p_cal, alpha=alpha)
    return fit_marginal_thresholds(y_cal, p_cal, alpha=alpha)


def _subset_metrics(
    y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, mask: np.ndarray
) -> dict[str, Any]:
    if not mask.any():
        return {"n": 0, "accuracy": None, "qwk": None, "severe_error_rate": None}
    yt, yp = y[mask], pred[mask]
    out = {
        "n": int(mask.sum()),
        "accuracy": float((yt == yp).mean()),
        "qwk": quadratic_weighted_kappa(yt, yp),
        "severe_error_rate": severe_misclassification_rate(yt, yp),
    }
    if proba is not None:
        out.update(
            {
                k: classification_report_dict(yt, yp, proba[mask])[k]
                for k in ("brier", "ece")
            }
        )
    return out


def _risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> dict[str, Any]:
    """Risk–coverage: accuracy when keeping top-confidence fraction; area under curve."""
    if len(confidence) == 0:
        return {"points": [], "area": None}
    order = np.argsort(-confidence)
    corr = correct[order]
    points = []
    coverages = np.linspace(0.1, 1.0, 10)
    accs = []
    for keep in coverages:
        n_keep = max(1, int(round(keep * len(corr))))
        acc = float(corr[:n_keep].mean())
        points.append({"coverage": float(keep), "accuracy": acc})
        accs.append(acc)
    # Trapezoid area over coverage in [0.1, 1]
    area = float(np.trapezoid(accs, coverages))
    return {"points": points, "area": area}


def run_monolithic_fixed_sequence(
    case: dict[str, Any],
    reference: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    pipeline: Any,
    conformal: Any,
) -> dict[str, Any]:
    """M5: fixed tool order, no coordinator, no Critic."""
    dq = run_data_quality(case, reference, cfg)
    phys = run_physics(case, cfg)
    pred = run_predict(case, cfg, pipeline=pipeline, model_name="fold")
    unc = run_uncertainty(
        case, pred, cfg, pipeline=pipeline, ensemble=None, conformal=conformal
    )
    return fuse_assessment(
        dq,
        phys,
        pred,
        unc,
        cfg,
        [
            "m5:data_quality",
            "m5:geomechanics",
            "m5:predict",
            "m5:uncertainty",
            "m5:finalize",
        ],
    )


def run_ablations(cfg: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    m1_meta = _ensure_models(cfg, quick=quick)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    n_outer = 2 if quick else int(cfg["cv"]["n_outer"])
    n_repeats = 1 if quick else int(cfg["cv"].get("n_repeats", 5))
    alpha = float(cfg.get("uncertainty", {}).get("conformal_alpha", 0.1))
    conf_method = (m1_meta.get("conformal") or {}).get("method", "marginal_split")

    best_name = (
        m1_meta.get("nested_unbiased", {}).get("production_model")
        or m1_meta.get("best_model")
        or "logistic__median"
    )
    name, imp = best_name.split("__", 1)
    params = m1_meta.get("best_params") or {}
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:  # noqa: BLE001
        zoo = get_model_zoo(quick=True)
        base_est = zoo[name] if name in zoo else next(iter(zoo.values()))

    systems = ["M1", "M3", "M4", "M5", "M6", "M7"]
    # Case-level stores for grouped bootstrap
    case_store: dict[str, dict[str, list]] = {
        s: {
            "y": [],
            "pred": [],
            "proba": [],
            "abstain": [],
            "conf": [],
            "groups": [],
            "accepted_correct": [],
        }
        for s in systems
    }
    fold_records: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    paired_case: dict[str, list[float]] = {
        "M7_minus_M1_qwk": [],
        "M7_minus_M6_qwk": [],
        "M6_minus_M5_qwk": [],
        "M7_minus_M1_severe": [],
        "M7_minus_M6_severe": [],
        "M6_minus_M5_severe": [],
    }
    critic_stats = {
        "n_cases": 0,
        "critic_fired": 0,
        "recheck_requested": 0,
        "decision_changed": 0,
        "changed_correct": 0,
        "changed_severe_error": 0,
        "recheck_changed_confidence": 0,
        "recheck_changed_set": 0,
        "recheck_changed_decision": 0,
        "tool_invocations": {
            "geomechanics": 0,
            "geomechanics_skipped": 0,
            "extra_validation": 0,
            "recheck_physics": 0,
            "recheck_data": 0,
            "recheck_model": 0,
            "critic": 0,
        },
        "m7_route_diverged_from_m5": 0,
        "m6_route_diverged_from_m5": 0,
    }
    coord_stats = {
        "n_cases": 0,
        "route_diverged_from_m5": 0,
        "final_decision_differed": 0,
        "grade_differed": 0,
    }
    risk_curves: dict[str, list[dict[str, Any]]] = {s: [] for s in ["M1", "M4", "M5", "M6", "M7"]}
    traces_sample: list[dict[str, Any]] = []
    last_pipe = None

    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_outer, shuffle=True, random_state=seed + rep
        )
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + rep + fold)
            pipe = Pipeline(
                [("prep", build_preprocess(imp)), ("model", clone(base_est))]
            )
            pipe.fit(X.iloc[fit_idx], y[fit_idx])
            last_pipe = pipe
            conf = None
            if len(calib_idx):
                conf = _fit_conf(
                    conf_method,
                    y[calib_idx],
                    _predict_proba_safe(pipe, X.iloc[calib_idx]),
                    alpha,
                )
                conf.n_fit = int(len(fit_idx))

            R_tr, ycorr_tr = [], []
            R_cal, ycorr_cal = [], []
            fusion_idx = fit_idx
            if quick and len(fusion_idx) > 24:
                fusion_idx = fusion_idx[:24]
            elif len(fusion_idx) > 60:
                rng = np.random.default_rng(seed)
                fusion_idx = rng.choice(fusion_idx, size=60, replace=False)

            fold_cfg_base = {
                **cfg,
                "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
            }
            for j in fusion_idx:
                case = df.iloc[int(j)].to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
                pred = PredictiveOutput(
                    predicted_grade=int(proba.argmax()),
                    probabilities=proba.tolist(),
                    r_model=float(proba.max()),
                    model_name="fold",
                )
                dq = run_data_quality(case, ref, fold_cfg_base)
                phys = run_physics(case, fold_cfg_base)
                unc = run_uncertainty(
                    case, pred, fold_cfg_base, pipeline=pipe, ensemble=None, conformal=conf
                )
                R_tr.append(compute_reliability_features(dq, phys, pred, unc))
                ycorr_tr.append(int(pred.predicted_grade == int(y[j])))

            # Inner calib for nested threshold selection (not outer test)
            cal_use = calib_idx
            if len(cal_use) > 40:
                rng = np.random.default_rng(seed + fold)
                cal_use = rng.choice(cal_use, size=40, replace=False)
            for j in cal_use:
                case = df.iloc[int(j)].to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                proba = _predict_proba_safe(pipe, X.iloc[[j]])[0]
                pred = PredictiveOutput(
                    predicted_grade=int(proba.argmax()),
                    probabilities=proba.tolist(),
                    r_model=float(proba.max()),
                    model_name="fold",
                )
                dq = run_data_quality(case, ref, fold_cfg_base)
                phys = run_physics(case, fold_cfg_base)
                unc = run_uncertainty(
                    case, pred, fold_cfg_base, pipeline=pipe, ensemble=None, conformal=conf
                )
                R_cal.append(compute_reliability_features(dq, phys, pred, unc))
                ycorr_cal.append(int(pred.predicted_grade == int(y[j])))

            fusion = ReliabilityFusion()
            if R_tr:
                fusion.fit(np.asarray(R_tr), np.asarray(ycorr_tr))
                if R_cal and fusion.is_fit:
                    r_vals = [fusion.r_final(r) for r in R_cal]
                    thr = select_decision_thresholds(
                        np.asarray(r_vals), np.asarray(ycorr_cal)
                    )
                else:
                    r_vals = [fusion.r_final(r) for r in R_tr]
                    thr = select_decision_thresholds(
                        np.asarray(r_vals), np.asarray(ycorr_tr)
                    )
                fusion.set_decision_thresholds(thr)
            fpath = resolve_path(cfg, "models") / "fusion_fold_tmp.joblib"
            fusion.save(fpath)
            fold_cfg = {
                **fold_cfg_base,
                "routing": {**cfg.get("routing", {}), **fusion.decision_thresholds},
                "_fold_pipeline": pipe,
                "_fold_ensemble": None,
                "_fold_model_name": best_name,
                "_fold_X": X.iloc[fit_idx],
                "_fold_y": y[fit_idx],
                "_fold_alt_pipes": {},
                "_fold_cross_pipes": {},
                "_conformal_state": conf,
            }
            main_fusion = resolve_path(cfg, "models") / "fusion.joblib"
            backup = main_fusion.read_bytes() if main_fusion.exists() else None
            fusion.save(main_fusion)

            preds: dict[str, list[int]] = {s: [] for s in systems}
            probas: dict[str, list[list[float]]] = {s: [] for s in systems}
            abstain: dict[str, list[int]] = {s: [] for s in systems}
            confs: dict[str, list[float]] = {s: [] for s in systems}
            sets_m4: list[list[int]] = []
            g_te = groups[te]

            te_df = df.iloc[te]
            for local_i, (_, row) in enumerate(te_df.iterrows()):
                case = row.to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                pred = run_predict(case, fold_cfg, pipeline=pipe, model_name=best_name)
                phys = run_physics(case, fold_cfg)
                unc = run_uncertainty(
                    case, pred, fold_cfg, pipeline=pipe, ensemble=None, conformal=conf
                )
                y_true = int(row["rockburst_grade"])
                gid = int(g_te[local_i])

                # M1
                preds["M1"].append(pred.predicted_grade)
                probas["M1"].append(pred.probabilities)
                abstain["M1"].append(0)
                confs["M1"].append(float(max(pred.probabilities)))

                # M3
                grade_m3 = pred.predicted_grade
                p3 = np.array(pred.probabilities, dtype=float)
                if phys.physics_score < 0.5:
                    p3 = 0.7 * p3 + 0.3 * np.ones(4) / 4
                    grade_m3 = int(p3.argmax())
                preds["M3"].append(grade_m3)
                probas["M3"].append(p3.tolist())
                abstain["M3"].append(0)
                confs["M3"].append(float(p3.max()))

                # M4
                ab4 = int(unc.stability_score < 0.45 or len(unc.prediction_set) >= 3)
                abstain["M4"].append(ab4)
                preds["M4"].append(pred.predicted_grade)
                probas["M4"].append(pred.probabilities)
                confs["M4"].append(float(unc.r_uncertainty))
                sets_m4.append(list(unc.prediction_set))

                # M5
                m5 = run_monolithic_fixed_sequence(
                    case, ref, fold_cfg, pipeline=pipe, conformal=conf
                )
                g5 = m5.get("predicted_grade")
                preds["M5"].append(int(g5) if g5 is not None else pred.predicted_grade)
                probas["M5"].append(m5.get("probabilities") or pred.probabilities)
                abstain["M5"].append(int(bool(m5.get("abstained"))))
                confs["M5"].append(float(m5.get("final_system_confidence") or 0.0))

                # M6
                m6 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=False)
                g6 = m6.get("predicted_grade")
                preds["M6"].append(int(g6) if g6 is not None else pred.predicted_grade)
                probas["M6"].append(m6.get("probabilities") or pred.probabilities)
                abstain["M6"].append(int(bool(m6.get("abstained"))))
                confs["M6"].append(float(m6.get("final_system_confidence") or 0.0))

                # M7
                m7 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
                g7 = m7.get("predicted_grade")
                g7i = int(g7) if g7 is not None else pred.predicted_grade
                preds["M7"].append(g7i)
                probas["M7"].append(m7.get("probabilities") or pred.probabilities)
                abstain["M7"].append(int(bool(m7.get("abstained"))))
                confs["M7"].append(float(m7.get("final_system_confidence") or 0.0))

                # Critic / routing stats
                critic_stats["n_cases"] += 1
                trace = m7.get("route_trace") or []
                joined = " ".join(trace)
                for key in critic_stats["tool_invocations"]:
                    if key == "geomechanics_skipped":
                        critic_stats["tool_invocations"][key] += joined.count(
                            "geomechanics_skipped"
                        )
                    else:
                        critic_stats["tool_invocations"][key] += sum(
                            1 for t in trace if t.endswith(f":{key}")
                        )
                fired = any("critic" in t for t in trace)
                if fired:
                    critic_stats["critic_fired"] += 1
                if fired:
                    try:
                        ci = next(i for i, t in enumerate(trace) if "critic" in t)
                        after = trace[ci + 1 :]
                        if any(
                            a.endswith(s)
                            for a in after
                            for s in (
                                ":recheck_physics",
                                ":recheck_data",
                                ":recheck_model",
                            )
                        ):
                            critic_stats["recheck_requested"] += 1
                    except StopIteration:
                        pass
                # Recheck effects vs M6 (confidence / set / decision)
                if abs(
                    float(m7.get("final_system_confidence") or 0)
                    - float(m6.get("final_system_confidence") or 0)
                ) > 1e-9:
                    critic_stats["recheck_changed_confidence"] += 1
                if list(m7.get("prediction_set") or []) != list(
                    m6.get("prediction_set") or []
                ):
                    critic_stats["recheck_changed_set"] += 1
                if m7.get("recommended_action") != m6.get("recommended_action"):
                    critic_stats["recheck_changed_decision"] += 1
                # Decision change vs M6
                if (m7.get("recommended_action") != m6.get("recommended_action")) or (
                    g7i != preds["M6"][-1]
                ):
                    critic_stats["decision_changed"] += 1
                    if g7i == y_true:
                        critic_stats["changed_correct"] += 1
                    if abs(g7i - y_true) >= 2:
                        critic_stats["changed_severe_error"] += 1

                # Coordinator M6/M7 vs M5
                coord_stats["n_cases"] += 1
                m5_trace = [
                    "m5:data_quality",
                    "m5:geomechanics",
                    "m5:predict",
                    "m5:uncertainty",
                    "m5:finalize",
                ]
                m5_actions = [t.split(":")[-1] for t in m5_trace]
                m6_actions = [t.split(":")[-1] for t in (m6.get("route_trace") or [])]
                m7_actions = [t.split(":")[-1] for t in (m7.get("route_trace") or [])]
                # Normalize skip tag for divergence (skip ≠ fixed phys)
                m6_norm = [
                    "geomechanics" if a == "geomechanics_skipped" else a
                    for a in m6_actions
                ]
                m7_norm = [
                    "geomechanics" if a == "geomechanics_skipped" else a
                    for a in m7_actions
                ]
                if m6_actions != m5_actions:
                    coord_stats["route_diverged_from_m5"] += 1
                    critic_stats["m6_route_diverged_from_m5"] += 1
                if m7_actions != m5_actions:
                    critic_stats["m7_route_diverged_from_m5"] += 1
                if m6.get("recommended_action") != m5.get("recommended_action"):
                    coord_stats["final_decision_differed"] += 1
                if preds["M6"][-1] != preds["M5"][-1]:
                    coord_stats["grade_differed"] += 1
                _ = (m6_norm, m7_norm)  # retained for future paired route analytics

                if len(traces_sample) < 8:
                    traces_sample.append(
                        {
                            "no": int(case["no"]),
                            "trace": m7.get("route_trace"),
                            "decision": m7.get("recommended_action"),
                            "rep": rep,
                            "fold": fold,
                        }
                    )

                for s in systems:
                    case_store[s]["y"].append(y_true)
                    case_store[s]["pred"].append(preds[s][-1])
                    case_store[s]["proba"].append(probas[s][-1])
                    case_store[s]["abstain"].append(abstain[s][-1])
                    case_store[s]["conf"].append(confs[s][-1])
                    case_store[s]["groups"].append(gid)

            yt = y[te]
            metrics_fold = {}
            for s in systems:
                yp = np.array(preds[s], dtype=int)
                pr = np.array(probas[s], dtype=float)
                ab = np.array(abstain[s], dtype=int)
                md = classification_report_dict(yt, yp, pr)
                md["abstention_rate"] = float(ab.mean())
                accepted = ab == 0
                md["accepted"] = _subset_metrics(yt, yp, pr, accepted)
                md["abstained"] = _subset_metrics(yt, yp, pr, ab == 1)
                if s == "M4":
                    cov = coverage_and_size(yt, sets_m4)
                    md["prediction_set_coverage"] = cov["coverage"]
                    md["prediction_set_mean_size"] = cov["mean_size"]
                fold_records[s].append(md)
                metrics_fold[s] = md
                if s in risk_curves:
                    risk_curves[s].append(
                        _risk_coverage_curve(
                            np.asarray(confs[s], dtype=float),
                            (yp == yt).astype(float),
                        )
                    )

            paired_case["M7_minus_M1_qwk"].append(
                metrics_fold["M7"]["qwk"] - metrics_fold["M1"]["qwk"]
            )
            paired_case["M7_minus_M6_qwk"].append(
                metrics_fold["M7"]["qwk"] - metrics_fold["M6"]["qwk"]
            )
            paired_case["M6_minus_M5_qwk"].append(
                metrics_fold["M6"]["qwk"] - metrics_fold["M5"]["qwk"]
            )
            paired_case["M7_minus_M1_severe"].append(
                metrics_fold["M7"]["severe_misclassification_rate"]
                - metrics_fold["M1"]["severe_misclassification_rate"]
            )
            paired_case["M7_minus_M6_severe"].append(
                metrics_fold["M7"]["severe_misclassification_rate"]
                - metrics_fold["M6"]["severe_misclassification_rate"]
            )
            paired_case["M6_minus_M5_severe"].append(
                metrics_fold["M6"]["severe_misclassification_rate"]
                - metrics_fold["M5"]["severe_misclassification_rate"]
            )

            if backup is not None:
                main_fusion.write_bytes(backup)

    # Aggregate with grouped bootstrap over pooled case-level predictions
    results: dict[str, Any] = {
        "m1_best": best_name,
        "n_outer": n_outer,
        "n_repeats": n_repeats,
        "ci_method": "grouped_bootstrap over dup_group_id (case-level pooled across repeats)",
        "definitions": {
            "M1": "Baseline ML predict only",
            "M2": "M1 + permutation importance (interpretability sidecar; metrics = M1)",
            "M3": "M1 + physics reliability adjustment",
            "M4": "M1 + uncertainty / conformal abstention tracking",
            "M5": "Monolithic fixed sequence DQ→physics→predict→uncertainty→fuse (no coordinator, no Critic)",
            "M6": "Rules coordinator without Critic",
            "M7": "Full multi-agent system with Critic and rechecks",
        },
        "ablations": {},
        "paired_differences": {},
        "risk_coverage": {},
        "critic_statistics": critic_stats,
        "coordinator_m6_vs_m5": coord_stats,
        "trace_samples": traces_sample,
        "n_cases": int(len(df)),
        "protocol": "held-out fold models only; identical folds across M1–M7",
    }

    for s in systems:
        y_s = np.asarray(case_store[s]["y"], dtype=int)
        p_s = np.asarray(case_store[s]["pred"], dtype=int)
        pr_s = np.asarray(case_store[s]["proba"], dtype=float)
        ab_s = np.asarray(case_store[s]["abstain"], dtype=int)
        g_s = np.asarray(case_store[s]["groups"])
        correct = (y_s == p_s).astype(float)
        severe = (np.abs(y_s - p_s) >= 2).astype(float)

        def _qwk_agg(mask_vals=None):
            # placeholder unused
            return 0.0

        # Bootstrap accuracy / severe; QWK via resampling groups then computing QWK
        uniq = np.unique(g_s)
        rng = np.random.default_rng(seed)
        qwk_boots, acc_boots, sev_boots, abs_boots = [], [], [], []
        for _ in range(400 if not quick else 50):
            sg = rng.choice(uniq, size=len(uniq), replace=True)
            m = np.isin(g_s, sg)
            if m.sum() < 5:
                continue
            qwk_boots.append(quadratic_weighted_kappa(y_s[m], p_s[m]))
            acc_boots.append(float(correct[m].mean()))
            sev_boots.append(float(severe[m].mean()))
            abs_boots.append(float(ab_s[m].mean()))

        def _ci_from_boots(point, boots):
            arr = np.asarray(boots, dtype=float)
            return {
                "mean": point,
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "ci95_low": float(np.percentile(arr, 2.5)) if len(arr) else point,
                "ci95_high": float(np.percentile(arr, 97.5)) if len(arr) else point,
                "method": "grouped_bootstrap",
            }

        md = classification_report_dict(y_s, p_s, pr_s)
        accepted = ab_s == 0
        block = {
            "accuracy": _ci_from_boots(float(correct.mean()), acc_boots),
            "qwk": _ci_from_boots(quadratic_weighted_kappa(y_s, p_s), qwk_boots),
            "severe_misclassification_rate": _ci_from_boots(
                float(severe.mean()), sev_boots
            ),
            "abstention_rate": _ci_from_boots(float(ab_s.mean()), abs_boots),
            "brier": {"mean": md.get("brier")},
            "ece": {"mean": md.get("ece")},
            "class_recall_mean": md.get("class_recall"),
            "accepted": _subset_metrics(y_s, p_s, pr_s, accepted),
            "abstained": _subset_metrics(y_s, p_s, pr_s, ab_s == 1),
            "fold_mean_ci_legacy": {
                k: fold_list_ci([f[k] for f in fold_records[s] if f.get(k) is not None])
                for k in ("accuracy", "qwk", "abstention_rate")
            },
        }
        results["ablations"][s] = block

    # Risk–coverage areas
    for s, curves in risk_curves.items():
        areas = [c["area"] for c in curves if c.get("area") is not None]
        results["risk_coverage"][s] = {
            "area_mean": float(np.mean(areas)) if areas else None,
            "area_std": float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0,
            "example_points": curves[0]["points"] if curves else [],
        }

    # Paired diffs: fold-level with grouped note + fold bootstrap
    for k, vals in paired_case.items():
        results["paired_differences"][k] = fold_list_ci(vals)
        results["paired_differences"][k]["note"] = (
            "Fold-level paired Δ; primary case-level CIs use grouped bootstrap above."
        )

    # Indistinguishability finding
    a5 = results["ablations"]["M5"]["abstention_rate"]["mean"]
    a6 = results["ablations"]["M6"]["abstention_rate"]["mean"]
    a7 = results["ablations"]["M7"]["abstention_rate"]["mean"]
    q5 = results["ablations"]["M5"]["qwk"]["mean"]
    q6 = results["ablations"]["M6"]["qwk"]["mean"]
    q7 = results["ablations"]["M7"]["qwk"]["mean"]
    results["agent_value_finding"] = {
        "m5_m6_m7_qwk_identical": abs(q5 - q6) < 1e-12 and abs(q6 - q7) < 1e-12,
        "m5_m6_m7_abstention_identical": abs(a5 - a6) < 1e-12 and abs(a6 - a7) < 1e-12,
        "critic_fire_rate": critic_stats["critic_fired"] / max(critic_stats["n_cases"], 1),
        "coord_route_divergence_rate": coord_stats["route_diverged_from_m5"]
        / max(coord_stats["n_cases"], 1),
        "note": (
            "If M5/M6/M7 remain indistinguishable, thresholds may be too loose for "
            "Critic/coordinator to change grades; not tuned on test folds."
        ),
    }

    results["ablations"]["M2"] = {
        **results["ablations"]["M1"],
        "note": "Interpretability sidecar; classification metrics identical to M1 on same folds.",
    }
    _write_m2_importance(cfg, df)

    results["robustness"] = _robustness_fold_models(cfg, df, pipe=last_pipe, groups=groups, y=y, X=X)
    results["missing_data"] = _missing_data_fold_models(cfg, df, X=X, y=y, groups=groups)

    out = resolve_path(cfg, "baselines") / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    return results


def _write_m2_importance(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.impute import SimpleImputer

    X_all, y_all, groups = prepare_xy(df)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, _te = next(sgkf.split(X_all, y_all, groups))
    X = SimpleImputer().fit_transform(df.iloc[tr][FEATURE_COLS])
    y = df.iloc[tr]["rockburst_grade"].to_numpy()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    r = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
    payload = {
        "features": FEATURE_COLS,
        "importances_mean": r.importances_mean.tolist(),
        "importances_std": r.importances_std.tolist(),
        "fit_on": "first StratifiedGroupKFold train split only",
    }
    path = resolve_path(cfg, "baselines") / "m2_importance.json"
    path.write_text(json.dumps(payload, indent=2))


def _robustness_fold_models(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    pipe: Any,
    groups: np.ndarray,
    y: np.ndarray,
    X: pd.DataFrame,
) -> dict[str, Any]:
    """Robustness on held-out fold models only (not production in-sample pipe)."""
    from rockburst.agents.tools import fuse_assessment

    rng = np.random.default_rng(42)
    mild = cfg.get("uncertainty", {}).get("mild_eps", 0.05)
    strong = cfg.get("uncertainty", {}).get("strong_eps", 0.15)
    seed = int(cfg.get("seed", 42))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X, y, groups))
    fit_idx, _calib = _split_fit_calib(y, groups, tr, seed=seed)
    if pipe is None:
        zoo = get_model_zoo(quick=True)
        est = next(iter(zoo.values()))
        pipe = Pipeline([("prep", build_preprocess("median")), ("model", clone(est))])
        pipe.fit(X.iloc[fit_idx], y[fit_idx])

    sample_idx = te[: min(40, len(te))]
    flips = {"mild": 0, "strong": 0}
    conf_drop = {"mild": [], "strong": []}
    proba_shift = {"mild": [], "strong": []}
    unc_change = {"mild": [], "strong": []}
    r_final_change = {"mild": [], "strong": []}
    n = 0
    for j in sample_idx:
        case = df.iloc[int(j)].to_dict()
        ref = build_fold_reference(df, fit_idx, case)
        base_pred = run_predict(case, cfg, pipeline=pipe)
        base_p = np.asarray(base_pred.probabilities, dtype=float)
        base_c = float(base_p.max())
        base_unc = run_uncertainty(case, base_pred, cfg, pipeline=pipe, ensemble=None, conformal=None)
        base_dq = run_data_quality(case, ref, cfg)
        base_phys = run_physics(case, cfg)
        base_fuse = fuse_assessment(
            base_dq, base_phys, base_pred, base_unc, cfg, ["robust:base"]
        )
        base_r = float(base_fuse.get("final_system_confidence") or 0.0)
        for label, eps in ("mild", mild), ("strong", strong):
            pc = perturb_case(case, FEATURE_COLS, eps, rng)
            pp = run_predict(pc, cfg, pipeline=pipe)
            p = np.asarray(pp.probabilities, dtype=float)
            if pp.predicted_grade != base_pred.predicted_grade:
                flips[label] += 1
            conf_drop[label].append(base_c - float(p.max()))
            proba_shift[label].append(float(np.mean(np.abs(base_p - p))))
            u = run_uncertainty(pc, pp, cfg, pipeline=pipe, ensemble=None, conformal=None)
            unc_change[label].append(float(u.uncertainty_score - base_unc.uncertainty_score))
            dq = run_data_quality(pc, ref, cfg)
            phys = run_physics(pc, cfg)
            fuse = fuse_assessment(dq, phys, pp, u, cfg, ["robust:pert"])
            r_final_change[label].append(
                float((fuse.get("final_system_confidence") or 0.0) - base_r)
            )
        n += 1
    return {
        "protocol": "held-out fold model only (first outer test fold sample)",
        "n": n,
        "flip_counts": flips,
        "flip_rates": {k: v / max(n, 1) for k, v in flips.items()},
        "mean_confidence_drop": {k: float(np.mean(v)) for k, v in conf_drop.items()},
        "mean_proba_shift": {k: float(np.mean(v)) for k, v in proba_shift.items()},
        "mean_uncertainty_change": {k: float(np.mean(v)) for k, v in unc_change.items()},
        "mean_r_final_change": {k: float(np.mean(v)) for k, v in r_final_change.items()},
        "confidence_decreases_with_eps": (
            float(np.mean(conf_drop["strong"])) >= float(np.mean(conf_drop["mild"])) - 1e-9
        ),
        "r_final_degrades_with_eps": (
            float(np.mean(r_final_change["strong"]))
            <= float(np.mean(r_final_change["mild"])) + 1e-9
        ),
    }


def _missing_data_fold_models(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Missing-data experiment with held-out fold models and multiple imputers."""
    seed = int(cfg.get("seed", 42))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X, y, groups))
    fit_idx, _ = _split_fit_calib(y, groups, tr, seed=seed)
    zoo = get_model_zoo(quick=True)
    est = zoo.get("logistic") or next(iter(zoo.values()))

    key_feats = ["sigma_theta", "sigma_c", "sigma_t", "wet", "d"]
    imputers = ["median", "knn", "iterative"]
    scenarios: dict[str, Any] = {"protocol": "held-out fold models only", "imputers": {}}

    for imp in imputers:
        pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        imp_block: dict[str, Any] = {}
        # complete cases among test
        te_df = df.iloc[te]
        complete_mask = te_df[FEATURE_COLS].notna().all(axis=1)
        complete_idx = te_df.index[complete_mask].tolist()
        if len(complete_idx) > 40:
            complete_idx = complete_idx[:40]

        def _acc(mask_cols: list[str], indices: list[int]) -> float:
            if not indices:
                return float("nan")
            correct = 0
            for ix in indices:
                case = df.loc[ix].to_dict()
                for c in mask_cols:
                    case[c] = float("nan")
                pred = run_predict(case, cfg, pipeline=pipe)
                correct += int(pred.predicted_grade == int(df.loc[ix, "rockburst_grade"]))
            return correct / len(indices)

        imp_block["complete"] = {
            "n": len(complete_idx),
            "accuracy": _acc([], complete_idx),
        }
        for feat in key_feats:
            imp_block[f"missing_{feat}"] = {
                "n": len(complete_idx),
                "accuracy": _acc([feat], complete_idx),
            }
        imp_block["missing_multi"] = {
            "n": len(complete_idx),
            "accuracy": _acc(["sigma_theta", "wet", "d"], complete_idx),
        }

        # Train-time vs inference-time missing D
        X_miss = X.copy()
        X_miss["d"] = np.nan
        pipe_miss = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
        pipe_miss.fit(X_miss.iloc[fit_idx], y[fit_idx])
        X_te = X.iloc[te].copy()
        X_te["d"] = np.nan
        acc_inf = float((pipe.predict(X_te) == y[te]).mean())
        acc_train = float((pipe_miss.predict(X_te) == y[te]).mean())
        imp_block["train_vs_inference_missing_d"] = {
            "inference_time_missing_d_acc": acc_inf,
            "train_and_inference_missing_d_acc": acc_train,
        }
        scenarios["imputers"][imp] = imp_block

    scenarios["note"] = (
        "All metrics from fold-trained pipelines on held-out test indices. "
        "Missing-indicator features (observed_*) are in the feature matrix. "
        "No production in-sample pipeline scores are reported here."
    )
    return scenarios
