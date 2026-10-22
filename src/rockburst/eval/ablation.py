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
    ordinal_mae,
    quadratic_weighted_kappa,
    severe_misclassification_rate,
)
from rockburst.fusion.features import compute_r_model, compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, select_decision_thresholds
from rockburst.graph.runner import run_graph
from rockburst.ml.pipelines import build_preprocess, get_model_zoo, prepare_xy
from rockburst.ml.train import (
    _predict_proba_safe,
    _split_fit_calib,
    fit_fold_ensemble,
    train_baselines,
)
from rockburst.ml.tuning import estimator_from_params
from rockburst.physics.relations import (
    FEATURE_COLS,
    PHYSICAL_MISSING_MASKS,
    apply_missing_mask,
    reconstruct_missing_from_physics,
)
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    coverage_and_size,
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
    conformal_prediction_set,
)
from rockburst.uncertainty.estimates import perturb_case, predictive_entropy


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
    ensemble = cfg.get("_fold_ensemble")
    unc = run_uncertainty(
        case,
        pred,
        cfg,
        pipeline=pipeline,
        ensemble=ensemble,
        conformal=conformal,
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
        "rule_fires": {},
        "m7_route_diverged_from_m5": 0,
        "m6_route_diverged_from_m5": 0,
    }
    _SUBSTANTIVE = frozenset(
        {
            "extra_validation",
            "alternative_imputer",
            "perturbation",
            "cross_check",
            "recheck",
            "recheck_physics",
            "recheck_data",
            "recheck_model",
            "critic",
        }
    )
    coord_stats = {
        "n_cases": 0,
        "route_diverged_from_m5": 0,
        "final_decision_differed": 0,
        "grade_differed": 0,
        "m6_vs_m7_substantive_divergence": 0,
        "divergence_causes": {
            "critic_only": 0,
            "extra_validation_only": 0,
            "recheck_only": 0,
            "other_or_mixed": 0,
        },
    }
    risk_curves: dict[str, list[dict[str, Any]]] = {s: [] for s in ["M1", "M4", "M5", "M6", "M7"]}
    # C4: dual ranking — max-prob vs R_final on the same M7 cases
    risk_curves_maxprob_vs_r: list[dict[str, Any]] = []
    traces_sample: list[dict[str, Any]] = []
    last_pipe = None
    models_meta = m1_meta.get("models") or {}
    ranked = sorted(
        models_meta.items(),
        key=lambda kv: (kv[1].get("pooled_oof") or {}).get("qwk", -1),
        reverse=True,
    )
    if not ranked:
        ranked = [(best_name, {"best_params": params, "fold_params": []})]

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

            fold_ensemble = fit_fold_ensemble(
                X, y, fit_idx, ranked, seed=seed + rep, fold=fold, top_k=3
            )

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
                "_fold_ensemble": fold_ensemble,
            }
            for j in fusion_idx:
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
                    model_name="fold",
                )
                dq = run_data_quality(case, ref, fold_cfg_base)
                phys = run_physics(case, fold_cfg_base)
                unc = run_uncertainty(
                    case,
                    pred,
                    fold_cfg_base,
                    pipeline=pipe,
                    ensemble=fold_ensemble,
                    conformal=conf,
                )
                R_tr.append(compute_reliability_features(dq, phys, pred, unc, case=case))
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
                obs = {
                    c: bool(np.isfinite(float(case[c]))) if case.get(c) is not None else False
                    for c in FEATURE_COLS
                }
                pred = PredictiveOutput(
                    predicted_grade=int(proba.argmax()),
                    probabilities=proba.tolist(),
                    r_model=compute_r_model(proba),
                    observed_mask=obs,
                    model_name="fold",
                )
                dq = run_data_quality(case, ref, fold_cfg_base)
                phys = run_physics(case, fold_cfg_base)
                unc = run_uncertainty(
                    case,
                    pred,
                    fold_cfg_base,
                    pipeline=pipe,
                    ensemble=fold_ensemble,
                    conformal=conf,
                )
                R_cal.append(compute_reliability_features(dq, phys, pred, unc, case=case))
                ycorr_cal.append(int(pred.predicted_grade == int(y[j])))

            fusion = ReliabilityFusion()
            fusion.conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
            if R_tr:
                fusion.fit(np.asarray(R_tr), np.asarray(ycorr_tr))
                if fusion.is_fit:
                    if R_cal:
                        r_raw = np.array([fusion._raw_score(r) for r in R_cal])
                        fusion.fit_calibrator(r_raw, np.asarray(ycorr_cal))
                        r_vals = [fusion.r_final(r) for r in R_cal]
                        thr = select_decision_thresholds(
                            np.asarray(r_vals), np.asarray(ycorr_cal)
                        )
                    else:
                        r_raw = np.array([fusion._raw_score(r) for r in R_tr])
                        fusion.fit_calibrator(r_raw, np.asarray(ycorr_tr))
                        r_vals = [fusion.r_final(r) for r in R_tr]
                        thr = select_decision_thresholds(
                            np.asarray(r_vals), np.asarray(ycorr_tr)
                        )
                    fusion.set_decision_thresholds(thr)
            # Keep fold fusion in memory — do not overwrite production fusion.joblib.
            fold_cfg = {
                **fold_cfg_base,
                "_fold_pipeline": pipe,
                "_fold_ensemble": fold_ensemble,
                "_fold_fusion": fusion,
                "_fold_model_name": best_name,
                "_fold_X": X.iloc[fit_idx],
                "_fold_y": y[fit_idx],
                "_fold_alt_pipes": {},
                "_fold_cross_pipes": {},
                "_conformal_state": conf,
            }

            preds: dict[str, list[int]] = {s: [] for s in systems}
            probas: dict[str, list[list[float]]] = {s: [] for s in systems}
            abstain: dict[str, list[int]] = {s: [] for s in systems}
            confs: dict[str, list[float]] = {s: [] for s in systems}
            sets_m4: list[list[int]] = []
            g_te = groups[te]

            te_df = df.iloc[te]
            fold_maxprob: list[float] = []
            fold_rfinal: list[float] = []
            fold_correct: list[float] = []
            for local_i, (_, row) in enumerate(te_df.iterrows()):
                case = row.to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                pred = run_predict(case, fold_cfg, pipeline=pipe, model_name=best_name)
                phys = run_physics(case, fold_cfg)
                unc = run_uncertainty(
                    case,
                    pred,
                    fold_cfg,
                    pipeline=pipe,
                    ensemble=fold_ensemble,
                    conformal=conf,
                )
                y_true = int(row["rockburst_grade"])
                gid = int(g_te[local_i])

                # M1
                preds["M1"].append(pred.predicted_grade)
                probas["M1"].append(pred.probabilities)
                abstain["M1"].append(0)
                confs["M1"].append(float(max(pred.probabilities)))
                fold_maxprob.append(float(max(pred.probabilities)))

                # M3: physics-criteria agreement adjusts confidence / abstention (≠ M1)
                grade_m3 = pred.predicted_grade
                p3 = np.array(pred.probabilities, dtype=float)
                crit = (phys.empirical_checks or {}).get("criteria_as_given") or {}
                agreement = float(crit.get("agreement") or 0.5)
                cons = crit.get("consensus_grade", -1)
                if phys.physics_score < 0.5:
                    p3 = 0.7 * p3 + 0.3 * np.ones(4) / 4
                    grade_m3 = int(p3.argmax())
                # Soft confidence shrink when criteria disagree; abstain on strong mismatch
                conf3 = float(p3.max()) * (0.45 + 0.55 * agreement)
                ab3 = 0
                if cons is not None and int(cons) >= 0:
                    if abs(int(cons) - int(grade_m3)) >= 2:
                        ab3 = 1
                    elif agreement < 0.5 and phys.physics_score < 0.65:
                        ab3 = 1
                preds["M3"].append(grade_m3)
                probas["M3"].append(p3.tolist())
                abstain["M3"].append(ab3)
                confs["M3"].append(conf3)

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
                fold_rfinal.append(float(m7.get("final_system_confidence") or 0.0))
                fold_correct.append(float(g7i == y_true))

                # Critic / routing stats
                critic_stats["n_cases"] += 1
                for rule in m7.get("critic_rules_fired") or []:
                    critic_stats["rule_fires"][rule] = (
                        int(critic_stats["rule_fires"].get(rule, 0)) + 1
                    )
                trace = m7.get("route_trace") or []
                joined = " ".join(trace)
                for key in critic_stats["tool_invocations"]:
                    if key == "geomechanics_skipped":
                        critic_stats["tool_invocations"][key] += sum(
                            1
                            for t in trace
                            if t.endswith("geomechanics_skipped")
                            and not t.startswith(("rejected:", "fallback:"))
                        )
                    else:
                        # Count executed tool steps only (rules:/llm:), not rejected/fallback tags.
                        critic_stats["tool_invocations"][key] += sum(
                            1
                            for t in trace
                            if t.endswith(f":{key}")
                            and not t.startswith(("rejected:", "fallback:"))
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

                # Coordinator M6/M7 vs M5 (substantive actions only)
                coord_stats["n_cases"] += 1

                def _norm_actions(trace: list[str] | None) -> list[str]:
                    out: list[str] = []
                    for t in trace or []:
                        a = t.split(":")[-1]
                        if a == "geomechanics_skipped":
                            a = "geomechanics"
                        out.append(a)
                    return out

                def _substantive(actions: list[str]) -> set[str]:
                    return {a for a in actions if a in _SUBSTANTIVE}

                m5_actions = [
                    "data_quality",
                    "geomechanics",
                    "predict",
                    "uncertainty",
                    "finalize",
                ]
                m6_norm = _norm_actions(m6.get("route_trace"))
                m7_norm = _norm_actions(m7.get("route_trace"))
                m6_sub = _substantive(m6_norm)
                m7_sub = _substantive(m7_norm)
                m5_sub = _substantive(m5_actions)
                if m6_sub != m5_sub:
                    coord_stats["route_diverged_from_m5"] += 1
                    critic_stats["m6_route_diverged_from_m5"] += 1
                if m7_sub != m5_sub:
                    critic_stats["m7_route_diverged_from_m5"] += 1
                if m6.get("recommended_action") != m5.get("recommended_action"):
                    coord_stats["final_decision_differed"] += 1
                if preds["M6"][-1] != preds["M5"][-1]:
                    coord_stats["grade_differed"] += 1
                if m6_sub != m7_sub:
                    coord_stats["m6_vs_m7_substantive_divergence"] += 1
                    only_m7 = m7_sub - m6_sub
                    only_m6 = m6_sub - m7_sub
                    causes = coord_stats["divergence_causes"]
                    if only_m7 == {"critic"} and not only_m6:
                        causes["critic_only"] += 1
                    elif only_m7 | only_m6 == {"extra_validation"}:
                        causes["extra_validation_only"] += 1
                    elif (only_m7 | only_m6) and (only_m7 | only_m6) <= {
                        "recheck",
                        "recheck_physics",
                        "recheck_data",
                        "recheck_model",
                    }:
                        causes["recheck_only"] += 1
                    else:
                        causes["other_or_mixed"] += 1

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

            risk_curves_maxprob_vs_r.append(
                {
                    "max_prob": _risk_coverage_curve(
                        np.asarray(fold_maxprob, dtype=float),
                        np.asarray(fold_correct, dtype=float),
                    ),
                    "r_final": _risk_coverage_curve(
                        np.asarray(fold_rfinal, dtype=float),
                        np.asarray(fold_correct, dtype=float),
                    ),
                }
            )

    # Aggregate with grouped bootstrap over pooled case-level predictions
    results: dict[str, Any] = {
        "m1_best": best_name,
        "n_outer": n_outer,
        "n_repeats": n_repeats,
        "ci_method": "grouped_bootstrap over dup_group_id (case-level pooled across repeats)",
        "definitions": {
            "M1": "Baseline ML predict only",
            "M2": "M1 + held-out permutation importance (interpretability sidecar; metrics = M1)",
            "M3": "M1 + physics-criteria agreement adjusts confidence/abstention",
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
        "protocol": (
            "held-out fold models only; identical folds across M1–M7; "
            "fold fusion kept in memory (no fusion.joblib overwrite); "
            "fold-trained ensembles passed to uncertainty"
        ),
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

    # C4: max-prob vs R_final on M7 correctness (same cases)
    mp_areas = [
        c["max_prob"]["area"]
        for c in risk_curves_maxprob_vs_r
        if c.get("max_prob", {}).get("area") is not None
    ]
    rf_areas = [
        c["r_final"]["area"]
        for c in risk_curves_maxprob_vs_r
        if c.get("r_final", {}).get("area") is not None
    ]
    mp_mean = float(np.mean(mp_areas)) if mp_areas else None
    rf_mean = float(np.mean(rf_areas)) if rf_areas else None
    results["risk_coverage_max_prob_vs_r_final"] = {
        "max_prob_area_mean": mp_mean,
        "r_final_area_mean": rf_mean,
        "r_final_minus_max_prob": (
            None if mp_mean is None or rf_mean is None else float(rf_mean - mp_mean)
        ),
        "r_final_beats_max_prob": (
            None if mp_mean is None or rf_mean is None else bool(rf_mean > mp_mean + 1e-9)
        ),
        "note": (
            "Locked fusion set A = r_model only; near-zero Δ means R_final ranking "
            "collapses to max-prob (expected until multi-feature fusion helps)."
        ),
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
        "m6_vs_m7_substantive_divergence_rate": coord_stats[
            "m6_vs_m7_substantive_divergence"
        ]
        / max(coord_stats["n_cases"], 1),
        "m6_vs_m7_divergence_causes": coord_stats["divergence_causes"],
        "note": (
            "Route divergence uses substantive actions only "
            "(extra_validation / recheck* / critic). "
            "If M5/M6/M7 remain indistinguishable on grades, agents may not "
            "change argmax; not tuned on test folds."
        ),
    }

    results["ablations"]["M2"] = {
        **results["ablations"]["M1"],
        "note": (
            "Held-out permutation importance sidecar; classification metrics "
            "identical to M1 on same folds. See m2_importance.json."
        ),
    }
    # Per-rule detection rates (M5)
    n_c = max(int(critic_stats["n_cases"]), 1)
    critic_stats["rule_detection_rates"] = {
        rule: {
            "count": int(cnt),
            "rate": float(cnt) / n_c,
            "wilson_ci_approx": _wilson_ci(int(cnt), n_c),
        }
        for rule, cnt in sorted(critic_stats["rule_fires"].items())
    }
    _write_m2_importance(cfg, df, m1_meta=m1_meta)

    results["robustness"] = _robustness_all_folds(
        cfg,
        df,
        X=X,
        y=y,
        groups=groups,
        m1_meta=m1_meta,
        n_outer=n_outer,
        n_repeats=n_repeats,
        quick=quick,
    )
    results["missing_data"] = _missing_data_all_folds(
        cfg,
        df,
        X=X,
        y=y,
        groups=groups,
        m1_meta=m1_meta,
        n_outer=n_outer,
        n_repeats=n_repeats,
        quick=quick,
    )

    out = resolve_path(cfg, "baselines") / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    return results


def _wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """Approximate Wilson score interval for a binomial rate."""
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _write_m2_importance(
    cfg: dict[str, Any], df: pd.DataFrame, *, m1_meta: dict[str, Any] | None = None
) -> None:
    """M7/M2: held-out permutation importance on fold-trained production family."""
    from sklearn.inspection import permutation_importance

    m1_meta = m1_meta or {}
    X_all, y_all, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    best_name = (
        m1_meta.get("nested_unbiased", {}).get("production_model")
        or m1_meta.get("best_model")
        or "logistic__median"
    )
    name, imp = best_name.split("__", 1)
    params = m1_meta.get("best_params") or {}
    try:
        est = estimator_from_params(name, params, seed)
    except Exception:  # noqa: BLE001
        zoo = get_model_zoo(quick=True)
        est = zoo.get(name) or next(iter(zoo.values()))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X_all, y_all, groups))
    fit_idx, _ = _split_fit_calib(y_all, groups, tr, seed=seed)
    pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
    pipe.fit(X_all.iloc[fit_idx], y_all[fit_idx])
    r = permutation_importance(
        pipe,
        X_all.iloc[te],
        y_all[te],
        n_repeats=5,
        random_state=seed,
        scoring="accuracy",
    )
    feat_names = list(X_all.columns)
    order = np.argsort(-r.importances_mean)
    ranked = [
        {
            "feature": feat_names[i],
            "importance_mean": float(r.importances_mean[i]),
            "importance_std": float(r.importances_std[i]),
        }
        for i in order
    ]
    payload = {
        "features": feat_names,
        "importances_mean": r.importances_mean.tolist(),
        "importances_std": r.importances_std.tolist(),
        "ranked": ranked,
        "top_features": [x["feature"] for x in ranked[:5]],
        "model": best_name,
        "protocol": (
            "permutation_importance on held-out StratifiedGroupKFold test fold; "
            "pipeline fit only on fold train/fit indices"
        ),
    }
    path = resolve_path(cfg, "baselines") / "m2_importance.json"
    path.write_text(json.dumps(payload, indent=2))


def _mean_boot_ci(
    vals: list[float], *, n_boot: int = 200, seed: int = 42
) -> dict[str, float | None]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": int(arr.size),
    }


def _robustness_all_folds(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    m1_meta: dict[str, Any],
    n_outer: int,
    n_repeats: int,
    quick: bool = False,
) -> dict[str, Any]:
    """C6: robustness on all outer folds/repeats with fold model + ensemble + conformal.

    Primary criterion: metrics degrade vs **unperturbed** baseline (not only mild vs strong).
    """
    seed = int(cfg.get("seed", 42))
    mild = float(cfg.get("uncertainty", {}).get("mild_eps", 0.05))
    strong = float(cfg.get("uncertainty", {}).get("strong_eps", 0.15))
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
        base_est = zoo.get(name) or next(iter(zoo.values()))
    models_meta = m1_meta.get("models") or {}
    ranked = sorted(
        models_meta.items(),
        key=lambda kv: (kv[1].get("pooled_oof") or {}).get("qwk", -1),
        reverse=True,
    )
    if not ranked:
        ranked = [(best_name, {"best_params": params, "fold_params": []})]

    per_case: dict[str, dict[str, list[float]]] = {
        lab: {
            "flip": [],
            "delta_p": [],
            "entropy_delta": [],
            "set_size_delta": [],
            "r_final_delta": [],
            "r_final_base": [],
            "r_final_pert": [],
        }
        for lab in ("mild", "strong")
    }
    n_cases = 0
    max_per_fold = 12 if quick else 40

    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_outer, shuffle=True, random_state=seed + rep
        )
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            fit_idx, calib_idx = _split_fit_calib(
                y, groups, tr, seed=seed + rep + fold
            )
            pipe = Pipeline(
                [("prep", build_preprocess(imp)), ("model", clone(base_est))]
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
                X, y, fit_idx, ranked, seed=seed + rep, fold=fold, top_k=3
            )
            fold_cfg = {
                **cfg,
                "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
                "_fold_pipeline": pipe,
                "_fold_ensemble": fold_ensemble,
                "_fold_model_name": best_name,
                "_conformal_state": conf,
            }
            te_use = list(te[: max_per_fold])
            rng = np.random.default_rng(seed + 1000 * rep + fold)
            for j in te_use:
                case = df.iloc[int(j)].to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                base_pred = run_predict(
                    case, fold_cfg, pipeline=pipe, model_name=best_name
                )
                base_p = np.asarray(base_pred.probabilities, dtype=float)
                base_unc = run_uncertainty(
                    case,
                    base_pred,
                    fold_cfg,
                    pipeline=pipe,
                    ensemble=fold_ensemble,
                    conformal=conf,
                )
                base_dq = run_data_quality(case, ref, fold_cfg)
                base_phys = run_physics(case, fold_cfg)
                base_fuse = fuse_assessment(
                    base_dq,
                    base_phys,
                    base_pred,
                    base_unc,
                    fold_cfg,
                    ["robust:base"],
                )
                base_r = float(base_fuse.get("final_system_confidence") or 0.0)
                base_ent = predictive_entropy(base_p)
                base_set_n = float(len(base_unc.prediction_set))
                for label, eps in ("mild", mild), ("strong", strong):
                    pc = perturb_case(case, FEATURE_COLS, eps, rng)
                    pp = run_predict(pc, fold_cfg, pipeline=pipe, model_name=best_name)
                    p = np.asarray(pp.probabilities, dtype=float)
                    u = run_uncertainty(
                        pc,
                        pp,
                        fold_cfg,
                        pipeline=pipe,
                        ensemble=fold_ensemble,
                        conformal=conf,
                    )
                    dq = run_data_quality(pc, ref, fold_cfg)
                    phys = run_physics(pc, fold_cfg)
                    fuse = fuse_assessment(
                        dq, phys, pp, u, fold_cfg, ["robust:pert"]
                    )
                    r_pert = float(fuse.get("final_system_confidence") or 0.0)
                    bucket = per_case[label]
                    bucket["flip"].append(
                        float(pp.predicted_grade != base_pred.predicted_grade)
                    )
                    bucket["delta_p"].append(float(np.mean(np.abs(base_p - p))))
                    bucket["entropy_delta"].append(
                        float(predictive_entropy(p) - base_ent)
                    )
                    bucket["set_size_delta"].append(
                        float(len(u.prediction_set) - base_set_n)
                    )
                    bucket["r_final_delta"].append(float(r_pert - base_r))
                    bucket["r_final_base"].append(base_r)
                    bucket["r_final_pert"].append(r_pert)
                n_cases += 1

    summary: dict[str, Any] = {
        "protocol": (
            "all outer folds×repeats; fold-trained pipeline+ensemble+conformal; "
            "perturb primaries only (M3); metrics vs unperturbed baseline"
        ),
        "n_cases": n_cases,
        "n_outer": n_outer,
        "n_repeats": n_repeats,
        "model": best_name,
        "eps": {"mild": mild, "strong": strong},
        "metrics": {},
        "degrades_vs_unperturbed": {},
    }
    for lab in ("mild", "strong"):
        b = per_case[lab]
        summary["metrics"][lab] = {
            "flip_rate": _mean_boot_ci(b["flip"], seed=seed + 1),
            "mean_abs_delta_p": _mean_boot_ci(b["delta_p"], seed=seed + 2),
            "mean_entropy_delta": _mean_boot_ci(b["entropy_delta"], seed=seed + 3),
            "mean_set_size_delta": _mean_boot_ci(b["set_size_delta"], seed=seed + 4),
            "mean_r_final_delta": _mean_boot_ci(b["r_final_delta"], seed=seed + 5),
        }
        r_d = summary["metrics"][lab]["mean_r_final_delta"]["mean"]
        e_d = summary["metrics"][lab]["mean_entropy_delta"]["mean"]
        s_d = summary["metrics"][lab]["mean_set_size_delta"]["mean"]
        summary["degrades_vs_unperturbed"][lab] = {
            "r_final_decreases": bool(r_d is not None and r_d < 0),
            "entropy_increases": bool(e_d is not None and e_d > 0),
            "set_size_increases": bool(s_d is not None and s_d > 0),
            "flip_rate_positive": bool(
                (summary["metrics"][lab]["flip_rate"]["mean"] or 0) > 0
            ),
        }
    # Secondary note: strong vs mild (not the acceptance criterion)
    r_m = summary["metrics"]["mild"]["mean_r_final_delta"]["mean"]
    r_s = summary["metrics"]["strong"]["mean_r_final_delta"]["mean"]
    summary["strong_vs_mild_note"] = {
        "r_final_delta_mild": r_m,
        "r_final_delta_strong": r_s,
        "strong_more_negative_than_mild": (
            r_m is not None and r_s is not None and r_s <= r_m + 1e-9
        ),
        "criterion": "primary = degrade vs unperturbed; mild-vs-strong is secondary",
    }
    return summary


def _score_missing_batch(
    *,
    df: pd.DataFrame,
    indices: list[int],
    mask_cols: tuple[str, ...] | list[str],
    pipe: Any,
    fold_cfg: dict[str, Any],
    fit_idx: np.ndarray,
    use_m7: bool,
    use_physics_recon: bool,
) -> dict[str, Any]:
    """Score one missingness scenario; optionally physics-reconstruct then M7."""
    ys, preds, r_deltas, set_sizes = [], [], [], []
    for ix in indices:
        case = df.loc[ix].to_dict()
        y_true = int(case["rockburst_grade"])
        masked = apply_missing_mask(case, mask_cols)
        if use_physics_recon:
            masked = reconstruct_missing_from_physics(masked)
        ref = build_fold_reference(df, fit_idx, case)
        if use_m7:
            # Complete-case baseline R_final for Δ
            base = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
            out = run_graph(masked, ref, fold_cfg, use_llm=False, allow_critic=True)
            g = out.get("predicted_grade")
            g_i = int(g) if g is not None else -1
            r_b = float(base.get("final_system_confidence") or 0.0)
            r_m = float(out.get("final_system_confidence") or 0.0)
            r_deltas.append(r_m - r_b)
            set_sizes.append(float(len(out.get("prediction_set") or [])))
        else:
            pred = run_predict(
                masked,
                fold_cfg,
                pipeline=pipe,
                model_name=fold_cfg.get("_fold_model_name"),
            )
            g_i = int(pred.predicted_grade)
            unc = run_uncertainty(
                masked,
                pred,
                fold_cfg,
                pipeline=pipe,
                ensemble=fold_cfg.get("_fold_ensemble"),
                conformal=fold_cfg.get("_conformal_state"),
            )
            set_sizes.append(float(len(unc.prediction_set)))
            r_deltas.append(float("nan"))
        ys.append(y_true)
        preds.append(g_i)
    yt = np.asarray(ys, dtype=int)
    yp = np.asarray(preds, dtype=int)
    valid = yp >= 0
    out: dict[str, Any] = {
        "n": int(valid.sum()),
        "accuracy": float((yp[valid] == yt[valid]).mean()) if valid.any() else None,
        "qwk": (
            quadratic_weighted_kappa(yt[valid], yp[valid]) if valid.any() else None
        ),
        "ordinal_mae": ordinal_mae(yt[valid], yp[valid]) if valid.any() else None,
        "severe_error_rate": (
            severe_misclassification_rate(yt[valid], yp[valid]) if valid.any() else None
        ),
        "mean_set_size": float(np.nanmean(set_sizes)) if set_sizes else None,
    }
    if use_m7:
        rd = [x for x in r_deltas if np.isfinite(x)]
        out["mean_delta_r_final"] = float(np.mean(rd)) if rd else None
    return out


def _missing_data_all_folds(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    m1_meta: dict[str, Any],
    n_outer: int,
    n_repeats: int,
    quick: bool = False,
) -> dict[str, Any]:
    """C7: physical missing masks + physics reconstruction + M7 across folds."""
    seed = int(cfg.get("seed", 42))
    alpha = float(cfg.get("uncertainty", {}).get("conformal_alpha", 0.1))
    conf_method = (m1_meta.get("conformal") or {}).get("method", "marginal_split")
    best_name = (
        m1_meta.get("nested_unbiased", {}).get("production_model")
        or m1_meta.get("best_model")
        or "logistic__median"
    )
    name, default_imp = best_name.split("__", 1)
    params = m1_meta.get("best_params") or {}
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:  # noqa: BLE001
        zoo = get_model_zoo(quick=True)
        base_est = zoo.get(name) or next(iter(zoo.values()))
    models_meta = m1_meta.get("models") or {}
    ranked = sorted(
        models_meta.items(),
        key=lambda kv: (kv[1].get("pooled_oof") or {}).get("qwk", -1),
        reverse=True,
    )
    if not ranked:
        ranked = [(best_name, {"best_params": params, "fold_params": []})]

    imputers = ["median", "knn"] if quick else ["median", "knn", "iterative"]
    max_per_fold = 8 if quick else 24
    # Collect per-scenario fold metrics then average
    scenario_keys = ["complete", *PHYSICAL_MISSING_MASKS.keys(), "physics_recon_theta"]
    store: dict[str, dict[str, list[dict[str, Any]]]] = {
        imp: {k: [] for k in scenario_keys} for imp in imputers
    }
    m7_store: dict[str, list[dict[str, Any]]] = {k: [] for k in scenario_keys}
    train_vs_inf: dict[str, list[dict[str, float]]] = {imp: [] for imp in imputers}

    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_outer, shuffle=True, random_state=seed + rep
        )
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            fit_idx, calib_idx = _split_fit_calib(
                y, groups, tr, seed=seed + rep + fold
            )
            te_df = df.iloc[te]
            complete_mask = te_df[FEATURE_COLS].notna().all(axis=1)
            complete_idx = te_df.index[complete_mask].tolist()[:max_per_fold]
            if not complete_idx:
                continue
            fold_ensemble = fit_fold_ensemble(
                X, y, fit_idx, ranked, seed=seed + rep, fold=fold, top_k=3
            )
            conf = None
            # Production-family pipe for M7 path
            pipe_prod = Pipeline(
                [("prep", build_preprocess(default_imp)), ("model", clone(base_est))]
            )
            pipe_prod.fit(X.iloc[fit_idx], y[fit_idx])
            if len(calib_idx):
                conf = _fit_conf(
                    conf_method,
                    y[calib_idx],
                    _predict_proba_safe(pipe_prod, X.iloc[calib_idx]),
                    alpha,
                )
                conf.n_fit = int(len(fit_idx))
            # Fold fusion for M7 decisions
            fusion = ReliabilityFusion()
            R_tr, ycorr_tr = [], []
            for j in fit_idx[: min(40, len(fit_idx))]:
                case = df.iloc[int(j)].to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                proba = _predict_proba_safe(pipe_prod, X.iloc[[j]])[0]
                pred = PredictiveOutput(
                    predicted_grade=int(proba.argmax()),
                    probabilities=proba.tolist(),
                    r_model=compute_r_model(proba),
                    model_name="fold",
                )
                dq = run_data_quality(case, ref, cfg)
                phys = run_physics(case, cfg)
                unc = run_uncertainty(
                    case,
                    pred,
                    cfg,
                    pipeline=pipe_prod,
                    ensemble=fold_ensemble,
                    conformal=conf,
                )
                R_tr.append(
                    compute_reliability_features(dq, phys, pred, unc, case=case, cfg=cfg)
                )
                ycorr_tr.append(int(pred.predicted_grade == int(y[j])))
            if R_tr:
                fusion.fit(np.asarray(R_tr), np.asarray(ycorr_tr))
                if fusion.is_fit:
                    r_vals = [fusion.r_final(r) for r in R_tr]
                    fusion.set_decision_thresholds(
                        select_decision_thresholds(
                            np.asarray(r_vals), np.asarray(ycorr_tr)
                        )
                    )
            fold_cfg_m7 = {
                **cfg,
                "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
                "_fold_pipeline": pipe_prod,
                "_fold_ensemble": fold_ensemble,
                "_fold_fusion": fusion,
                "_fold_model_name": best_name,
                "_fold_X": X.iloc[fit_idx],
                "_fold_y": y[fit_idx],
                "_conformal_state": conf,
            }

            # M7 scenarios (production family)
            m7_store["complete"].append(
                _score_missing_batch(
                    df=df,
                    indices=complete_idx,
                    mask_cols=(),
                    pipe=pipe_prod,
                    fold_cfg=fold_cfg_m7,
                    fit_idx=fit_idx,
                    use_m7=True,
                    use_physics_recon=False,
                )
            )
            for sk, cols in PHYSICAL_MISSING_MASKS.items():
                m7_store[sk].append(
                    _score_missing_batch(
                        df=df,
                        indices=complete_idx,
                        mask_cols=cols,
                        pipe=pipe_prod,
                        fold_cfg=fold_cfg_m7,
                        fit_idx=fit_idx,
                        use_m7=True,
                        use_physics_recon=False,
                    )
                )
            # Physics reconstruction: mask only σθ, then reconstruct
            m7_store["physics_recon_theta"].append(
                _score_missing_batch(
                    df=df,
                    indices=complete_idx,
                    mask_cols=("sigma_theta",),
                    pipe=pipe_prod,
                    fold_cfg=fold_cfg_m7,
                    fit_idx=fit_idx,
                    use_m7=True,
                    use_physics_recon=True,
                )
            )

            for imp_kind in imputers:
                pipe = Pipeline(
                    [("prep", build_preprocess(imp_kind)), ("model", clone(base_est))]
                )
                pipe.fit(X.iloc[fit_idx], y[fit_idx])
                fold_cfg = {
                    **cfg,
                    "_fold_pipeline": pipe,
                    "_fold_ensemble": fold_ensemble,
                    "_fold_model_name": best_name,
                    "_conformal_state": conf,
                }
                store[imp_kind]["complete"].append(
                    _score_missing_batch(
                        df=df,
                        indices=complete_idx,
                        mask_cols=(),
                        pipe=pipe,
                        fold_cfg=fold_cfg,
                        fit_idx=fit_idx,
                        use_m7=False,
                        use_physics_recon=False,
                    )
                )
                for sk, cols in PHYSICAL_MISSING_MASKS.items():
                    store[imp_kind][sk].append(
                        _score_missing_batch(
                            df=df,
                            indices=complete_idx,
                            mask_cols=cols,
                            pipe=pipe,
                            fold_cfg=fold_cfg,
                            fit_idx=fit_idx,
                            use_m7=False,
                            use_physics_recon=False,
                        )
                    )
                store[imp_kind]["physics_recon_theta"].append(
                    _score_missing_batch(
                        df=df,
                        indices=complete_idx,
                        mask_cols=("sigma_theta",),
                        pipe=pipe,
                        fold_cfg=fold_cfg,
                        fit_idx=fit_idx,
                        use_m7=False,
                        use_physics_recon=True,
                    )
                )
                # Train- vs inference-time missing D
                X_miss = X.copy()
                X_miss["d"] = np.nan
                pipe_miss = Pipeline(
                    [("prep", build_preprocess(imp_kind)), ("model", clone(base_est))]
                )
                pipe_miss.fit(X_miss.iloc[fit_idx], y[fit_idx])
                X_te = X.iloc[te].copy()
                X_te["d"] = np.nan
                train_vs_inf[imp_kind].append(
                    {
                        "inference_time_missing_d_acc": float(
                            (pipe.predict(X_te) == y[te]).mean()
                        ),
                        "train_and_inference_missing_d_acc": float(
                            (pipe_miss.predict(X_te) == y[te]).mean()
                        ),
                    }
                )

    def _agg(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        if not blocks:
            return {}
        keys = [
            "accuracy",
            "qwk",
            "ordinal_mae",
            "severe_error_rate",
            "mean_set_size",
            "mean_delta_r_final",
        ]
        out: dict[str, Any] = {"n_folds": len(blocks)}
        for k in keys:
            vals = [b[k] for b in blocks if b.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        out["n_cases_mean"] = float(np.mean([b.get("n", 0) for b in blocks]))
        return out

    imputers_out: dict[str, Any] = {}
    for imp_kind in imputers:
        block = {sk: _agg(store[imp_kind][sk]) for sk in scenario_keys}
        tvs = train_vs_inf[imp_kind]
        if tvs:
            block["train_vs_inference_missing_d"] = {
                "inference_time_missing_d_acc": float(
                    np.mean([t["inference_time_missing_d_acc"] for t in tvs])
                ),
                "train_and_inference_missing_d_acc": float(
                    np.mean([t["train_and_inference_missing_d_acc"] for t in tvs])
                ),
            }
        imputers_out[imp_kind] = block

    return {
        "protocol": (
            "all outer folds×repeats; production-family models; physical masks "
            "(σθ+SCF; σc+SCF/B1/B2; σt+B1/B2); physics reconstruction; "
            "imputer predict path + full M7 path"
        ),
        "physical_masks": {k: list(v) for k, v in PHYSICAL_MISSING_MASKS.items()},
        "model": best_name,
        "n_outer": n_outer,
        "n_repeats": n_repeats,
        "imputers": imputers_out,
        "m7": {sk: _agg(m7_store[sk]) for sk in scenario_keys},
        "note": (
            "ΔR_final reported on M7 path (missing vs complete same case). "
            "physics_recon_theta masks σθ then sets σθ=SCF·σc when possible."
        ),
    }


# Back-compat aliases (older call sites / docs)
def _robustness_fold_models(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Use _robustness_all_folds (Slice 5)")


def _missing_data_fold_models(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Use _missing_data_all_folds (Slice 5)")
