"""Deterministic agent tool wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.fusion.features import (
    compute_r_model,
    compute_reliability_features,
    r_vector_dict,
)
from rockburst.fusion.stacker import ReliabilityFusion
from rockburst.physics.relations import FEATURE_COLS, check_case
from rockburst.quality.checks import score_data_quality
from rockburst.uncertainty.conformal import ConformalState
from rockburst.uncertainty.estimates import (
    perturb_case,
    stability_from_preds,
    uncertainty_bundle,
)


def run_data_quality(case: dict[str, Any], reference: pd.DataFrame, cfg: dict[str, Any]) -> DataQualityOutput:
    q = cfg.get("quality", {})
    phys = cfg.get("physics", {})
    out = score_data_quality(
        case,
        reference,
        iqr_factor=q.get("iqr_factor", 3.0),
        scf_ood_threshold=phys.get("scf_ood_threshold", 2.0),
        near_dup_rel_tol=q.get("near_dup_rel_tol", 0.01),
        group_col=cfg.get("group_col", "dup_group_id"),
    )
    return DataQualityOutput(**out)


def run_physics(case: dict[str, Any], cfg: dict[str, Any]) -> PhysicsOutput:
    from rockburst.physics.relations import DEFAULT_ABS_TOL, DEFAULT_REL_TOL

    p = cfg.get("physics", {})
    res = check_case(
        case,
        rel_tol=p.get("rel_tol", DEFAULT_REL_TOL),
        abs_tol=p.get("abs_tol", DEFAULT_ABS_TOL),
        scf_ood_threshold=p.get("scf_ood_threshold", 2.0),
        wet_range=tuple(p.get("wet_range", [0.5, 40.0])),
        d_range=tuple(p.get("d_range", [50.0, 2500.0])),
        b1_implausible_max=float(p.get("b1_implausible_max", 200.0)),
    )
    return PhysicsOutput(**res.to_dict())


def _case_frame(case: dict[str, Any]) -> pd.DataFrame:
    row = {c: case.get(c, np.nan) for c in FEATURE_COLS}
    df = pd.DataFrame([row])
    for c in FEATURE_COLS:
        df[f"observed_{c}"] = (~df[c].isna()).astype(float)
    return df


def _pad_proba(proba: np.ndarray, n: int = 4) -> np.ndarray:
    proba = np.asarray(proba, dtype=float).ravel()
    if len(proba) < n:
        pad = np.zeros(n)
        pad[: len(proba)] = proba
        return pad
    return proba[:n]


def _predict_with_pipe(pipe: Any, case: dict[str, Any]) -> np.ndarray:
    X = _case_frame(case)
    if hasattr(pipe, "predict_proba"):
        return _pad_proba(pipe.predict_proba(X)[0])
    pred = int(pipe.predict(X)[0])
    proba = np.zeros(4)
    proba[pred] = 1.0
    return proba


def run_predict(
    case: dict[str, Any],
    cfg: dict[str, Any],
    *,
    pipeline: Any | None = None,
    model_name: str | None = None,
) -> PredictiveOutput:
    if pipeline is None:
        models_dir = Path(cfg.get("_root", ".")) / cfg["paths"]["models"]
        path = models_dir / "best_baseline.joblib"
        if not path.exists():
            from rockburst.ml.train import train_baselines

            train_baselines(cfg, quick=True)
        blob = joblib.load(path)
        pipe = blob["pipeline"]
        name = str(blob.get("name", "baseline"))
        # optional conformal attached to blob
        if "conformal" in blob and "_conformal_state" not in cfg:
            cfg["_conformal_state"] = ConformalState.from_dict(blob["conformal"])
    else:
        pipe = pipeline
        name = model_name or "fold_pipeline"

    proba = _predict_with_pipe(pipe, case)
    observed_mask = {
        c: bool(np.isfinite(float(case[c]))) if case.get(c) is not None else False
        for c in FEATURE_COLS
    }
    return PredictiveOutput(
        predicted_grade=int(proba.argmax()),
        probabilities=proba.tolist(),
        r_model=compute_r_model(proba),
        observed_mask=observed_mask,
        model_name=name,
    )


def run_uncertainty(
    case: dict[str, Any],
    pred: PredictiveOutput,
    cfg: dict[str, Any],
    *,
    pipeline: Any | None = None,
    ensemble: dict[str, Any] | None = None,
    conformal: ConformalState | None = None,
) -> UncertaintyOutput:
    u = cfg.get("uncertainty", {})
    models_dir = Path(cfg.get("_root", ".")) / cfg["paths"]["models"]

    if pipeline is None:
        models_path = models_dir / "best_baseline.joblib"
        blob = joblib.load(models_path)
        pipe = blob["pipeline"]
        if conformal is None and "conformal" in blob:
            conformal = ConformalState.from_dict(blob["conformal"])
    else:
        pipe = pipeline

    # Leakage rule (M6/C6): never load full-data ensemble.joblib when a fold
    # pipeline is in play. OOF/ablation must pass fold-trained ensembles explicitly.
    # Production serving (pipeline is None) may load the on-disk production ensemble.
    if ensemble is None:
        ensemble_probas: list[np.ndarray] = []
        if pipeline is None:
            ens_path = models_dir / "ensemble.joblib"
            if ens_path.exists():
                ens = joblib.load(ens_path)
                for p in ens.values():
                    ensemble_probas.append(_predict_with_pipe(p, case))
    else:
        ensemble_probas = [_predict_with_pipe(p, case) for p in ensemble.values()]

    if conformal is None:
        conformal = cfg.get("_conformal_state")

    base = np.asarray(pred.probabilities, dtype=float)
    thr = conformal.class_thresholds if conformal is not None else None
    alpha = float(conformal.alpha) if conformal is not None else float(u.get("conformal_alpha", 0.1))
    bundle = uncertainty_bundle(
        base, ensemble_probas, alpha=alpha, class_thresholds=thr
    )

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    mild, strong = float(u.get("mild_eps", 0.05)), float(u.get("strong_eps", 0.15))
    n_pert = int(u.get("n_perturb", 10))
    preds, probas = [], []
    for eps in (mild, strong):
        for _ in range(max(1, n_pert // 2)):
            pc = perturb_case(case, FEATURE_COLS, eps, rng)
            pp = _predict_with_pipe(pipe, pc)
            preds.append(int(np.argmax(pp)))
            probas.append(pp)
    stab = stability_from_preds(pred.predicted_grade, preds, base, probas)
    return UncertaintyOutput(
        uncertainty_score=bundle["uncertainty_score"],
        r_uncertainty=bundle["r_uncertainty"],
        prediction_set=bundle["prediction_set"],
        model_disagreement=bundle["model_disagreement"],
        prediction_stability=stab["prediction_stability"],
        stability_score=stab["stability_score"],
        entropy=bundle["entropy"],
    )


# Rules that escalate unresolved Critic objections all the way to abstain (M5).
_SEVERE_CRITIC_RULES = frozenset(
    {
        "identity_violation",
        "ml_vs_criteria",
        "implausible_stress_ratio",
        "suspected_row_shift",
        "conflicting_near_dup_labels",
        "wide_conformal_edge",
    }
)


def run_critic(
    data_q: DataQualityOutput | None,
    phys: PhysicsOutput | None,
    pred: PredictiveOutput | None,
    unc: UncertaintyOutput | None,
    cfg: dict[str, Any],
    *,
    validation_objections: list[str] | None = None,
) -> CriticOutput:
    challenges, contradictions = [], []
    rules_fired: list[str] = []
    request: str = "none"
    endorse = True
    rt = cfg.get("routing", {})
    objections = list(validation_objections or [])

    def _fire(rule: str, *, req: str | None = None) -> None:
        nonlocal request, endorse
        rules_fired.append(rule)
        endorse = False
        if req and request == "none":
            request = req

    if phys and phys.physics_score < 0.6:
        challenges.append("Low physics consistency")
        _fire("low_physics")
        if pred and pred.r_model > 0.8:
            contradictions.append("High model confidence despite physics failure")
            _fire("physics_vs_model_confidence", req="physics")
    if data_q and data_q.ood_flag and pred and pred.r_model > 0.75:
        contradictions.append("OOD input with overconfident prediction")
        _fire("ood_overconfident", req="data")
    if unc and unc.model_disagreement >= rt.get("disagreement_threshold", 0.35):
        challenges.append(
            f"Strong ensemble disagreement (TV={unc.model_disagreement:.3f})"
        )
        _fire("ensemble_disagreement", req="model")
    if unc and len(unc.prediction_set) >= rt.get("large_set_size", 3):
        challenges.append("Large conformal prediction set")
        _fire("large_prediction_set", req="model")
        # Wide ordinal edge: set spans ≥2 grade steps
        pset = sorted(int(g) for g in unc.prediction_set)
        if pset and (pset[-1] - pset[0]) >= 2:
            challenges.append("Wide conformal ordinal edge")
            _fire("wide_conformal_edge", req="model")
    if phys and any("b1 inconsistency" in i for i in phys.issues):
        challenges.append("Derived B1 does not match sigma_c/sigma_t")
        _fire("b1_inconsistency", req="physics")
    if phys and any("inconsistency" in i for i in phys.issues):
        challenges.append("Identity violation on SCF/B1/B2")
        _fire("identity_violation", req="physics")
    if phys and any("implausible_sigma_c_over_sigma_t" in i for i in phys.issues):
        challenges.append("Implausible sigma_c/sigma_t ratio")
        _fire("implausible_stress_ratio", req="physics")
    if phys and any("suspected_row_shift" in i for i in phys.issues):
        challenges.append("Suspected table row-shift in listed derived features")
        _fire("suspected_row_shift", req="physics")
    if data_q and getattr(data_q, "conflicting_near_dup_labels", False):
        challenges.append("Near-duplicates with conflicting labels")
        contradictions.append("Conflicting near-duplicate labels in reference")
        _fire("conflicting_near_dup_labels", req="data")
    # ML grade vs empirical criteria consensus
    if phys is not None and pred is not None:
        crit = (phys.empirical_checks or {}).get("criteria_as_given") or {}
        cons = crit.get("consensus_grade")
        if cons is not None and int(cons) >= 0 and int(pred.predicted_grade) != int(cons):
            contradictions.append(
                f"ML grade {pred.predicted_grade} ≠ criteria consensus {cons} "
                f"(agreement={crit.get('agreement')})"
            )
            challenges.append("ML vs physics-criteria grade mismatch")
            _fire("ml_vs_criteria", req="physics")
    if objections:
        challenges.extend([f"validation:{o}" for o in objections])
        _fire("validation_objection")
        if request == "none":
            if any("physics" in o for o in objections):
                request = "physics"
            elif any("imputer" in o or "perturbation" in o for o in objections):
                request = "data"
            else:
                request = "model"
    escalate = bool(set(rules_fired) & _SEVERE_CRITIC_RULES)
    return CriticOutput(
        challenges=challenges,
        contradictions=contradictions,
        request_recheck=request if request in {"none", "physics", "data", "model"} else "none",  # type: ignore[arg-type]
        endorse=endorse,
        rules_fired=rules_fired,
        escalate_abstain=escalate,
    )


_DOWNGRADE = {
    "accept": "accept_with_caution",
    "accept_with_caution": "engineering_verification",
    # M5: remove floor at engineering_verification — escalate to abstain
    "engineering_verification": "abstain",
    "abstain": "abstain",
}

# Rockburst intensity names (grades 0–3); used in engineering decision text (M15).
INTENSITY_NAMES = ("none", "light", "moderate", "strong")


def intensity_name(grade: int | None) -> str:
    if grade is None:
        return "unknown"
    g = int(grade)
    if 0 <= g < len(INTENSITY_NAMES):
        return INTENSITY_NAMES[g]
    return f"grade_{g}"


def _inter_agent_disagreement(
    phys: PhysicsOutput | None,
    pred: PredictiveOutput | None,
    critic: CriticOutput | None,
) -> dict[str, Any]:
    """Structured ML vs physics-criteria vs Critic disagreement (M15)."""
    crit = {}
    if phys is not None:
        crit = (phys.empirical_checks or {}).get("criteria_as_given") or {}
    cons = crit.get("consensus_grade")
    ml = int(pred.predicted_grade) if pred is not None else None
    cons_i = int(cons) if cons is not None and int(cons) >= 0 else None
    ml_vs_criteria = (
        ml is not None and cons_i is not None and ml != cons_i
    )
    critic_rules = list(getattr(critic, "rules_fired", []) or []) if critic else []
    critic_objects = bool(critic is not None and not critic.endorse)
    parties: list[str] = []
    if ml_vs_criteria:
        parties.append("ml_vs_criteria")
    if critic_objects:
        parties.append("critic_vs_fusion_path")
    if "ml_vs_criteria" in critic_rules:
        parties.append("critic_flags_ml_vs_criteria")
    return {
        "ml_grade": ml,
        "ml_intensity": intensity_name(ml),
        "criteria_consensus_grade": cons_i,
        "criteria_intensity": intensity_name(cons_i),
        "criteria_grades": crit.get("grades") or {},
        "ml_vs_criteria": ml_vs_criteria,
        "critic_endorse": None if critic is None else bool(critic.endorse),
        "critic_rules_fired": critic_rules,
        "parties": parties,
        "has_disagreement": bool(parties),
    }


def format_engineering_decision(
    *,
    predicted_grade: int | None,
    prediction_set: list[int],
    r_final: float,
    action: str,
    disagreement: dict[str, Any] | None = None,
) -> str:
    """Human-readable engineering decision with intensity names (M15)."""
    pred_name = intensity_name(predicted_grade)
    set_names = [intensity_name(g) for g in prediction_set]
    set_txt = ", ".join(set_names) if set_names else "empty"
    lines = [
        f"Predicted intensity: {pred_name} (grade {predicted_grade}).",
        f"Conformal set: {{{set_txt}}} (grades {list(prediction_set)}).",
        f"System confidence R_final={float(r_final):.3f}.",
        f"Recommended action: {action}.",
    ]
    if disagreement and disagreement.get("has_disagreement"):
        parts = []
        if disagreement.get("ml_vs_criteria"):
            parts.append(
                f"ML {disagreement.get('ml_intensity')} vs criteria "
                f"{disagreement.get('criteria_intensity')}"
            )
        if disagreement.get("critic_endorse") is False:
            rules = disagreement.get("critic_rules_fired") or []
            parts.append(f"Critic objections ({', '.join(rules) or 'unspecified'})")
        if parts:
            lines.append("Inter-agent disagreement: " + "; ".join(parts) + ".")
    return " ".join(lines)


def enforce_decision_consistency(
    *,
    action: str,
    r_final: float,
    high_conf: float = 0.95,
) -> tuple[str, float, str | None]:
    """Enforce consistent (action, R_final) pairs (M15).

    Abstain + near-certain R_final is invalid UX: keep abstain (Critic/gates may
    require it) but clamp reported confidence. Accept + very low R_final → abstain.

    Returns (action, r_final, consistency_note).
    """
    r = float(r_final)
    if action == "abstain" and r >= high_conf:
        clamped = min(r, 0.49)
        return (
            "abstain",
            clamped,
            (
                f"consistency: abstain with R_final={r:.3f}≥{high_conf}; "
                f"clamped reported confidence to {clamped:.3f}"
            ),
        )
    if action == "accept" and r < 0.35:
        return (
            "abstain",
            r,
            (
                f"consistency: accept incompatible with R_final={r:.3f}<0.35; "
                "changed action to abstain"
            ),
        )
    return action, r, None


def fuse_assessment(
    data_q: DataQualityOutput | None,
    phys: PhysicsOutput | None,
    pred: PredictiveOutput | None,
    unc: UncertaintyOutput | None,
    cfg: dict[str, Any],
    route_trace: list[str],
    *,
    critic: CriticOutput | None = None,
    recheck_payloads: dict[str, Any] | None = None,
    physics_skipped: bool = False,
) -> dict[str, Any]:
    r_vec = compute_reliability_features(
        data_q, phys, pred, unc, case=None, cfg=cfg
    )
    r_parts = r_vector_dict(r_vec)
    r_data = r_parts["r_data"]
    r_physics = r_parts["r_physics"]
    r_model = r_parts["r_model"]
    r_unc = r_parts["r_uncertainty"]
    r_stab = r_parts["r_stability"]
    # Prefer in-memory fold fusion (ablation) over overwriting production fusion.joblib.
    fusion = cfg.get("_fold_fusion")
    if fusion is None:
        fusion_path = Path(cfg.get("_root", ".")) / cfg["paths"]["models"] / "fusion.joblib"
        fusion = (
            ReliabilityFusion.load(fusion_path) if fusion_path.exists() else ReliabilityFusion()
        )
    explanation = cfg.get("_case_explanation") or _default_explanation(cfg)
    if not fusion.is_fit:
        decision = "abstain"
        r_final = 0.0
        grade = pred.predicted_grade if pred else None
        pset = list(unc.prediction_set) if unc else []
        plain = format_engineering_decision(
            predicted_grade=grade,
            prediction_set=pset,
            r_final=r_final,
            action=decision,
        )
        plain = (
            f"{plain} Fusion stacker not fitted; abstain "
            "(no heuristic reliability weights)."
        )
        return {
            "predicted_grade": grade,
            "predicted_intensity": intensity_name(grade),
            "prediction_probability": float(max(pred.probabilities)) if pred else None,
            "prediction_set": pset,
            "prediction_set_intensities": [intensity_name(g) for g in pset],
            "physics_consistency": r_physics,
            "input_reliability": r_data,
            "ood_status": "Yes" if (data_q and data_q.ood_flag) else "No",
            "prediction_stability": unc.prediction_stability if unc else "unknown",
            "agent_disagreement": "unknown",
            "inter_agent_disagreement": _inter_agent_disagreement(phys, pred, critic),
            "final_system_confidence": r_final,
            "decision": plain,
            "recommended_action": decision,
            "probabilities": pred.probabilities if pred else [],
            "route_trace": route_trace,
            "abstained": True,
            "critic_endorse": None if critic is None else bool(critic.endorse),
            "downgrade_reason": "fusion_unfitted",
            "consistency_note": None,
            "physics_skipped": physics_skipped,
            "recheck_payloads": recheck_payloads or {},
            "r_vector": {**r_parts, "r_final": r_final},
            "explanation": explanation,
            "critic_rules_fired": list(getattr(critic, "rules_fired", []) or []),
            "delivered_planner": "rules_fsm",
        }
    r_final = fusion.r_final(r_vec)
    # C3: use learned thresholds on the fusion object by default; YAML hand-set
    # routing.abstain_r_final / accept_r_final must not silently override.
    decision = fusion.decide(r_final, cfg.get("routing", {}))
    if unc is not None:
        if len(unc.prediction_set) >= 3 and unc.stability_score < 0.55:
            decision = "abstain"
        elif unc.model_disagreement >= 0.5 and r_final < 0.75:
            decision = "abstain"
    if data_q is not None and phys is not None:
        if data_q.ood_flag and phys.physics_score < 0.7 and (pred is None or pred.r_model < 0.6):
            decision = "abstain"

    # Unresolved Critic: escalate to abstain on severe rules; else one-step downgrade (M5).
    downgrade_reason = None
    if critic is not None and not critic.endorse and decision != "abstain":
        if getattr(critic, "escalate_abstain", False):
            downgrade_reason = (
                f"Critic severe rules {sorted(set(critic.rules_fired) & _SEVERE_CRITIC_RULES)}; "
                f"escalated {decision} → abstain"
            )
            decision = "abstain"
        else:
            new_decision = _DOWNGRADE.get(decision, decision)
            if new_decision != decision:
                downgrade_reason = (
                    f"Critic unresolved (endorse=False, recheck={critic.request_recheck}); "
                    f"downgraded {decision} → {new_decision}"
                )
                decision = new_decision

    grade = pred.predicted_grade if pred else None
    pset = list(unc.prediction_set) if unc else []
    decision, r_final, consistency_note = enforce_decision_consistency(
        action=decision, r_final=float(r_final)
    )
    if consistency_note:
        downgrade_reason = (
            f"{downgrade_reason}; {consistency_note}"
            if downgrade_reason
            else consistency_note
        )

    disagreement = _inter_agent_disagreement(phys, pred, critic)
    # Prefer structured inter-agent label when ML≠criteria or Critic objects
    if disagreement.get("has_disagreement"):
        dis_label = "inter_agent"
    elif unc:
        dis_label = (
            "high"
            if unc.model_disagreement >= 0.35
            else "moderate"
            if unc.model_disagreement >= 0.15
            else "low"
        )
    else:
        dis_label = "low"

    plain = format_engineering_decision(
        predicted_grade=grade,
        prediction_set=pset,
        r_final=float(r_final),
        action=decision,
        disagreement=disagreement,
    )
    action_blurb = {
        "accept": "Prediction accepted; routine monitoring recommended.",
        "accept_with_caution": "Accept with caution; verify key measurements.",
        "engineering_verification": (
            "Confidence insufficient for routine accept; engineering verification recommended."
        ),
        "abstain": "Inadequate confidence; abstain and request additional data/review.",
    }[decision]
    plain = f"{plain} {action_blurb}"
    if downgrade_reason:
        plain = f"{plain} ({downgrade_reason})"
    return {
        "predicted_grade": grade,
        "predicted_intensity": intensity_name(grade),
        "prediction_probability": float(max(pred.probabilities)) if pred else None,
        "prediction_set": pset,
        "prediction_set_intensities": [intensity_name(g) for g in pset],
        "physics_consistency": r_physics,
        "input_reliability": r_data,
        "ood_status": "Yes" if (data_q and data_q.ood_flag) else "No",
        "prediction_stability": unc.prediction_stability if unc else "unknown",
        "agent_disagreement": dis_label,
        "inter_agent_disagreement": disagreement,
        "final_system_confidence": r_final,
        "decision": plain,
        "recommended_action": decision,
        "probabilities": pred.probabilities if pred else [],
        "route_trace": route_trace,
        "abstained": decision == "abstain",
        "critic_endorse": None if critic is None else bool(critic.endorse),
        "downgrade_reason": downgrade_reason,
        "consistency_note": consistency_note,
        "physics_skipped": physics_skipped,
        "recheck_payloads": recheck_payloads or {},
        "r_vector": {**r_parts, "r_final": r_final},
        "explanation": explanation,
        "critic_rules_fired": list(getattr(critic, "rules_fired", []) or []),
        "delivered_planner": "rules_fsm",
    }


def _default_explanation(cfg: dict[str, Any]) -> str:
    tip = cfg.get("_m2_top_features")
    if tip:
        return f"Top held-out permutation features: {', '.join(tip)}"
    try:
        path = Path(cfg.get("_root", ".")) / cfg["paths"]["baselines"] / "m2_importance.json"
        if path.exists():
            import json

            payload = json.loads(path.read_text())
            tops = payload.get("top_features") or []
            if tops:
                return f"Top held-out permutation features: {', '.join(tops[:5])}"
    except Exception:  # noqa: BLE001
        pass
    return "Production model; see m2_importance.json for held-out permutation ranks."
