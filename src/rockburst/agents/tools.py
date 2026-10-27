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
    # Sensitivity pipelines may intentionally omit observed_* indicators.
    # Align inference to the exact schema recorded by sklearn at fit time.
    expected = getattr(pipe, "feature_names_in_", None)
    if expected is not None:
        X = X.reindex(columns=list(expected))
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
        if blob.get("decode_rule") and "decode_rule" not in cfg:
            cfg["decode_rule"] = blob["decode_rule"]
    else:
        pipe = pipeline
        name = model_name or "fold_pipeline"

    proba = _predict_with_pipe(pipe, case)
    observed_mask = {
        c: bool(np.isfinite(float(case[c]))) if case.get(c) is not None else False
        for c in FEATURE_COLS
    }
    # RM10: honour production decode_rule from cfg / artefact (default argmax).
    decode = str(
        cfg.get("decode_rule")
        or (cfg.get("ml") or {}).get("decode_rule")
        or "argmax"
    )
    if decode == "median":
        from rockburst.ml.ordinal import median_of_distribution

        grade = int(median_of_distribution(proba)[0])
    else:
        grade = int(proba.argmax())
    return PredictiveOutput(
        predicted_grade=grade,
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


_SEVERE_CRITIC_RULES: frozenset[str] = frozenset()

def critic_rule_stat_hooks() -> dict[str, Any]:
    """Schema for per-rule flagged vs unflagged error / AUROC (F16 / Phase 5)."""
    return {
        "per_rule_fields": [
            "rule",
            "n_flagged",
            "n_unflagged",
            "error_flagged",
            "error_unflagged",
            "auroc",
            "ci_low",
            "ci_high",
        ],
        "severity_source": (
            "Promote to severe only when flagged error ≫ unflagged and AUROC "
            "justifies it; do not hand-severe all |set|≥3 (RM2)."
        ),
        "default_severe_rules": sorted(_SEVERE_CRITIC_RULES),
    }


def severe_rules_from_stats(stats: dict[str, Any] | None) -> set[str]:
    """Select severe rules only when grouped-bootstrap evidence supports harm."""
    severe: set[str] = set()
    for rule, block in (stats or {}).items():
        if not isinstance(block, dict):
            continue
        auc_ci = block.get("auroc_ci95") or block.get("ci95_auroc")
        delta_ci = block.get("error_delta_ci95") or block.get("ci95_error_delta")
        if (
            isinstance(auc_ci, (list, tuple))
            and len(auc_ci) == 2
            and isinstance(delta_ci, (list, tuple))
            and len(delta_ci) == 2
            and float(auc_ci[0]) > 0.5
            and float(delta_ci[0]) > 0.0
        ):
            severe.add(str(rule))
    return severe


def disabled_rules_from_stats(stats: dict[str, Any] | None) -> set[str]:
    """Disable rules whose grouped-bootstrap AUROC is confidently below chance."""
    disabled: set[str] = set()
    for rule, block in (stats or {}).items():
        if not isinstance(block, dict):
            continue
        auc_ci = block.get("auroc_ci95") or block.get("ci95_auroc")
        if (
            isinstance(auc_ci, (list, tuple))
            and len(auc_ci) == 2
            and float(auc_ci[1]) < 0.5
        ):
            disabled.add(str(rule))
    return disabled


def run_critic(
    data_q: DataQualityOutput | None,
    phys: PhysicsOutput | None,
    pred: PredictiveOutput | None,
    unc: UncertaintyOutput | None,
    cfg: dict[str, Any],
    *,
    validation_objections: list[str] | None = None,
    local_explanation: dict[str, float] | None = None,
    resolved_rules: set[str] | None = None,
) -> CriticOutput:
    challenges, contradictions = [], []
    rules_fired: list[str] = []
    request: str = "none"
    endorse = True
    rt = cfg.get("routing", {})
    objections = list(validation_objections or [])
    rule_stats = cfg.get("_critic_rule_statistics")
    suppressed = set(resolved_rules or set()) | disabled_rules_from_stats(rule_stats)
    severe = severe_rules_from_stats(rule_stats)
    records: list[dict[str, Any]] = []

    def _fire(rule: str, *, req: str | None = None) -> None:
        nonlocal request, endorse
        if rule in suppressed:
            return
        if rule in rules_fired:
            return
        rules_fired.append(rule)
        records.append(
            {
                "rule": rule,
                "status": "open",
                "requested_recheck": req or "none",
                "resolution_reason": None,
            }
        )
        endorse = False
        if req and request == "none":
            request = req

    if phys and phys.physics_score < 0.6:
        challenges.append("Low physics consistency")
        _fire("low_physics")
        if pred and pred.r_model > 0.8:
            contradictions.append("High model confidence despite physics failure")
            _fire("physics_vs_model_confidence", req="physics")
    # RM6: OOD + failed physics → severe (case 146 class).
    if (
        data_q
        and data_q.ood_flag
        and phys is not None
        and float(phys.physics_score) < 0.7
    ):
        contradictions.append("OOD input combined with failed physics")
        challenges.append("OOD + failed physics")
        _fire("ood_and_failed_physics", req="data")
    if data_q and data_q.ood_flag and pred and pred.r_model > 0.75:
        contradictions.append("OOD input with overconfident prediction")
        _fire("ood_overconfident", req="data")
    if unc and unc.model_disagreement >= rt.get("disagreement_threshold", 0.35):
        challenges.append(
            f"Strong ensemble disagreement (TV={unc.model_disagreement:.3f})"
        )
        _fire("ensemble_disagreement", req="model")
    # RM2: large set (non-severe by default) + distinct edge-of-wide-set check.
    if unc and len(unc.prediction_set) >= rt.get("large_set_size", 3):
        challenges.append("Large conformal prediction set")
        _fire("large_prediction_set", req="model")
        pset = sorted(int(g) for g in unc.prediction_set)
        span_ok = bool(pset) and (pset[-1] - pset[0]) >= 2
        if (
            pred is not None
            and span_ok
            and int(pred.predicted_grade) in {pset[0], pset[-1]}
        ):
            challenges.append(
                f"Predicted grade {pred.predicted_grade} at edge of wide set {pset}"
            )
            _fire("edge_of_wide_set", req="model")
    if phys and any(
        ("scf inconsistency" in i) or ("b2 inconsistency" in i)
        for i in phys.issues
    ):
        challenges.append("Identity violation on SCF or B2")
        _fire("identity_violation", req="physics")
    if phys and any("implausible_sigma_c_over_sigma_t" in i for i in phys.issues):
        challenges.append("Implausible sigma_c/sigma_t ratio")
        _fire("implausible_strength_ratio", req="physics")
    if phys and any("large_b1_identity_gap" in i for i in phys.issues):
        challenges.append("Large B1 identity gap (not row-shift without neighbours)")
        _fire("large_b1_identity_gap", req="physics")
    if phys and any("suspected_row_shift" in i for i in phys.issues):
        challenges.append("Suspected table row-shift (block diagnose)")
        _fire("suspected_row_shift", req="physics")
    if phys and any("sigma_t_ge_sigma_c" in i for i in phys.issues):
        challenges.append("Tensile strength ≥ UCS")
        _fire("sigma_t_ge_sigma_c", req="physics")
    if phys and any("tangential_stress_exceeds_strength" in i for i in phys.issues):
        challenges.append("Tangential boundary stress reaches or exceeds UCS")
        _fire("tangential_stress_exceeds_strength", req="physics")
    if phys and any("depth_stress_implausible" in i for i in phys.issues):
        challenges.append("Tangential stress is outside the Kirsch depth range")
        _fire("depth_stress_implausible", req="physics")
    if data_q and getattr(data_q, "conflicting_near_dup_labels", False):
        challenges.append("Near-duplicates with conflicting labels")
        contradictions.append("Conflicting near-duplicate labels in reference")
        _fire("conflicting_near_dup_labels", req="data")
    # ML grade vs empirical criteria consensus (RM5: severity from agreement).
    if phys is not None and pred is not None:
        crit = (phys.empirical_checks or {}).get("criteria_as_given") or {}
        cons = crit.get("consensus_grade")
        agr = float(crit.get("agreement") or 0.0)
        if cons is not None and int(cons) >= 0 and int(pred.predicted_grade) != int(cons):
            contradictions.append(
                f"ML grade {pred.predicted_grade} ≠ criteria consensus {cons} "
                f"(agreement={agr:.3f}, rule={crit.get('consensus_rule')})"
            )
            challenges.append("ML vs physics-criteria grade mismatch")
            _fire("ml_vs_criteria", req="physics")
    if objections:
        challenges.extend([f"validation:{o}" for o in objections])
        _fire("validation_objection")
        if any("monotonic_tendency" in o for o in objections):
            _fire("monotonic_tendency_violation", req="model")
        if request == "none":
            if any("physics" in o for o in objections):
                request = "physics"
            elif any(
                "imputer" in o or "perturbation" in o or "monotonic" in o
                for o in objections
            ):
                request = "data" if "monotonic" not in " ".join(objections) else "model"
            else:
                request = "model"
    # RM9: implausible local explanation directions (SCF/Wet/σθ vs severity).
    attrib = local_explanation
    if isinstance(attrib, dict) and pred is not None:
        from rockburst.ml.explain import check_implausible_explanation_directions

        for flag in check_implausible_explanation_directions(
            {k: float(v) for k, v in attrib.items() if isinstance(v, (int, float))},
            int(pred.predicted_grade),
        ):
            contradictions.append(flag)
            challenges.append("Implausible explanation direction vs rockburst severity prior")
            _fire("implausible_explanation_direction", req="model")
            break
    escalate = bool(set(rules_fired) & severe)
    parties: set[str] = set()
    if any(r in rules_fired for r in {"ml_vs_criteria", "physics_vs_model_confidence"}):
        parties.update({"predictive_agent", "geomechanics_agent"})
    if any(
        r in rules_fired
        for r in {"ensemble_disagreement", "large_prediction_set", "edge_of_wide_set"}
    ):
        parties.update({"predictive_agent", "uncertainty_agent"})
    if any(r in rules_fired for r in {"ood_overconfident", "ood_and_failed_physics"}):
        parties.update({"data_quality_agent", "predictive_agent"})
    disagreement_level = (
        "High" if escalate or len(parties) >= 3 else "Moderate" if parties else "Low"
    )
    return CriticOutput(
        challenges=challenges,
        contradictions=contradictions,
        request_recheck=request if request in {"none", "physics", "data", "model"} else "none",  # type: ignore[arg-type]
        endorse=endorse,
        rules_fired=rules_fired,
        escalate_abstain=escalate,
        objection_records=records,
        disagreement_level=disagreement_level,
        disagreeing_parties=sorted(parties),
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
    if critic is not None:
        parties.extend(p for p in critic.disagreeing_parties if p not in parties)
    grade_gap = abs(ml - cons_i) if ml is not None and cons_i is not None else 0
    computed_level = (
        "High"
        if grade_gap >= 2 or (critic_objects and len(critic_rules) >= 2)
        else "Moderate"
        if parties
        else "Low"
    )
    level = (
        critic.disagreement_level
        if critic is not None and critic.disagreement_level != "Low"
        else computed_level
    )
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
        "level": level,
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
    criteria_grade = None
    if disagreement:
        criteria_grade = disagreement.get("criteria_consensus_grade")
    hazard_grade = max(
        [g for g in [predicted_grade, criteria_grade, *prediction_set] if g is not None],
        default=-1,
    )
    if hazard_grade >= 3:
        lines.append("Hazard statement: High rockburst potential; engineering verification required.")
    elif hazard_grade == 2:
        lines.append("Hazard statement: Moderate rockburst potential; engineering verification recommended.")
    else:
        lines.append("Hazard statement: No high-intensity indication in the available evidence; continue monitoring.")
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
            lines.append(
                f"Inter-agent disagreement ({disagreement.get('level', 'Moderate')}): "
                + "; ".join(parts)
                + "."
            )
    return " ".join(lines)


def enforce_decision_consistency(
    *,
    action: str,
    r_final: float,
    high_conf: float = 0.95,
) -> tuple[str, float, str | None]:
    """Align action with R_final without rewriting confidence (RC3).

    Critic/hard gates may force abstain while calibrated R_final stays high —
    keep the true R_final and record ``decision_override_reason`` instead of the
    old 0.49 clamp. Accept + very low R_final → abstain (action change only).

    Returns (action, r_final_unchanged, decision_override_reason | None).
    """
    r = float(r_final)
    if action == "abstain" and r >= high_conf:
        return (
            "abstain",
            r,
            (
                f"override: abstain with R_final={r:.3f}≥{high_conf} "
                "(Critic/gates); reported R_final left unchanged"
            ),
        )
    if action == "accept" and r < 0.35:
        return (
            "abstain",
            r,
            (
                f"override: accept incompatible with R_final={r:.3f}<0.35; "
                "changed action to abstain (R_final unchanged)"
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
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Single train/serve/eval vector path; component calibrators only if explicit.
    r_vec = compute_reliability_features(
        data_q, phys, pred, unc, case=case, cfg=cfg
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
            "decision_override_reason": None,
            "r_final": r_final,
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
    # RM6: OOD + failed physics → abstain (no high-r_model escape; case 146).
    if data_q is not None and phys is not None:
        if data_q.ood_flag and float(phys.physics_score) < 0.7:
            decision = "abstain"

    # Unresolved Critic: escalate to abstain on severe rules; else one-step downgrade (M5).
    downgrade_reason = None
    if critic is not None and not critic.endorse and decision != "abstain":
        if getattr(critic, "escalate_abstain", False):
            fired_severe = sorted(set(critic.rules_fired))
            downgrade_reason = (
                f"Critic severe rules {fired_severe}; "
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
    r_final_reported = float(r_final)
    decision, r_final_reported, decision_override_reason = enforce_decision_consistency(
        action=decision, r_final=r_final_reported
    )
    consistency_note = decision_override_reason
    if decision_override_reason:
        downgrade_reason = (
            f"{downgrade_reason}; {decision_override_reason}"
            if downgrade_reason
            else decision_override_reason
        )

    disagreement = _inter_agent_disagreement(phys, pred, critic)
    # Prefer structured inter-agent label when ML≠criteria or Critic objects
    if disagreement.get("has_disagreement"):
        dis_label = str(disagreement.get("level") or "Moderate")
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
        r_final=float(r_final_reported),
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
        # RC3: R_final always the calibrated score; overrides never clamp it.
        "final_system_confidence": r_final_reported,
        "r_final": r_final_reported,
        "decision": plain,
        "recommended_action": decision,
        "probabilities": pred.probabilities if pred else [],
        "route_trace": route_trace,
        "abstained": decision == "abstain",
        "critic_endorse": None if critic is None else bool(critic.endorse),
        "downgrade_reason": downgrade_reason,
        "consistency_note": consistency_note,
        "decision_override_reason": decision_override_reason,
        "physics_skipped": physics_skipped,
        "recheck_payloads": recheck_payloads or {},
        "r_vector": {**r_parts, "r_final": r_final_reported},
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
