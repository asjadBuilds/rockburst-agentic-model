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
    p = cfg.get("physics", {})
    res = check_case(
        case,
        rel_tol=p.get("rel_tol", 0.05),
        abs_tol=p.get("abs_tol", 0.05),
        scf_ood_threshold=p.get("scf_ood_threshold", 2.0),
        wet_range=tuple(p.get("wet_range", [0.5, 40.0])),
        d_range=tuple(p.get("d_range", [50.0, 2500.0])),
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

    if ensemble is None:
        ens_path = models_dir / "ensemble.joblib"
        ensemble_probas = []
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
    request: str = "none"
    endorse = True
    rt = cfg.get("routing", {})
    objections = list(validation_objections or [])

    if phys and phys.physics_score < 0.6:
        challenges.append("Low physics consistency")
        if pred and pred.r_model > 0.8:
            contradictions.append("High model confidence despite physics failure")
            request = "physics"
            endorse = False
    if data_q and data_q.ood_flag and pred and pred.r_model > 0.75:
        contradictions.append("OOD input with overconfident prediction")
        request = "data"
        endorse = False
    if unc and unc.model_disagreement >= rt.get("disagreement_threshold", 0.35):
        challenges.append("Strong ensemble disagreement")
        request = "model" if request == "none" else request
        endorse = False
    if unc and len(unc.prediction_set) >= rt.get("large_set_size", 3):
        challenges.append("Large conformal prediction set")
        endorse = False
        if request == "none":
            request = "model"
    if phys and any("b1 inconsistency" in i for i in phys.issues):
        challenges.append("Derived B1 does not match sigma_c/sigma_t")
        if request == "none":
            request = "physics"
        endorse = False
    if objections:
        challenges.extend([f"validation:{o}" for o in objections])
        endorse = False
        if request == "none":
            if any("physics" in o for o in objections):
                request = "physics"
            elif any("imputer" in o or "perturbation" in o for o in objections):
                request = "data"
            else:
                request = "model"
    return CriticOutput(
        challenges=challenges,
        contradictions=contradictions,
        request_recheck=request if request in {"none", "physics", "data", "model"} else "none",  # type: ignore[arg-type]
        endorse=endorse,
    )


_DOWNGRADE = {
    "accept": "accept_with_caution",
    "accept_with_caution": "engineering_verification",
    "engineering_verification": "engineering_verification",
    "abstain": "abstain",
}


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
    r_vec = compute_reliability_features(data_q, phys, pred, unc)
    r_parts = r_vector_dict(r_vec)
    r_data = r_parts["r_data"]
    r_physics = r_parts["r_physics"]
    r_model = r_parts["r_model"]
    r_unc = r_parts["r_uncertainty"]
    r_stab = r_parts["r_stability"]
    fusion_path = Path(cfg.get("_root", ".")) / cfg["paths"]["models"] / "fusion.joblib"
    fusion = ReliabilityFusion.load(fusion_path) if fusion_path.exists() else ReliabilityFusion()
    if not fusion.is_fit:
        decision = "abstain"
        r_final = 0.0
        plain = (
            "Fusion stacker not fitted; abstain (no heuristic reliability weights)."
        )
        return {
            "predicted_grade": pred.predicted_grade if pred else None,
            "prediction_probability": float(max(pred.probabilities)) if pred else None,
            "prediction_set": unc.prediction_set if unc else [],
            "physics_consistency": r_physics,
            "input_reliability": r_data,
            "ood_status": "Yes" if (data_q and data_q.ood_flag) else "No",
            "prediction_stability": unc.prediction_stability if unc else "unknown",
            "agent_disagreement": "unknown",
            "final_system_confidence": r_final,
            "decision": plain,
            "recommended_action": decision,
            "probabilities": pred.probabilities if pred else [],
            "route_trace": route_trace,
            "abstained": True,
            "critic_endorse": None if critic is None else bool(critic.endorse),
            "downgrade_reason": "fusion_unfitted",
            "physics_skipped": physics_skipped,
            "recheck_payloads": recheck_payloads or {},
            "r_vector": {**r_parts, "r_final": r_final},
        }
    r_final = fusion.r_final(r_vec)
    decision = fusion.decide(r_final, cfg.get("routing", {}))
    if unc is not None:
        if len(unc.prediction_set) >= 3 and unc.stability_score < 0.55:
            decision = "abstain"
        elif unc.model_disagreement >= 0.5 and r_final < 0.75:
            decision = "abstain"
    if data_q is not None and phys is not None:
        if data_q.ood_flag and phys.physics_score < 0.7 and (pred is None or pred.r_model < 0.6):
            decision = "abstain"

    # Pre-registered: unresolved Critic objections downgrade decision one level
    downgrade_reason = None
    if critic is not None and not critic.endorse and decision != "abstain":
        new_decision = _DOWNGRADE.get(decision, decision)
        if new_decision != decision:
            downgrade_reason = (
                f"Critic unresolved (endorse=False, recheck={critic.request_recheck}); "
                f"downgraded {decision} → {new_decision}"
            )
            decision = new_decision

    grade = pred.predicted_grade if pred else None
    dis_label = "low"
    if unc:
        dis_label = (
            "high"
            if unc.model_disagreement >= 0.5
            else "moderate"
            if unc.model_disagreement >= 0.25
            else "low"
        )
    plain = {
        "accept": "Prediction accepted; routine monitoring recommended.",
        "accept_with_caution": "Accept with caution; verify key measurements.",
        "engineering_verification": "Confidence insufficient for routine accept; engineering verification recommended.",
        "abstain": "Inadequate confidence; abstain and request additional data/review.",
    }[decision]
    if downgrade_reason:
        plain = f"{plain} ({downgrade_reason})"
    return {
        "predicted_grade": grade,
        "prediction_probability": float(max(pred.probabilities)) if pred else None,
        "prediction_set": unc.prediction_set if unc else [],
        "physics_consistency": r_physics,
        "input_reliability": r_data,
        "ood_status": "Yes" if (data_q and data_q.ood_flag) else "No",
        "prediction_stability": unc.prediction_stability if unc else "unknown",
        "agent_disagreement": dis_label,
        "final_system_confidence": r_final,
        "decision": plain,
        "recommended_action": decision,
        "probabilities": pred.probabilities if pred else [],
        "route_trace": route_trace,
        "abstained": decision == "abstain",
        "critic_endorse": None if critic is None else bool(critic.endorse),
        "downgrade_reason": downgrade_reason,
        "physics_skipped": physics_skipped,
        "recheck_payloads": recheck_payloads or {},
        "r_vector": {**r_parts, "r_final": r_final},
    }
