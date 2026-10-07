"""LangGraph-style coordinator with Gemini planner + rule-based fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import hashlib
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from rockburst.agents.rechecks import (
    alt_imputer_reprediction,
    cross_check_model_reprediction,
    needs_extra_validation,
    physics_recheck_perturbed_tolerance,
    targeted_perturbation_recheck,
    validation_objections_from_rechecks,
)
from rockburst.agents.schemas import PhysicsOutput
from rockburst.agents.tools import (
    fuse_assessment,
    run_critic,
    run_data_quality,
    run_physics,
    run_predict,
    run_uncertainty,
)
from rockburst.config import ROOT, resolve_path
from rockburst.physics.precheck import should_skip_geomechanics

Action = Literal[
    "data_quality",
    "geomechanics",
    "predict",
    "uncertainty",
    "extra_validation",
    "critic",
    "recheck_physics",
    "recheck_data",
    "recheck_model",
    "finalize",
]

ALLOWED_ACTIONS = frozenset(
    {
        "data_quality",
        "geomechanics",
        "predict",
        "uncertainty",
        "extra_validation",
        "critic",
        "recheck_physics",
        "recheck_data",
        "recheck_model",
        "finalize",
    }
)

# Actions the LLM may propose (rechecks are rules-driven after Critic)
LLM_ALLOWED_ACTIONS = frozenset(
    {
        "data_quality",
        "geomechanics",
        "predict",
        "uncertainty",
        "extra_validation",
        "critic",
        "finalize",
    }
)


class PlannerAction(BaseModel):
    action: Action


PLANNER_STATS: dict[str, int] = {
    "api_calls": 0,
    "cache_hits": 0,
    "fallbacks": 0,
    "llm_steps": 0,
    "rules_steps": 0,
    "rejected": 0,
}


@dataclass
class CaseState:
    case: dict[str, Any]
    loop: int = 0
    max_loops: int = 8
    data_q: Any = None
    phys: Any = None
    pred: Any = None
    unc: Any = None
    critic: Any = None
    route_trace: list[str] = field(default_factory=list)
    recheck_counts: dict[str, int] = field(
        default_factory=lambda: {"physics": 0, "data": 0, "model": 0}
    )
    physics_skipped: bool = False
    extra_validation_done: bool = False
    recheck_payloads: dict[str, Any] = field(default_factory=dict)
    validation_objections: list[str] = field(default_factory=list)
    done: bool = False
    result: dict[str, Any] | None = None


def action_preconditions_met(action: str, state: CaseState) -> bool:
    """Reject actions whose required inputs do not exist yet."""
    if action == "data_quality":
        return True
    if action == "geomechanics":
        return state.data_q is not None
    if action == "predict":
        return state.data_q is not None
    if action == "uncertainty":
        return state.pred is not None
    if action == "extra_validation":
        return state.pred is not None and state.unc is not None and state.data_q is not None
    if action == "critic":
        return state.pred is not None and state.unc is not None
    if action in {"recheck_physics", "recheck_data", "recheck_model"}:
        return state.critic is not None and not state.critic.endorse
    if action == "finalize":
        return state.unc is not None and state.pred is not None
    return False


def rule_based_next_action(
    state: CaseState, cfg: dict[str, Any], *, allow_critic: bool = True
) -> Action:
    """Deterministic planner implementing required routing predicates."""
    rt = cfg.get("routing", {})
    if state.loop >= state.max_loops:
        return "finalize"
    if state.data_q is None:
        return "data_quality"

    # Conditional geomechanics (pre-registered skip rule)
    if state.phys is None and not state.physics_skipped:
        if should_skip_geomechanics(state.case, cfg):
            # Mark skip via placeholder physics; actual skip recorded in step
            return "geomechanics"  # step will detect clean precheck and skip-run
        return "geomechanics"

    if state.pred is None:
        return "predict"
    if state.unc is None:
        return "uncertainty"

    # Extra validation branch (missing D / imputed / low R_data)
    if (
        allow_critic
        and not state.extra_validation_done
        and needs_extra_validation(state.data_q, cfg)
    ):
        return "extra_validation"

    need_critic = False
    if allow_critic:
        if state.unc and state.unc.model_disagreement >= rt.get(
            "disagreement_threshold", 0.35
        ):
            need_critic = True
        if state.unc and len(state.unc.prediction_set) >= rt.get("large_set_size", 3):
            need_critic = True
        if state.data_q and state.data_q.ood_flag:
            need_critic = True
        if state.phys and state.phys.physics_score < 0.6:
            need_critic = True
        if state.validation_objections:
            need_critic = True

    if need_critic and state.critic is None:
        return "critic"

    if state.critic is not None and not state.critic.endorse:
        req = state.critic.request_recheck
        if req == "physics" and state.recheck_counts["physics"] < 1:
            state.recheck_counts["physics"] += 1
            state.critic = None
            return "recheck_physics"
        if req == "data" and state.recheck_counts["data"] < 1:
            state.recheck_counts["data"] += 1
            state.critic = None
            return "recheck_data"
        if req == "model" and state.recheck_counts["model"] < 1:
            state.recheck_counts["model"] += 1
            state.critic = None
            return "recheck_model"
        return "finalize"
    return "finalize"


def _cache_path(cfg: dict[str, Any], prompt: str) -> Path:
    cache_rel = cfg.get("gemini", {}).get("cache_dir", "artifacts/planner_cache")
    cache_dir = ROOT / cache_rel
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]
    return cache_dir / f"{key}.json"


def gemini_next_action(state: CaseState, cfg: dict[str, Any]) -> Action | None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    model_id = os.getenv(
        "GEMINI_PLANNER_MODEL",
        cfg.get("gemini", {}).get("planner_model", "gemini-3.8-flash"),
    )
    temperature = float(cfg.get("gemini", {}).get("temperature", 0.0))
    max_retries = int(cfg.get("gemini", {}).get("max_retries", 5))
    summary = {
        "loop": state.loop,
        "max_loops": state.max_loops,
        "has_data_q": state.data_q is not None,
        "has_phys": state.phys is not None,
        "has_pred": state.pred is not None,
        "has_unc": state.unc is not None,
        "has_critic": state.critic is not None,
        "data_issues": getattr(state.data_q, "issues", None),
        "ood": getattr(state.data_q, "ood_flag", None),
        "physics_score": getattr(state.phys, "physics_score", None),
        "disagreement": getattr(state.unc, "model_disagreement", None),
        "prediction_set": getattr(state.unc, "prediction_set", None),
        "critic": state.critic.model_dump() if state.critic is not None else None,
        "allowed_actions": sorted(LLM_ALLOWED_ACTIONS),
    }
    prompt = (
        "You are the Coordinator for a rockburst assessment graph. "
        'Respond with JSON only: {"action": "<one of allowed_actions>"}. '
        "Prefer data_quality then geomechanics then predict then uncertainty. "
        "Call critic on OOD, low physics, or strong disagreement. "
        "You may finalize when evidence is adequate or to abstain when inadequate. "
        f"State: {json.dumps(summary, sort_keys=True)}"
    )
    cpath = _cache_path(cfg, prompt + "|" + model_id)
    if cpath.exists():
        try:
            cached = json.loads(cpath.read_text())
            action = PlannerAction.model_validate(cached).action
            if action not in LLM_ALLOWED_ACTIONS:
                return None
            PLANNER_STATS["cache_hits"] += 1
            return action
        except (ValidationError, json.JSONDecodeError, KeyError):
            pass

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)
            PLANNER_STATS["api_calls"] += 1
            resp = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            text = (resp.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return None
            payload = json.loads(text[start : end + 1])
            parsed = PlannerAction.model_validate(payload)
            if parsed.action not in LLM_ALLOWED_ACTIONS:
                return None
            cpath.write_text(json.dumps({"action": parsed.action}))
            return parsed.action
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                time.sleep(min(2**attempt, 30))
                continue
            return None
    _ = last_err
    return None


def _fold_xy(cfg: dict[str, Any], reference: pd.DataFrame) -> tuple[Any, Any]:
    """Training frame for alt-imputer / cross-check; prefer fold inject."""
    if cfg.get("_fold_X") is not None and cfg.get("_fold_y") is not None:
        return cfg["_fold_X"], cfg["_fold_y"]
    from rockburst.ml.pipelines import prepare_xy

    X, y, _ = prepare_xy(reference)
    return X, y


def _run_extra_validation(state: CaseState, cfg: dict[str, Any], reference: pd.DataFrame) -> None:
    X_tr, y_tr = _fold_xy(cfg, reference)
    fold_pipe = cfg.get("_fold_pipeline")
    primary_imp = "knn"
    name = str(cfg.get("_fold_model_name") or "")
    if "__" in name:
        primary_imp = name.split("__", 1)[1]
    pred = state.pred
    primary_proba = np.asarray(pred.probabilities, dtype=float)
    primary_grade = int(pred.predicted_grade)
    missing = list(getattr(state.data_q, "missing_features", []) or [])
    flagged = missing or [
        c
        for c in ["sigma_theta", "sigma_c", "sigma_t", "scf", "b1", "b2", "wet", "d"]
        if state.case.get(c) is None
        or (
            isinstance(state.case.get(c), float)
            and not np.isfinite(float(state.case[c]))
        )
    ]
    alt = alt_imputer_reprediction(
        state.case,
        cfg,
        X_train=X_tr,
        y_train=y_tr,
        primary_imputer=primary_imp,
        primary_proba=primary_proba,
        primary_grade=primary_grade,
    )
    state.recheck_payloads["alt_imputer"] = alt
    if fold_pipe is not None:
        pert = targeted_perturbation_recheck(
            state.case,
            cfg,
            pipeline=fold_pipe,
            flagged_features=flagged,
            primary_grade=primary_grade,
            primary_proba=primary_proba,
        )
        state.recheck_payloads["targeted_perturbation"] = pert
    state.validation_objections = validation_objections_from_rechecks(
        state.recheck_payloads
    )
    state.extra_validation_done = True


def step(
    state: CaseState,
    reference: pd.DataFrame,
    cfg: dict[str, Any],
    use_llm: bool,
    *,
    allow_critic: bool = True,
) -> CaseState:
    action: Action
    source = "rules"
    if use_llm:
        llm_action = gemini_next_action(state, cfg)
        if llm_action is not None:
            if not action_preconditions_met(llm_action, state):
                state.route_trace.append(f"rejected:{llm_action}")
                PLANNER_STATS["rejected"] += 1
                action = rule_based_next_action(
                    state, cfg, allow_critic=allow_critic
                )
                source = "fallback"
                PLANNER_STATS["fallbacks"] += 1
            else:
                action = llm_action
                source = "llm"
                PLANNER_STATS["llm_steps"] += 1
        else:
            action = rule_based_next_action(state, cfg, allow_critic=allow_critic)
            source = "fallback"
            PLANNER_STATS["fallbacks"] += 1
    else:
        action = rule_based_next_action(state, cfg, allow_critic=allow_critic)
        PLANNER_STATS["rules_steps"] += 1

    if action == "critic" and not allow_critic:
        action = "finalize"

    # Guard even rules path for safety (e.g. finalize before unc)
    if not action_preconditions_met(action, state) and action != "finalize":
        state.route_trace.append(f"rejected:{action}")
        PLANNER_STATS["rejected"] += 1
        action = rule_based_next_action(state, cfg, allow_critic=allow_critic)
        source = "fallback"

    state.route_trace.append(f"{source}:{action}")
    state.loop += 1

    fold_pipe = cfg.get("_fold_pipeline")
    fold_ens = cfg.get("_fold_ensemble")
    fold_conf = cfg.get("_conformal_state")

    if action == "data_quality":
        state.data_q = run_data_quality(state.case, reference, cfg)
    elif action == "geomechanics":
        if should_skip_geomechanics(state.case, cfg):
            state.physics_skipped = True
            state.phys = PhysicsOutput(
                physics_score=1.0,
                exact_checks={"skipped": True, "reason": "cheap_precheck_clean"},
                empirical_checks={},
                issues=[],
            )
            state.route_trace[-1] = f"{source}:geomechanics_skipped"
        else:
            state.phys = run_physics(state.case, cfg)
    elif action == "predict":
        state.pred = run_predict(
            state.case, cfg, pipeline=fold_pipe, model_name=cfg.get("_fold_model_name")
        )
    elif action == "uncertainty":
        if state.pred is None:
            state.pred = run_predict(
                state.case,
                cfg,
                pipeline=fold_pipe,
                model_name=cfg.get("_fold_model_name"),
            )
        state.unc = run_uncertainty(
            state.case,
            state.pred,
            cfg,
            pipeline=fold_pipe,
            ensemble=fold_ens,
            conformal=fold_conf,
        )
    elif action == "extra_validation":
        _run_extra_validation(state, cfg, reference)
    elif action == "critic":
        state.critic = run_critic(
            state.data_q,
            state.phys,
            state.pred,
            state.unc,
            cfg,
            validation_objections=state.validation_objections,
        )
    elif action == "recheck_physics":
        payload = physics_recheck_perturbed_tolerance(state.case, cfg)
        state.recheck_payloads["physics_tol"] = payload
        # Update phys score from stated band (new information vs first pass)
        state.phys = PhysicsOutput(
            physics_score=float(payload["physics_score_stated"]),
            exact_checks={"borderline": payload["borderline_relations"]},
            empirical_checks={},
            issues=list(payload.get("issues_stated") or []),
        )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
    elif action == "recheck_data":
        X_tr, y_tr = _fold_xy(cfg, reference)
        name = str(cfg.get("_fold_model_name") or "")
        primary_imp = name.split("__", 1)[1] if "__" in name else "knn"
        pred = state.pred
        alt = alt_imputer_reprediction(
            state.case,
            cfg,
            X_train=X_tr,
            y_train=y_tr,
            primary_imputer=primary_imp,
            primary_proba=np.asarray(pred.probabilities),
            primary_grade=int(pred.predicted_grade),
        )
        state.recheck_payloads["alt_imputer"] = alt
        missing = list(getattr(state.data_q, "missing_features", []) or [])
        if fold_pipe is not None:
            pert = targeted_perturbation_recheck(
                state.case,
                cfg,
                pipeline=fold_pipe,
                flagged_features=missing or None,
                primary_grade=int(pred.predicted_grade),
                primary_proba=np.asarray(pred.probabilities),
            )
            state.recheck_payloads["targeted_perturbation"] = pert
        # Lower data reliability when large spread
        if state.data_q is not None and alt.get("large_spread"):
            state.data_q = state.data_q.model_copy(
                update={
                    "data_reliability": float(
                        max(0.0, state.data_q.data_reliability - 0.15)
                    )
                }
            )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
    elif action == "recheck_model":
        X_tr, y_tr = _fold_xy(cfg, reference)
        name = str(cfg.get("_fold_model_name") or "")
        primary_imp = name.split("__", 1)[1] if "__" in name else "knn"
        pred = state.pred
        cross = cross_check_model_reprediction(
            state.case,
            cfg,
            X_train=X_tr,
            y_train=y_tr,
            primary_imputer=primary_imp,
            primary_grade=int(pred.predicted_grade),
            primary_proba=np.asarray(pred.probabilities),
        )
        state.recheck_payloads["cross_check"] = cross
        if state.unc is not None and cross.get("large_disagreement"):
            state.unc.model_disagreement = float(
                max(state.unc.model_disagreement, cross.get("soft_disagreement", 0.35))
            )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
    elif action == "finalize":
        state.result = fuse_assessment(
            state.data_q,
            state.phys,
            state.pred,
            state.unc,
            cfg,
            state.route_trace,
            critic=state.critic,
            recheck_payloads=state.recheck_payloads,
            physics_skipped=state.physics_skipped,
        )
        state.done = True
    return state


def run_graph(
    case: dict[str, Any],
    reference: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    use_llm: bool = False,
    allow_critic: bool = True,
) -> dict[str, Any]:
    state = CaseState(
        case=case, max_loops=int(cfg.get("routing", {}).get("max_loops", 8))
    )
    while not state.done and state.loop < state.max_loops + 2:
        state = step(
            state, reference, cfg, use_llm=use_llm, allow_critic=allow_critic
        )
    if not state.done:
        state.result = fuse_assessment(
            state.data_q,
            state.phys,
            state.pred,
            state.unc,
            cfg,
            state.route_trace + ["forced_finalize"],
            critic=state.critic,
            recheck_payloads=state.recheck_payloads,
            physics_skipped=state.physics_skipped,
        )
        state.result["recommended_action"] = "abstain"
        state.result["abstained"] = True
        state.result["decision"] = (
            "Forced stop at loop limit; abstain pending review."
        )
    return state.result or {}


def assess_case(
    cfg: dict[str, Any], case_no: int, use_llm: bool = False
) -> dict[str, Any]:
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        path = resolve_path(cfg, "raw_csv")
    df = pd.read_csv(path)
    row = df.loc[df["no"] == case_no]
    if row.empty:
        raise ValueError(f"Case {case_no} not found")
    case = row.iloc[0].to_dict()
    from rockburst.quality.fold_ref import build_fold_reference

    ref = build_fold_reference(df, np.arange(len(df)), case)
    if len(ref) < 5:
        ref = df
    result = run_graph(case, ref, cfg, use_llm=use_llm)
    traces = resolve_path(cfg, "traces")
    traces.mkdir(parents=True, exist_ok=True)
    out_path = traces / f"case_{case_no}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["trace_path"] = str(out_path)
    return result
