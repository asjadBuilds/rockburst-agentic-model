"""Deterministic rules-FSM coordinator with an optional Gemini routing planner."""

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
    monotonic_tendency_recheck,
    needs_extra_validation,
    physics_recheck_perturbed_tolerance,
    targeted_perturbation_recheck,
    validation_objections_from_rechecks,
)
from rockburst.agents.schemas import PhysicsOutput
from rockburst.agents.tools import (
    format_engineering_decision,
    fuse_assessment,
    run_critic,
    run_data_quality,
    run_predict,
    run_uncertainty,
)
from rockburst.config import ROOT, resolve_path
from rockburst.physics.precheck import physics_for_geomechanics

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
    "forced_abstain",
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
        "forced_abstain",
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
    max_loops: int = 24
    max_critic_recheck_cycles: int = 3
    critic_recheck_cycles: int = 0
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
    # TC8: explanations belong to one assessment, never the shared fold config.
    local_explanation: dict[str, float] | None = None
    # RM1: snapshots so Phase 5 can report how often rechecks change outcomes.
    recheck_effects: list[dict[str, Any]] = field(default_factory=list)
    # RM2: accumulate Critic rules across visits (recheck clears live critic).
    critic_rules_union: list[str] = field(default_factory=list)
    # TC1: durable per-rule lifecycle; resolved rules are suppressed on re-entry.
    objection_records: list[dict[str, Any]] = field(default_factory=list)
    resolved_critic_rules: list[str] = field(default_factory=list)
    termination_reason: str | None = None
    done: bool = False
    result: dict[str, Any] | None = None


def _record_critic_rules(state: CaseState) -> None:
    """Union rules from this Critic visit into ``critic_rules_union`` (RM2)."""
    if state.critic is None:
        return
    for rule in list(getattr(state.critic, "rules_fired", []) or []):
        if rule not in state.critic_rules_union:
            state.critic_rules_union.append(str(rule))
    for record in list(getattr(state.critic, "objection_records", []) or []):
        rule = str(record.get("rule") or "")
        if not rule:
            continue
        existing = next(
            (r for r in state.objection_records if r.get("rule") == rule), None
        )
        if existing is None:
            state.objection_records.append(dict(record))
        elif existing.get("status") != "resolved":
            existing.update(record)


def _resolve_rechecked_objections(
    state: CaseState, *, kind: str, payload: dict[str, Any]
) -> None:
    """Resolve only objections for which the recheck added affirmative evidence."""
    prior_rules = set(getattr(state.critic, "rules_fired", []) or [])
    resolved: set[str] = set()
    reason = ""
    if kind == "model":
        delta = abs(int(payload.get("cross_grade", -99)) - int(payload.get("primary_grade", 99)))
        soft = float(payload.get("soft_disagreement", 1.0))
        threshold = float(payload.get("resolution_threshold", 0.35))
        if delta <= 1 and soft < threshold:
            resolved |= prior_rules & {
                "ensemble_disagreement",
                "large_prediction_set",
                "edge_of_wide_set",
                "implausible_explanation_direction",
            }
            reason = "independent_model_within_one_grade_and_below_disagreement_threshold"
    elif kind == "data":
        alt = payload.get("alt") or payload
        perturbation = payload.get("targeted_perturbation") or {}
        stable = (
            (bool(alt.get("skipped")) or not bool(alt.get("large_spread")))
            and not bool(perturbation.get("low_stability"))
        )
        if stable:
            resolved |= prior_rules & {"validation_objection"}
            reason = "imputer_and_targeted_perturbation_checks_stable_or_not_applicable"
    elif kind == "physics":
        issues = " ".join(str(x) for x in payload.get("issues_stated") or [])
        if not payload.get("low_physics"):
            if "scf inconsistency" not in issues and "b2 inconsistency" not in issues:
                resolved |= prior_rules & {"identity_violation"}
            if "large_b1_identity_gap" not in issues:
                resolved |= prior_rules & {"large_b1_identity_gap"}
            consensus = payload.get("criteria_consensus_grade")
            primary = state.pred.predicted_grade if state.pred is not None else None
            if (
                consensus is not None
                and primary is not None
                and abs(int(consensus) - int(primary)) <= 1
            ):
                resolved |= prior_rules & {"ml_vs_criteria"}
            reason = "tightened_physics_recheck_satisfied_relevant_relation"

    for record in state.objection_records:
        if record.get("rule") not in prior_rules:
            continue
        if record.get("rule") in resolved:
            record.update(
                {"status": "resolved", "resolution_reason": reason, "recheck": kind}
            )
            rule = str(record["rule"])
            if rule not in state.resolved_critic_rules:
                state.resolved_critic_rules.append(rule)
        else:
            record.update(
                {
                    "status": "open",
                    "resolution_reason": "recheck_did_not_clear_objection",
                    "recheck": kind,
                }
            )


def _widen_prediction_set(state: CaseState, extra_grades: list[int]) -> None:
    """Expand conformal set with extra grades (contiguous) so rechecks can change it."""
    if state.unc is None:
        return
    grades = [int(g) for g in (list(state.unc.prediction_set) + list(extra_grades)) if 0 <= int(g) <= 3]
    if not grades:
        return
    lo, hi = min(grades), max(grades)
    new_set = list(range(lo, hi + 1))
    if new_set != list(state.unc.prediction_set):
        state.unc = state.unc.model_copy(update={"prediction_set": new_set})


def _maybe_attach_local_explanation(state: CaseState, cfg: dict[str, Any]) -> None:
    """Attach local attributions to this case so fold configs cannot leak them."""
    if state.local_explanation is not None:
        return
    rt = cfg.get("routing", {})
    if cfg.get("_skip_local_explanation") or rt.get("skip_local_explanation"):
        return
    pipe = cfg.get("_fold_pipeline")
    if pipe is None or state.pred is None:
        return
    from rockburst.ml.explain import local_permutation_attributions
    from rockburst.physics.relations import FEATURE_COLS

    row = {c: state.case.get(c, np.nan) for c in FEATURE_COLS}
    X_row = pd.DataFrame([row])
    for c in FEATURE_COLS:
        X_row[f"observed_{c}"] = (~X_row[c].isna()).astype(float)
    try:
        attrib = local_permutation_attributions(
            pipe,
            X_row,
            feature_names=list(FEATURE_COLS),
            predicted_grade=int(state.pred.predicted_grade),
            n_repeats=4,
            seed=int(cfg.get("seed", 42)) + int(state.case.get("no") or 0),
        )
        # Severity drivers only (Critic direction check).
        state.local_explanation = {
            k: float(attrib[k])
            for k in ("scf", "wet", "sigma_theta")
            if k in attrib
        }
    except Exception:  # noqa: BLE001 — never block Critic on explain failure
        state.local_explanation = {}


def _apply_critic_rules_to_result(state: CaseState) -> None:
    """Prefer union of Critic fires over the final (possibly cleared) critic."""
    if state.result is None:
        return
    if state.critic_rules_union:
        state.result["critic_rules_fired"] = list(state.critic_rules_union)


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
    if action == "forced_abstain":
        return True
    return False


def _missing_evidence(state: CaseState) -> bool:
    return state.pred is None or state.unc is None or state.data_q is None


def _unresolved_objections(state: CaseState) -> bool:
    if state.critic is not None and not state.critic.endorse:
        return True
    return bool(state.validation_objections)


def _case_missing_d(case: dict[str, Any] | None) -> bool:
    if not case:
        return False
    v = case.get("d")
    if v is None:
        return True
    try:
        return not bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return True


def _agent_snapshot(state: CaseState) -> dict[str, Any]:
    """Lightweight evidence snapshot for RM1 recheck-effect counters."""
    pset = list(state.unc.prediction_set) if state.unc is not None else []
    return {
        "physics_score": (
            float(state.phys.physics_score) if state.phys is not None else None
        ),
        "data_reliability": (
            float(state.data_q.data_reliability) if state.data_q is not None else None
        ),
        "model_disagreement": (
            float(state.unc.model_disagreement) if state.unc is not None else None
        ),
        "prediction_set": pset,
        "has_criteria": bool(
            state.phys is not None
            and (state.phys.empirical_checks or {}).get("criteria_as_given")
        ),
    }


def _merge_physics_recheck(
    first: PhysicsOutput | None,
    payload: dict[str, Any],
) -> PhysicsOutput:
    """RM1: merge recheck into first-pass PhysicsOutput; keep criteria."""
    base_exact = dict(first.exact_checks) if first is not None else {}
    base_emp = dict(first.empirical_checks) if first is not None else {}
    base_issues = list(first.issues) if first is not None else []
    score = float(payload.get("physics_score_stated", 0.5))
    borderline = list(payload.get("borderline_relations") or [])
    if borderline:
        score = max(0.0, score - 0.05 * len(borderline))
    exact = {
        **base_exact,
        "recheck_borderline": borderline,
        "recheck_physics_score_stated": float(
            payload.get("physics_score_stated", score)
        ),
        "recheck_physics_score_tight": float(
            payload.get("physics_score_tight", score)
        ),
    }
    issues = list(base_issues)
    for iss in payload.get("issues_stated") or []:
        if iss not in issues:
            issues.append(iss)
    if borderline:
        issues.append(f"recheck_borderline_relations={borderline}")
    # Preserve first-pass criteria so second Critic can still see ml_vs_criteria.
    emp = dict(base_emp)
    emp["recheck_tolerance"] = {
        "borderline_relations": borderline,
        "low_physics": bool(payload.get("low_physics")),
    }
    return PhysicsOutput(
        physics_score=float(np.clip(score, 0.0, 1.0)),
        exact_checks=exact,
        empirical_checks=emp,
        issues=issues,
    )


def _routing_meta(state: CaseState) -> dict[str, Any]:
    """Documented termination / budget fields (RC5) + RM1 recheck effects."""
    changed_conf = sum(
        1
        for e in state.recheck_effects
        if e.get("changed_physics_score")
        or e.get("changed_data_reliability")
        or e.get("changed_disagreement")
    )
    changed_set = sum(1 for e in state.recheck_effects if e.get("changed_set"))
    return {
        "termination_reason": state.termination_reason,
        "critic_recheck_cycles": int(state.critic_recheck_cycles),
        "max_critic_recheck_cycles": int(state.max_critic_recheck_cycles),
        "steps_taken": int(state.loop),
        "max_loops": int(state.max_loops),
        "missing_d": _case_missing_d(state.case),
        "recheck_effects": list(state.recheck_effects),
        "critic_objections": list(state.objection_records),
        "resolved_critic_rules": list(state.resolved_critic_rules),
        "recheck_changed_evidence_count": int(changed_conf + changed_set),
        "termination_conditions": [
            "finalize: evidence complete and Critic endorses or is not required",
            "forced_abstain_critic_cycles: Critic→recheck budget exhausted with unresolved objections",
            "forced_abstain_max_steps: absolute step ceiling with unresolved objections/evidence",
            "forced_abstain_safety: loop exited without finalize",
        ],
    }


def apply_forced_abstain(
    state: CaseState,
    cfg: dict[str, Any],
    *,
    reason: str = "forced_abstain_max_steps",
) -> CaseState:
    """Hard stop: fuse available evidence and abstain (non-convergence safeguard)."""
    state.termination_reason = reason
    trace = list(state.route_trace)
    if "forced_finalize" not in trace:
        trace = trace + ["forced_finalize"]
    state.result = fuse_assessment(
        state.data_q,
        state.phys,
        state.pred,
        state.unc,
        cfg,
        trace,
        critic=state.critic,
        recheck_payloads=state.recheck_payloads,
        physics_skipped=state.physics_skipped,
        case=state.case,
    )
    state.result["recommended_action"] = "abstain"
    state.result["abstained"] = True
    disagreement = state.result.get("inter_agent_disagreement") or {}
    state.result["decision"] = (
        format_engineering_decision(
            predicted_grade=(state.pred.predicted_grade if state.pred else None),
            prediction_set=(list(state.unc.prediction_set) if state.unc else []),
            r_final=float(state.result.get("r_final") or 0.0),
            action="abstain",
            disagreement=disagreement,
        )
        + f" Forced stop—non-convergence safeguard ({reason}); engineering review required."
    )
    state.result["route_trace"] = trace
    state.result.update(_routing_meta(state))
    _apply_critic_rules_to_result(state)
    state.done = True
    return state


def rule_based_next_action(
    state: CaseState, cfg: dict[str, Any], *, allow_critic: bool = True
) -> Action:
    """Deterministic planner implementing required routing predicates.

    Termination (RC5):
    - Primary non-convergence: ``critic_recheck_cycles >= max_critic_recheck_cycles``
      with unresolved Critic/validation objections → ``forced_abstain``.
    - Safety ceiling: ``loop >= max_loops`` with unresolved objections/missing
      evidence → ``forced_abstain``; otherwise ``finalize``.
    - Ordinary finalize when evidence is complete and Critic endorses / not needed.
    Missing-D and present-D share the same Critic→recheck cycle budget; extra
    validation is not charged against that budget.
    """
    rt = cfg.get("routing", {})
    # Absolute step ceiling (safety; set well above longest legitimate path).
    if state.loop >= state.max_loops:
        if _unresolved_objections(state) or _missing_evidence(state):
            return "forced_abstain"
        return "finalize"
    if state.data_q is None:
        return "data_quality"

    # Geomechanics always runs once; clean cases keep full criteria (RM4).
    if state.phys is None and not state.physics_skipped:
        return "geomechanics"

    if state.pred is None:
        return "predict"
    if state.unc is None:
        return "uncertainty"

    # TM3: every case receives the monotonic check. Expensive imputer/targeted
    # checks remain conditional inside _run_extra_validation.
    if not state.extra_validation_done:
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
        if state.phys and any(
            key in issue
            for issue in state.phys.issues
            for key in (
                "tangential_stress_exceeds_strength",
                "depth_stress_implausible",
            )
        ):
            need_critic = True
        if state.phys and state.pred:
            criteria = (state.phys.empirical_checks or {}).get("criteria_as_given") or {}
            consensus = criteria.get("consensus_grade")
            if consensus is not None and int(consensus) >= 0:
                need_critic = need_critic or int(consensus) != int(state.pred.predicted_grade)

    # RC5: Critic→recheck cycle budget (not total graph steps).
    # If triggers remain after the budget is spent, forced abstain — do not
    # quietly finalize just because the last recheck cleared critic state.
    if need_critic and state.critic is None:
        if state.critic_recheck_cycles >= state.max_critic_recheck_cycles:
            return "forced_abstain"
        return "critic"

    # C1: keep Critic until the recheck tool runs (preconditions require it).
    if state.critic is not None and not state.critic.endorse:
        req = state.critic.request_recheck
        if req in {"physics", "data", "model"}:
            # Identical deterministic rechecks are never repeated.
            if state.recheck_counts.get(req, 0) > 0:
                state.termination_reason = "finalize_unresolved_no_new_evidence"
                return "finalize"
            if state.critic_recheck_cycles < state.max_critic_recheck_cycles:
                state.recheck_counts[req] = int(state.recheck_counts.get(req, 0)) + 1
                return {
                    "physics": "recheck_physics",
                    "data": "recheck_data",
                    "model": "recheck_model",
                }[req]
            return "forced_abstain"
        # No recheck requested but still objecting — escalate via fuse path
        if state.critic_recheck_cycles >= state.max_critic_recheck_cycles:
            return "forced_abstain"
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
        cfg.get("gemini", {}).get("planner_model", "gemini-3.5-flash"),
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


def _validation_pipeline(cfg: dict[str, Any]) -> Any | None:
    """Return the current fold pipeline or the shipped production pipeline."""
    pipe = cfg.get("_fold_pipeline")
    if pipe is not None:
        return pipe
    try:
        import joblib

        model_path = (
            Path(cfg.get("_root", "."))
            / cfg["paths"]["models"]
            / "best_baseline.joblib"
        )
        if model_path.exists():
            return joblib.load(model_path).get("pipeline")
    except (KeyError, OSError, ValueError, TypeError):
        return None
    return None


def _run_extra_validation(state: CaseState, cfg: dict[str, Any], reference: pd.DataFrame) -> None:
    X_tr, y_tr = _fold_xy(cfg, reference)
    fold_pipe = _validation_pipeline(cfg)
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
    conditional = needs_extra_validation(state.data_q, cfg, phys=state.phys)
    if conditional:
        alt = alt_imputer_reprediction(
            state.case,
            cfg,
            X_train=X_tr,
            y_train=y_tr,
            primary_imputer=primary_imp,
            primary_proba=primary_proba,
            primary_grade=primary_grade,
            primary_pipeline=fold_pipe,
            missing_features=missing,
        )
        state.recheck_payloads["alt_imputer"] = alt
    if fold_pipe is not None and conditional:
        pert = targeted_perturbation_recheck(
            state.case,
            cfg,
            pipeline=fold_pipe,
            flagged_features=flagged,
            primary_grade=primary_grade,
            primary_proba=primary_proba,
        )
        state.recheck_payloads["targeted_perturbation"] = pert
    if fold_pipe is not None:
        mono = monotonic_tendency_recheck(
            state.case,
            pipeline=fold_pipe,
            primary_grade=primary_grade,
        )
        state.recheck_payloads["monotonic_tendency"] = mono
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
    if (
        not action_preconditions_met(action, state)
        and action not in {"finalize", "forced_abstain"}
    ):
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
        # RM4: cheap identities/ranges; clean → one criteria; dirty → full check_case.
        res = physics_for_geomechanics(state.case, cfg)
        state.phys = PhysicsOutput(**res.to_dict())
        policy = (res.empirical_checks or {}).get("data_policy") or {}
        if policy.get("b") == "recompute skipped on clean RM4 path":
            state.physics_skipped = True
            exact = dict(state.phys.exact_checks or {})
            exact["skipped"] = True
            exact["reason"] = "precheck_clean_criteria_retained"
            state.phys = state.phys.model_copy(update={"exact_checks": exact})
            state.route_trace[-1] = f"{source}:geomechanics_skipped"
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
        # RM9: attach local attributions before Critic so direction rule can fire.
        _maybe_attach_local_explanation(state, cfg)
        state.critic = run_critic(
            state.data_q,
            state.phys,
            state.pred,
            state.unc,
            cfg,
            validation_objections=state.validation_objections,
            local_explanation=state.local_explanation,
            resolved_rules=set(state.resolved_critic_rules),
        )
        _record_critic_rules(state)
    elif action == "recheck_physics":
        before = _agent_snapshot(state)
        payload = physics_recheck_perturbed_tolerance(state.case, cfg)
        state.recheck_payloads["physics_tol"] = payload
        # RM1: merge into first-pass PhysicsOutput; never wipe criteria.
        state.phys = _merge_physics_recheck(state.phys, payload)
        # Borderline physics → widen set around predicted grade (set can change).
        if payload.get("borderline_relations") or payload.get("low_physics"):
            g = int(state.pred.predicted_grade) if state.pred is not None else 1
            _widen_prediction_set(state, [max(0, g - 1), g, min(3, g + 1)])
        after = _agent_snapshot(state)
        state.recheck_effects.append(
            {
                "kind": "physics",
                "before": before,
                "after": after,
                "changed_physics_score": before.get("physics_score")
                != after.get("physics_score"),
                "changed_set": before.get("prediction_set")
                != after.get("prediction_set"),
                "preserved_criteria": bool(after.get("has_criteria")),
            }
        )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
        _resolve_rechecked_objections(state, kind="physics", payload=payload)
        state.critic = None  # clear only after successful recheck
        state.critic_recheck_cycles += 1
    elif action == "recheck_data":
        before = _agent_snapshot(state)
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
            primary_pipeline=fold_pipe,
            missing_features=list(
                getattr(state.data_q, "missing_features", []) or []
            ),
        )
        state.recheck_payloads["alt_imputer"] = alt
        missing = list(getattr(state.data_q, "missing_features", []) or [])
        pert = None
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
        # Lower data reliability when large spread (can change R_final).
        if state.data_q is not None and alt.get("large_spread"):
            state.data_q = state.data_q.model_copy(
                update={
                    "data_reliability": float(
                        max(0.0, state.data_q.data_reliability - 0.15)
                    )
                }
            )
        # RM1: refresh set when alt-imputer / perturbation disagree.
        if alt.get("large_spread") or (pert and pert.get("low_stability")):
            g = int(pred.predicted_grade)
            extras = [max(0, g - 1), g, min(3, g + 1)]
            for ag in (alt.get("grades") or {}).values():
                extras.append(int(ag))
            _widen_prediction_set(state, extras)
        after = _agent_snapshot(state)
        state.recheck_effects.append(
            {
                "kind": "data",
                "before": before,
                "after": after,
                "changed_data_reliability": before.get("data_reliability")
                != after.get("data_reliability"),
                "changed_set": before.get("prediction_set")
                != after.get("prediction_set"),
            }
        )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
        _resolve_rechecked_objections(
            state,
            kind="data",
            payload={"alt": alt, "targeted_perturbation": pert or {}},
        )
        state.critic = None  # clear only after successful recheck
        state.critic_recheck_cycles += 1
    elif action == "recheck_model":
        before = _agent_snapshot(state)
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
            state.unc = state.unc.model_copy(
                update={
                    "model_disagreement": float(
                        max(
                            state.unc.model_disagreement,
                            cross.get("soft_disagreement", 0.35),
                        )
                    )
                }
            )
            # RM1: include cross-check grade so prediction_set can change.
            extras = [int(pred.predicted_grade)]
            if cross.get("cross_grade") is not None:
                extras.append(int(cross["cross_grade"]))
            _widen_prediction_set(state, extras)
        after = _agent_snapshot(state)
        state.recheck_effects.append(
            {
                "kind": "model",
                "before": before,
                "after": after,
                "changed_disagreement": before.get("model_disagreement")
                != after.get("model_disagreement"),
                "changed_set": before.get("prediction_set")
                != after.get("prediction_set"),
            }
        )
        state.validation_objections = validation_objections_from_rechecks(
            state.recheck_payloads
        )
        _resolve_rechecked_objections(state, kind="model", payload=cross)
        state.critic = None  # clear only after successful recheck
        state.critic_recheck_cycles += 1
    elif action == "forced_abstain":
        if state.critic_recheck_cycles >= state.max_critic_recheck_cycles:
            reason = "forced_abstain_critic_cycles"
        else:
            reason = "forced_abstain_max_steps"
        apply_forced_abstain(state, cfg, reason=reason)
    elif action == "finalize":
        if state.termination_reason is None:
            state.termination_reason = "finalize"
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
            case=state.case,
        )
        state.result.update(_routing_meta(state))
        _apply_critic_rules_to_result(state)
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
    rt = cfg.get("routing", {})
    state = CaseState(
        case=case,
        max_loops=int(rt.get("max_loops", 24)),
        max_critic_recheck_cycles=int(rt.get("max_critic_recheck_cycles", 3)),
    )
    # Allow one terminating step at loop == max_loops (finalize or forced_abstain).
    while not state.done and state.loop <= state.max_loops:
        state = step(
            state, reference, cfg, use_llm=use_llm, allow_critic=allow_critic
        )
    if not state.done:
        apply_forced_abstain(state, cfg, reason="forced_abstain_safety")
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
    # RC8: ship repo-relative paths only (no /home/... absolute traces).
    try:
        result["trace_path"] = str(out_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        result["trace_path"] = str(Path("artifacts") / "traces" / f"case_{case_no}.json")
    return result
