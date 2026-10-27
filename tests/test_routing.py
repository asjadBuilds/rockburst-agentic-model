"""Routing / Critic / precheck unit tests (Item 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rockburst.agents.rechecks import needs_extra_validation
from rockburst.agents.schemas import CriticOutput, DataQualityOutput
from rockburst.agents.tools import fuse_assessment
from rockburst.config import load_config, resolve_path
from rockburst.graph.runner import (
    CaseState,
    action_preconditions_met,
    rule_based_next_action,
    run_graph,
    step,
)
from rockburst.physics.precheck import cheap_physics_precheck, should_skip_geomechanics


@pytest.fixture
def cfg_df():
    cfg = load_config()
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        pytest.skip("processed csv missing; run extract first")
    return cfg, pd.read_csv(path)


def test_physics_skipped_when_precheck_clean(cfg_df):
    cfg, df = cfg_df
    # Find a row that passes the cheap pre-check
    clean = None
    for _, row in df.iterrows():
        case = row.to_dict()
        if should_skip_geomechanics(case, cfg):
            clean = case
            break
    if clean is None:
        pytest.skip("no clean precheck row")
    result = run_graph(clean, df, cfg, use_llm=False, allow_critic=False)
    trace = " ".join(result.get("route_trace") or [])
    assert "geomechanics_skipped" in trace
    assert result.get("physics_skipped") is True


def test_physics_runs_when_precheck_fails(cfg_df):
    cfg, df = cfg_df
    # Missing D fails pre-check
    row = df[df["d"].isna()].iloc[0].to_dict()
    assert not should_skip_geomechanics(row, cfg)
    result = run_graph(row, df, cfg, use_llm=False, allow_critic=False)
    trace = " ".join(result.get("route_trace") or [])
    assert "geomechanics_skipped" not in trace
    assert ":geomechanics" in trace or "rules:geomechanics" in trace


def test_missing_d_triggers_extra_validation(cfg_df):
    cfg, df = cfg_df
    row = df[df["d"].isna()].iloc[0].to_dict()
    dq = DataQualityOutput(
        data_reliability=0.8,
        ood_flag=False,
        missing_features=["d"],
    )
    assert needs_extra_validation(dq, cfg)
    result = run_graph(row, df, cfg, use_llm=False, allow_critic=True)
    trace = " ".join(result.get("route_trace") or [])
    assert "extra_validation" in trace


def test_m6_extra_validation_without_critic(cfg_df):
    """M6 = coordinator + extra_validation; Critic disabled."""
    cfg, df = cfg_df
    row = df[df["d"].isna()].iloc[0].to_dict()
    result = run_graph(row, df, cfg, use_llm=False, allow_critic=False)
    actions = [t.split(":")[-1] for t in (result.get("route_trace") or [])]
    assert "extra_validation" in actions
    assert "critic" not in actions


def test_recheck_calls_different_tool(cfg_df):
    cfg, df = cfg_df
    # Force a critic→recheck_physics path via bad B1 case
    row = df.loc[df["no"] == 20]
    if row.empty:
        pytest.skip("row 20 missing")
    case = row.iloc[0].to_dict()
    result = run_graph(case, df, cfg, use_llm=False, allow_critic=True)
    trace = result.get("route_trace") or []
    joined = " ".join(trace)
    # If critic requested physics recheck, must use recheck_physics not a second geomechanics
    if "recheck_physics" in joined:
        assert joined.count(":geomechanics") <= 1 or "geomechanics_skipped" in joined
        assert "recheck_physics" in joined
        assert result.get("recheck_payloads", {}).get("physics_tol") is not None
    else:
        # Still verify the recheck tool exists and differs
        from rockburst.agents.rechecks import physics_recheck_perturbed_tolerance

        payload = physics_recheck_perturbed_tolerance(case, cfg)
        assert payload["tool"] == "physics_recheck_perturbed_tolerance"
        assert "borderline_relations" in payload


def test_non_endorsing_critic_keeps_state_until_recheck():
    """C1: Critic must remain for recheck preconditions; clear only after tool runs."""
    cfg = load_config()
    state = CaseState(
        case={},
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        pred=None,
        unc=None,
        critic=CriticOutput(
            challenges=["Low physics consistency"],
            request_recheck="physics",
            endorse=False,
        ),
    )
    # Minimal stubs so planner reaches recheck branch
    from rockburst.agents.schemas import (
        PhysicsOutput,
        PredictiveOutput,
        UncertaintyOutput,
    )

    state.phys = PhysicsOutput(physics_score=0.4, issues=["b1 inconsistency"])
    state.pred = PredictiveOutput(
        predicted_grade=1,
        probabilities=[0.05, 0.85, 0.05, 0.05],
        r_model=0.9,
    )
    state.unc = UncertaintyOutput(
        uncertainty_score=0.2,
        r_uncertainty=0.8,
        prediction_set=[1],
        model_disagreement=0.0,
        prediction_stability="high",
        stability_score=0.9,
        entropy=0.5,
    )
    state.extra_validation_done = True  # M2 identity gate already satisfied
    action = rule_based_next_action(state, cfg, allow_critic=True)
    assert action == "recheck_physics"
    assert state.critic is not None
    assert state.critic.endorse is False
    assert action_preconditions_met(action, state)


def test_non_endorsing_critic_executes_recheck_with_payload(cfg_df):
    """C1: non-endorsing Critic → executed recheck_* and payload in recheck_payloads."""
    cfg, df = cfg_df
    from rockburst.agents.schemas import (
        PhysicsOutput,
        PredictiveOutput,
        UncertaintyOutput,
    )
    from rockburst.config import resolve_path
    import joblib

    mp = resolve_path(cfg, "models") / "best_baseline.joblib"
    if not mp.exists():
        pytest.skip("model artifact missing")

    case = df.iloc[0].to_dict()
    state = CaseState(case=case, max_loops=8)
    state.data_q = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    state.phys = PhysicsOutput(physics_score=0.4, issues=["b1 inconsistency"])
    state.pred = PredictiveOutput(
        predicted_grade=1,
        probabilities=[0.05, 0.85, 0.05, 0.05],
        r_model=0.9,
    )
    state.unc = UncertaintyOutput(
        uncertainty_score=0.2,
        r_uncertainty=0.8,
        prediction_set=[1],
        model_disagreement=0.0,
        prediction_stability="high",
        stability_score=0.9,
        entropy=0.5,
    )
    state.extra_validation_done = True  # isolate critic→recheck path
    state.critic = CriticOutput(
        challenges=["Derived B1 does not match sigma_c/sigma_t"],
        request_recheck="physics",
        endorse=False,
    )
    state = step(state, df.head(50), cfg, use_llm=False, allow_critic=True)
    assert any(t.endswith(":recheck_physics") for t in state.route_trace)
    assert not any(t.startswith("rejected:recheck_physics") for t in state.route_trace)
    assert state.recheck_payloads.get("physics_tol") is not None
    assert state.critic is None  # cleared after successful recheck


def test_loop_limit_forces_abstain_on_missing_evidence(cfg_df):
    """C2/RC5: absolute max_loops safety ceiling with missing evidence → forced abstain."""
    cfg, df = cfg_df
    # Safety ceiling only — not the production non-convergence path (critic cycles).
    cfg = {**cfg, "routing": {**cfg.get("routing", {}), "max_loops": 2}}
    case = df.iloc[0].to_dict()
    result = run_graph(case, df.head(50), cfg, use_llm=False, allow_critic=False)
    assert result.get("abstained") is True
    assert result.get("recommended_action") == "abstain"
    assert "forced_finalize" in (result.get("route_trace") or [])
    assert "Forced stop" in (result.get("decision") or "")


def test_loop_limit_forces_abstain_on_unresolved_critic():
    """RC5: Critic→recheck cycle budget exhausted with unresolved Critic → forced_abstain."""
    cfg = load_config()
    from rockburst.agents.schemas import (
        PhysicsOutput,
        PredictiveOutput,
        UncertaintyOutput,
    )

    state = CaseState(
        case={},
        loop=6,
        max_loops=24,
        max_critic_recheck_cycles=3,
        critic_recheck_cycles=3,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(physics_score=0.5),
        critic=CriticOutput(
            challenges=["unresolved"],
            request_recheck="physics",
            endorse=False,
        ),
        pred=PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.05, 0.85, 0.05, 0.05],
            r_model=0.9,
        ),
        unc=UncertaintyOutput(
            uncertainty_score=0.2,
            r_uncertainty=0.8,
            prediction_set=[1],
            model_disagreement=0.0,
            prediction_stability="high",
            stability_score=0.9,
            entropy=0.5,
        ),
        extra_validation_done=True,
    )
    assert rule_based_next_action(state, cfg) == "forced_abstain"


def test_guard_rejects_invalid_action():
    state = CaseState(case={})
    assert not action_preconditions_met("critic", state)
    assert not action_preconditions_met("finalize", state)
    assert not action_preconditions_met("uncertainty", state)
    assert action_preconditions_met("data_quality", state)


def test_unresolved_objection_downgrades_decision(cfg_df):
    cfg, _ = cfg_df
    critic = CriticOutput(
        challenges=["test"],
        request_recheck="none",
        endorse=False,
    )
    # Build minimal fuse with high R so base decision would be accept-ish
    from rockburst.agents.schemas import (
        DataQualityOutput,
        PhysicsOutput,
        PredictiveOutput,
        UncertaintyOutput,
    )

    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.95)
    pred = PredictiveOutput(
        predicted_grade=1,
        probabilities=[0.05, 0.85, 0.05, 0.05],
        r_model=0.85,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.2,
        r_uncertainty=0.8,
        prediction_set=[1],
        model_disagreement=0.0,
        prediction_stability="high",
        stability_score=0.9,
        entropy=0.5,
    )
    out = fuse_assessment(
        dq, phys, pred, unc, cfg, ["rules:finalize"], critic=critic
    )
    # Without critic, decide() may vary; with endorse=False must not be plain accept
    # unless already abstain. Downgrade from accept → caution, etc.
    if out["recommended_action"] != "abstain":
        assert out.get("downgrade_reason") or out["recommended_action"] in {
            "accept_with_caution",
            "engineering_verification",
        }


def test_cheap_precheck_pure():
    cfg = load_config()
    case = {
        "sigma_theta": 50.0,
        "sigma_c": 100.0,
        "sigma_t": 10.0,
        "scf": 0.5,
        "b1": 10.0,
        "b2": (100 - 10) / (100 + 10),
        "wet": 10.0,
        "d": 500.0,
    }
    out = cheap_physics_precheck(case, cfg)
    assert out["clean"] is True
    case_bad = dict(case)
    case_bad["d"] = None
    assert cheap_physics_precheck(case_bad, cfg)["clean"] is False
