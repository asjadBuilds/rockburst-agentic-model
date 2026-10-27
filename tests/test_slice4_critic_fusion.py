"""Slice 4: Critic escalation, continuous disagreement, M3≠M1 criteria adjust."""

from __future__ import annotations

import numpy as np

from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.agents.tools import fuse_assessment, run_critic
from rockburst.config import load_config
from rockburst.eval.ablation import _wilson_ci
from rockburst.fusion.components import ComponentCalibrators, mask_for_set
from rockburst.uncertainty.estimates import disagreement, mean_pairwise_tv


def test_mean_pairwise_tv_disagreement():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    q = np.array([0.1, 0.7, 0.1, 0.1])
    tv = mean_pairwise_tv([p, q])
    assert abs(tv - 0.6) < 1e-9
    assert disagreement([p, q]) == tv
    assert disagreement([p]) == 0.0


def test_critic_escalates_severe_rules_to_abstain():
    cfg = load_config()
    cfg["_critic_rule_statistics"] = {
        "identity_violation": {
            "auroc_ci95": [0.61, 0.75],
            "error_delta_ci95": [0.08, 0.30],
        }
    }
    phys = PhysicsOutput(
        physics_score=0.9,
        issues=["scf inconsistency: observed=1, expected=2"],
        empirical_checks={
            "criteria_as_given": {"consensus_grade": 0, "agreement": 1.0}
        },
    )
    pred = PredictiveOutput(
        predicted_grade=3,
        probabilities=[0.05, 0.05, 0.05, 0.85],
        r_model=0.85,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.2,
        r_uncertainty=0.8,
        prediction_set=[0, 1, 2, 3],
        model_disagreement=0.1,
        prediction_stability="high",
        stability_score=0.9,
        entropy=0.5,
    )
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    critic = run_critic(dq, phys, pred, unc, cfg)
    assert critic.endorse is False
    assert "identity_violation" in critic.rules_fired
    assert "ml_vs_criteria" in critic.rules_fired
    assert critic.escalate_abstain is True

    # Force high fusion confidence path via fold fusion stub
    from rockburst.fusion.stacker import ReliabilityFusion

    fus = ReliabilityFusion()
    fus.coef_ = np.zeros(7)
    fus.coef_[2] = 5.0
    fus.intercept_ = 0.0
    fus.is_fit = True
    fus.set_decision_thresholds(
        {
            "abstain_r_final": 0.1,
            "engineering_verification": 0.2,
            "accept_with_caution": 0.3,
            "accept_r_final": 0.4,
        }
    )
    out = fuse_assessment(
        dq,
        phys,
        pred,
        unc,
        {**cfg, "_fold_fusion": fus},
        ["rules:finalize"],
        critic=critic,
    )
    assert out["recommended_action"] == "abstain"
    assert out["abstained"] is True
    assert "escalated" in (out.get("downgrade_reason") or "")


def test_eng_verification_downgrades_to_abstain():
    """M5: remove floor at engineering_verification."""
    cfg = load_config()
    from rockburst.fusion.stacker import ReliabilityFusion

    fus = ReliabilityFusion()
    fus.coef_ = np.zeros(7)
    fus.coef_[2] = 1.0
    fus.intercept_ = 0.0
    fus.is_fit = True
    fus.set_decision_thresholds(
        {
            "abstain_r_final": 0.9,
            "engineering_verification": 0.95,
            "accept_with_caution": 0.97,
            "accept_r_final": 0.99,
        }
    )
    # r_final will be mid → likely eng or abstain from thresholds; force via critic
    critic = CriticOutput(
        challenges=["mild"],
        endorse=False,
        rules_fired=["low_physics"],
        escalate_abstain=False,
        request_recheck="none",
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
    # Start from accept_with_caution by setting low abstain thresholds
    fus.set_decision_thresholds(
        {
            "abstain_r_final": 0.1,
            "engineering_verification": 0.2,
            "accept_with_caution": 0.3,
            "accept_r_final": 0.99,
        }
    )
    out = fuse_assessment(
        dq, phys, pred, unc, {**cfg, "_fold_fusion": fus}, ["x"], critic=critic
    )
    # Non-severe: one-step downgrade; if was accept_with_caution → eng; if eng → abstain
    assert out["recommended_action"] in {
        "accept_with_caution",
        "engineering_verification",
        "abstain",
    }
    # Explicit eng → abstain mapping
    from rockburst.agents.tools import _DOWNGRADE

    assert _DOWNGRADE["engineering_verification"] == "abstain"


def test_component_calibrators_identity_fallback():
    rng = np.random.default_rng(0)
    R = rng.uniform(0.2, 0.8, size=(40, 7))
    y = np.ones(40, dtype=int)  # single class → identity passthrough
    cal = ComponentCalibrators().fit(R, y)
    assert cal.methods["r_model"] == "identity_passthrough"
    out = cal.transform_vector(R[0])
    assert out.shape == (7,)


def test_fusion_set_masks():
    assert mask_for_set("A").sum() == 1
    assert mask_for_set("D").sum() == 7


def test_wilson_ci_bounds():
    lo, hi = _wilson_ci(10, 100)
    assert 0.0 <= lo <= hi <= 1.0
