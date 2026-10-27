"""Slice 7: engineering decision text, consistency, intensity names."""

from __future__ import annotations

from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
import numpy as np
import pytest

from rockburst.agents.tools import (
    enforce_decision_consistency,
    format_engineering_decision,
    fuse_assessment,
    intensity_name,
)
from rockburst.fusion.stacker import ReliabilityFusion


def test_intensity_names():
    assert intensity_name(0) == "none"
    assert intensity_name(1) == "light"
    assert intensity_name(2) == "moderate"
    assert intensity_name(3) == "strong"
    assert intensity_name(None) == "unknown"


def test_format_engineering_decision_includes_intensity_set_confidence():
    text = format_engineering_decision(
        predicted_grade=2,
        prediction_set=[1, 2, 3],
        r_final=0.72,
        action="accept_with_caution",
        disagreement={
            "has_disagreement": True,
            "ml_vs_criteria": True,
            "ml_intensity": "moderate",
            "criteria_intensity": "light",
            "critic_endorse": False,
            "critic_rules_fired": ["ml_vs_criteria"],
        },
    )
    assert "moderate" in text
    assert "light" in text
    assert "R_final=0.720" in text
    assert "accept_with_caution" in text
    assert "Inter-agent disagreement" in text


def test_consistency_keeps_r_final_on_abstain():
    # RC3: no 0.49 clamp — report true R_final beside override reason.
    action, r, note = enforce_decision_consistency(action="abstain", r_final=0.99)
    assert action == "abstain"
    assert r == pytest.approx(0.99)
    assert note is not None and "unchanged" in note


def test_fuse_assessment_emits_m15_fields(tmp_path, monkeypatch):
    # Fit a trivial fusion so decide() works
    fusion = ReliabilityFusion()
    rng = np.random.default_rng(0)
    X = rng.random((40, fusion.n_features if hasattr(fusion, "n_features") else 5))
    # Use public API
    from rockburst.fusion.stacker import FEATURE_NAMES

    R = rng.random((60, len(FEATURE_NAMES)))
    y = (R[:, 0] > 0.5).astype(int)
    fusion.fit(R, y)
    path = tmp_path / "fusion.joblib"
    fusion.save(path)

    cfg = {
        "_root": str(tmp_path),
        "paths": {"models": ".", "baselines": "."},
        "routing": {},
        "_fold_fusion": fusion,
    }
    dq = DataQualityOutput(data_reliability=0.8, ood_flag=False)
    phys = PhysicsOutput(
        physics_score=0.7,
        empirical_checks={
            "criteria_as_given": {
                "consensus_grade": 1,
                "grades": {"russenes": 1},
                "agreement": 1.0,
            }
        },
    )
    pred = PredictiveOutput(
        predicted_grade=3,
        probabilities=[0.05, 0.05, 0.1, 0.8],
        r_model=0.8,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.2,
        r_uncertainty=0.8,
        prediction_set=[2, 3],
        model_disagreement=0.1,
        prediction_stability="high",
        stability_score=0.9,
        entropy=0.5,
    )
    critic = CriticOutput(
        endorse=False,
        rules_fired=["ml_vs_criteria"],
        escalate_abstain=True,
        challenges=["ML vs criteria"],
    )
    out = fuse_assessment(dq, phys, pred, unc, cfg, ["rules:finalize"], critic=critic)
    assert out["predicted_intensity"] == "strong"
    assert "moderate" in out["prediction_set_intensities"] or "strong" in out[
        "prediction_set_intensities"
    ]
    assert out["delivered_planner"] == "rules_fsm"
    assert out["inter_agent_disagreement"]["ml_vs_criteria"] is True
    assert out["agent_disagreement"] in {"Moderate", "High"}
    assert "Predicted intensity" in out["decision"]
    # Severe critic → abstain; R_final may be high → consistency may rewrite
    assert out["recommended_action"] in {
        "abstain",
        "engineering_verification",
        "accept_with_caution",
        "accept",
    }
