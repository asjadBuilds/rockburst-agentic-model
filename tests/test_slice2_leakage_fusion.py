"""Slice 2: fold-local ensembles, learned thresholds, calibrator bounds."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rockburst.agents.schemas import (
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.agents.tools import fuse_assessment, run_uncertainty
from rockburst.config import load_config
from rockburst.fusion.stacker import ReliabilityFusion


def test_uncertainty_with_pipeline_does_not_load_disk_ensemble():
    cfg = load_config()
    pred = PredictiveOutput(
        predicted_grade=1,
        probabilities=[0.1, 0.7, 0.1, 0.1],
        r_model=0.7,
    )
    pipe = MagicMock()
    pipe.predict_proba = MagicMock(return_value=np.array([[0.1, 0.7, 0.1, 0.1]]))

    with patch("rockburst.agents.tools.joblib.load") as load_mock:
        unc = run_uncertainty(
            {"sigma_c": 100.0},
            pred,
            cfg,
            pipeline=pipe,
            ensemble=None,
            conformal=None,
        )
        load_mock.assert_not_called()
    assert unc.model_disagreement == 0.0


def test_decide_ignores_hand_routing_thresholds_by_default():
    f = ReliabilityFusion()
    f.is_fit = True
    f.coef_ = np.zeros(7)
    f.coef_[2] = 1.0
    f.set_decision_thresholds(
        {
            "abstain_r_final": 0.2,
            "engineering_verification": 0.3,
            "accept_with_caution": 0.4,
            "accept_r_final": 0.5,
        }
    )
    # Hand-set routing would push abstain very high if applied
    routing = {
        "abstain_r_final": 0.95,
        "accept_r_final": 0.99,
        "engineering_verification": 0.96,
        "accept_with_caution": 0.97,
    }
    assert f.decide(0.6, routing) == "accept"
    routing["override_decision_thresholds"] = True
    assert f.decide(0.6, routing) == "abstain"


def test_fuse_assessment_uses_fold_fusion_not_disk():
    cfg = load_config()
    fold = ReliabilityFusion()
    fold.coef_ = np.zeros(7)
    fold.coef_[2] = 2.0
    fold.intercept_ = -0.5
    fold.is_fit = True
    fold.set_decision_thresholds(
        {
            "abstain_r_final": 0.1,
            "engineering_verification": 0.2,
            "accept_with_caution": 0.3,
            "accept_r_final": 0.4,
        }
    )
    cfg = {**cfg, "_fold_fusion": fold}
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.9)
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
    with patch("rockburst.agents.tools.ReliabilityFusion.load") as load_mock:
        out = fuse_assessment(dq, phys, pred, unc, cfg, ["rules:finalize"])
        load_mock.assert_not_called()
    assert out["abstained"] is False or out["recommended_action"] != "fusion_unfitted"
    assert "final_system_confidence" in out
    assert 0.0 < out["final_system_confidence"] < 1.0


def test_calibrator_bounds_away_from_exact_01():
    rng = np.random.default_rng(0)
    r_raw = rng.uniform(0.2, 0.8, size=80)
    y = (r_raw > 0.5).astype(int)
    f = ReliabilityFusion()
    f.conf_eps = 1e-3
    f.fit_calibrator(r_raw, y)
    assert f.calibrator is not None
    assert f.calibration_n == 80
    assert f.calibration_method in {"platt", "beta", "smoothed_isotonic", "isotonic"}
    # Build a fitted identity-ish stacker to score
    f.coef_ = np.array([0, 0, 1.0, 0, 0, 0, 0], dtype=float)
    f.intercept_ = 0.0
    f.is_fit = True
    vals = [f.r_final(np.array([0, 0, x, 0, 0, 0, 0], dtype=float)) for x in (0.0, 0.5, 1.0)]
    assert all(1e-3 <= v <= 1.0 - 1e-3 for v in vals)


def test_platt_when_few_unique_raw_scores():
    r_raw = np.array([0.5] * 20 + [0.6] * 20)
    y = np.array([0] * 20 + [1] * 20)
    f = ReliabilityFusion()
    f.fit_calibrator(r_raw, y)
    assert f.calibration_method == "platt"
