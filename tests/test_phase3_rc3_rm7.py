"""Phase 3: RC3 calibration/bands/clamp + RM7 set-compare / component AUROC hooks."""

from __future__ import annotations

import numpy as np
import pytest

from rockburst.agents.tools import enforce_decision_consistency, fuse_assessment
from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.eval.metrics import ece_with_ci, reliability_diagram_bins
from rockburst.ml.calibration import (
    apply_temperature,
    temperature_calibration_report,
)
from rockburst.ml.pipelines import prepare_xy_sensitivity_variants
from rockburst.uncertainty.conformal import (
    fit_class_conditional_alpha,
    fit_mondrian_thresholds,
)
from rockburst.fusion.set_compare import (
    compare_fusion_sets,
    component_aurocs,
    production_mask_from_comparison,
    production_fusion_from_oof,
)
from rockburst.fusion.stacker import (
    DECISION_BAND_UTILITIES,
    DECISION_BAND_UTILITIES_JUSTIFICATION,
    FEATURE_NAMES,
    ReliabilityFusion,
    bands_are_nonempty,
    decision_band_occupancy,
    decision_utility_sensitivity,
    ensure_nonempty_bands,
    select_decision_thresholds,
)


def test_rc3_no_049_clamp_keeps_r_final():
    action, r, note = enforce_decision_consistency(action="abstain", r_final=0.99)
    assert action == "abstain"
    assert r == pytest.approx(0.99)
    assert note is not None
    assert "unchanged" in note
    assert "0.49" not in note


def test_rc3_fuse_emits_override_reason_without_clamping():
    fusion = ReliabilityFusion()
    rng = np.random.default_rng(0)
    R = rng.random((80, len(FEATURE_NAMES)))
    y = (R[:, 2] > 0.5).astype(int)
    fusion.fit(R, y)
    fusion.fit_calibrator(np.array([fusion._raw_score(r) for r in R]), y)
    fusion.set_decision_thresholds(
        ensure_nonempty_bands(
            {
                "abstain_r_final": 0.2,
                "engineering_verification": 0.3,
                "accept_with_caution": 0.4,
                "accept_r_final": 0.5,
            }
        )
    )
    cfg = {
        "_root": ".",
        "paths": {"models": ".", "baselines": "."},
        "routing": {},
        "_fold_fusion": fusion,
    }
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.9)
    pred = PredictiveOutput(
        predicted_grade=3,
        probabilities=[0.05, 0.05, 0.1, 0.8],
        r_model=0.95,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.1,
        r_uncertainty=0.9,
        prediction_set=[3],
        model_disagreement=0.0,
        prediction_stability="high",
        stability_score=0.95,
        entropy=0.2,
    )
    critic = CriticOutput(
        endorse=False,
        rules_fired=["ood_physics"],
        escalate_abstain=True,
        challenges=["OOD+physics"],
    )
    out = fuse_assessment(dq, phys, pred, unc, cfg, ["rules:finalize"], critic=critic)
    assert out["recommended_action"] == "abstain"
    assert out["final_system_confidence"] > 0.5
    # A reason is required only when Critic overrides the learned decision.
    learned = fusion.decide(out["r_final"])
    if learned != "abstain":
        assert (
            out.get("downgrade_reason")
            or out.get("decision_override_reason")
            or out.get("consistency_note")
        )
    assert out["r_final"] == out["final_system_confidence"]


def test_rc3_bands_nonempty_even_when_collapsed():
    collapsed = {
        "abstain_r_final": 0.60,
        "engineering_verification": 0.60,
        "accept_with_caution": 0.60,
        "accept_r_final": 0.60,
    }
    fixed = ensure_nonempty_bands(collapsed, min_gap=0.05)
    assert bands_are_nonempty(fixed, min_gap=0.049)
    assert fixed["accept_r_final"] > fixed["accept_with_caution"]
    assert fixed["accept_with_caution"] > fixed["engineering_verification"]


def test_rc3_select_thresholds_never_empty_caution_band():
    rng = np.random.default_rng(1)
    r = rng.uniform(0.2, 0.9, size=120)
    y = (r > 0.55).astype(int)
    thr = select_decision_thresholds(r, y, min_gap=0.05)
    assert bands_are_nonempty(thr, min_gap=0.049)
    assert thr["selection_mode"] == "engineering_cost_utility"
    assert "utilities_justification" in thr
    assert DECISION_BAND_UTILITIES["accept_on_error"] < 0
    assert "engineering" in DECISION_BAND_UTILITIES_JUSTIFICATION.lower() or "cost" in (
        DECISION_BAND_UTILITIES_JUSTIFICATION.lower()
    )


def test_rc3_target_accept_rate_mode():
    rng = np.random.default_rng(2)
    r = rng.uniform(0.1, 0.95, size=100)
    y = (r > 0.5).astype(int)
    thr = select_decision_thresholds(r, y, target_accept_rate=0.3, min_gap=0.05)
    assert thr["selection_mode"] == "target_accept_rate"
    assert bands_are_nonempty(thr, min_gap=0.049)


def test_rc3_calibrator_methods_and_bounds():
    rng = np.random.default_rng(3)
    r_raw = rng.uniform(0.15, 0.85, size=200)
    y = (r_raw > 0.5).astype(int)
    for method in ("platt", "beta", "smoothed_isotonic"):
        f = ReliabilityFusion()
        f.conf_eps = 1e-3
        f.fit_calibrator(r_raw, y, method=method)
        assert f.calibrator is not None
        assert f.calibration_n == 200
        assert f.calibration_method == method
        f.coef_ = np.array([0, 0, 1.0, 0, 0, 0, 0], dtype=float)
        f.intercept_ = 0.0
        f.is_fit = True
        lo, hi = f.conf_eps, 1.0 - f.conf_eps
        for x in (0.0, 0.5, 1.0):
            v = f.r_final(np.array([0, 0, x, 0, 0, 0, 0], dtype=float))
            assert lo <= v <= hi
        point, ilo, ihi = f.r_final_interval(np.array([0, 0, 0.7, 0, 0, 0, 0]))
        assert np.isfinite(point)
        assert np.isnan(ilo) and np.isnan(ihi)  # no fabricated constant interval


def test_rc3_production_calib_uses_full_crossfit_n():
    rng = np.random.default_rng(4)
    n, n_folds = 60, 5
    R = rng.random((n, len(FEATURE_NAMES)))
    y_correct = (R[:, 2] > 0.45).astype(int)
    fold_of = np.repeat(np.arange(n_folds), n // n_folds)
    fusion, meta = production_fusion_from_oof(
        R, y_correct, fold_of, conf_eps=1e-3, interval_boot=12
    )
    assert fusion.is_fit
    assert fusion.calibrator is not None
    # Full OOF scored across folds → calibration_n ≈ n (not ~n/5).
    assert fusion.calibration_n >= int(0.8 * n)
    assert meta["calibration_protocol"] == "production_stacker_raw_oof_single_serve_path"
    regenerated = np.array([fusion.r_final(row) for row in R])
    assert np.allclose(regenerated, meta["served_r_final_oof"])
    assert meta["interval_members"] > 0
    point, lo, hi = fusion.r_final_interval(R[0])
    assert np.isfinite([point, lo, hi]).all()
    assert lo <= hi
    assert bands_are_nonempty(fusion.decision_thresholds, min_gap=0.049)


def test_rc3_reliability_ece_hooks():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=80)
    conf = np.clip(y.astype(float) * 0.6 + rng.uniform(0.1, 0.4, size=80), 0.01, 0.99)
    diagram = reliability_diagram_bins(y, conf)
    assert isinstance(diagram, list)
    ece = ece_with_ci(y, conf, n_boot=30, seed=5)
    assert "ece" in ece
    assert ece["ece"] >= 0.0


def test_rm7_component_auroc_and_set_compare():
    rng = np.random.default_rng(6)
    n = 80
    R = rng.random((n, len(FEATURE_NAMES)))
    # Make r_model informative; others near noise.
    y_correct = (R[:, 2] > 0.5).astype(int)
    R[:, 2] = 0.2 + 0.6 * y_correct + rng.normal(0, 0.05, size=n)
    R[:, 2] = np.clip(R[:, 2], 0.01, 0.99)
    aurocs = component_aurocs(R, y_correct)
    assert aurocs["r_model"] is not None
    assert aurocs["r_model"] > 0.7

    y = rng.integers(0, 4, size=n)
    # Ensure each class present for StratifiedGroupKFold
    y[:4] = np.arange(4)
    groups = np.arange(n) // 2
    out = compare_fusion_sets(
        R, y_correct, y, groups, seed=6, n_outer=4, n_boot=20, conf_eps=1e-3
    )
    assert set(out["sets"]) >= {"A", "B", "C", "D"}
    assert "component_auroc" in out
    assert out["component_auroc"]["per_component"]["r_model"]["auroc"] is not None
    assert "delta_vs_best_ci" in out["sets"]["A"]
    assert out["locked_production_set"] == out["chosen_set"]
    assert out["production_set_from_rule"] == out["chosen_set"]
    assert "component_constants_justification" in out


def test_dynamic_set_mask_filters_confidently_inverted_reliability():
    comparison = {
        "chosen_set": "D",
        "selection_rule": "test rule",
        "component_auroc": {
            "per_component": {
                "r_data": {"ci95": [0.1, 0.4]},
                "r_model": {"ci95": [0.7, 0.9]},
            }
        },
    }
    mask, report = production_mask_from_comparison(comparison)
    assert not mask[FEATURE_NAMES.index("r_data")]
    assert mask[FEATURE_NAMES.index("r_model")]
    assert report["selected_set"] == "D"
    assert "r_data" in report["excluded_inverted_reliability_channels"]


def test_decision_occupancy_and_utility_sensitivity_are_reported():
    scores = np.linspace(0.05, 0.95, 40)
    correct = (scores > 0.5).astype(int)
    thresholds = ensure_nonempty_bands({})
    occupancy = decision_band_occupancy(scores, thresholds)
    assert sum(occupancy.values()) == len(scores)
    sensitivity = decision_utility_sensitivity(scores, correct)
    assert set(sensitivity) == {
        "lower_false_accept_cost", "registered", "higher_false_accept_cost"
    }


def test_grade3_conformal_is_distinct_from_mondrian():
    y = np.repeat(np.arange(4), 12)
    p = np.full((len(y), 4), 0.1)
    p[np.arange(len(y)), y] = np.tile(
        np.linspace(0.25, 0.95, 12), 4
    )
    p /= p.sum(axis=1, keepdims=True)
    mondrian = fit_mondrian_thresholds(y, p, alpha=0.2)
    grade3 = fit_class_conditional_alpha(y, p, alpha=0.2, focus_class=3)
    # Non-grade-3 thresholds are pooled in the grade-3 variant.
    assert np.allclose(grade3.class_thresholds[:3], grade3.class_thresholds[0])
    assert not np.allclose(grade3.class_thresholds, mondrian.class_thresholds)


def test_temperature_calibration_and_indicator_sensitivity_hooks():
    import pandas as pd

    p = np.array([[0.98, 0.01, 0.005, 0.005], [0.8, 0.1, 0.05, 0.05]] * 8)
    y = np.array([0, 1] * 8)
    report = temperature_calibration_report(p, y)
    calibrated = apply_temperature(p, report["temperature"])
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert report["nll_after"] <= report["nll_before"] + 1e-9

    cols = ["sigma_theta", "sigma_c", "sigma_t", "scf", "b1", "b2", "wet", "d"]
    df = pd.DataFrame([{**{c: 1.0 for c in cols}, "rockburst_grade": 0}])
    variants = prepare_xy_sensitivity_variants(df)
    x_with = variants["observed_indicators"][0]
    x_without = variants["no_observed_indicators"][0]
    assert x_with.shape[1] == 2 * x_without.shape[1]


def test_no_indicator_pipeline_uses_its_fitted_schema_at_inference():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from rockburst.agents.tools import run_predict
    from rockburst.ml.pipelines import build_preprocess
    from rockburst.physics.relations import FEATURE_COLS

    rng = np.random.default_rng(18)
    X = pd.DataFrame(rng.normal(size=(24, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = np.tile(np.arange(4), 6)
    pipe = Pipeline(
        [("prep", build_preprocess("median")), ("model", LogisticRegression(max_iter=500))]
    ).fit(X, y)
    out = run_predict(X.iloc[0].to_dict(), {}, pipeline=pipe)
    assert len(out.probabilities) == 4
    assert sum(out.probabilities) == pytest.approx(1.0)

    from rockburst.agents.rechecks import monotonic_tendency_recheck

    recheck = monotonic_tendency_recheck(
        X.iloc[0].to_dict(), pipeline=pipe, primary_grade=out.predicted_grade
    )
    assert len(recheck["details"]) == 2
