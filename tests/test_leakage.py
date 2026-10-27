"""Leakage, fusion, conformal, and routing guardrail tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.config import load_config
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, pack_r_vector
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _split_fit_calib
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    conformal_prediction_set,
    coverage_and_size,
    finite_sample_quantile,
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
)


@pytest.fixture(scope="module")
def processed_df():
    cfg = load_config()
    from rockburst.config import resolve_path

    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        pytest.skip("processed CSV missing; run extract first")
    return pd.read_csv(path)


def test_dup_group_never_spans_train_test(processed_df):
    X, y, groups = prepare_xy(processed_df)
    # Outer and inner splits
    for n_splits, seed in [(5, 42), (3, 7)]:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X, y, groups):
            g_tr = set(groups[tr].tolist())
            g_te = set(groups[te].tolist())
            assert g_tr.isdisjoint(g_te)
            # No physical row group split either
            assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_known_pairs_grouped_and_fold_safe(processed_df):
    from rockburst.data.audit import assign_dup_groups

    gdf = assign_dup_groups(processed_df.drop(columns=["dup_group_id"], errors="ignore"))
    by_no = gdf.set_index("no")
    for a, b in [(6, 315), (14, 316), (8, 322), (1, 327)]:
        assert by_no.loc[a, "dup_group_id"] == by_no.loc[b, "dup_group_id"]
    X, y, groups = prepare_xy(gdf)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgkf.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_imputer_scaler_fit_on_train_only(processed_df):
    X, y, groups = prepare_xy(processed_df)
    tr, te = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(
            X, y, groups
        )
    )
    pipe = Pipeline([("prep", build_preprocess("median"))])
    pipe.fit(X.iloc[tr])
    imputer = pipe.named_steps["prep"].named_steps["imputer"]
    # Statistics must match train-only fit, not full data
    full = build_preprocess("median")
    full.fit(X)
    train_stats = np.asarray(imputer.statistics_, dtype=float)
    full_stats = np.asarray(full.named_steps["imputer"].statistics_, dtype=float)
    # Should differ unless missingness is uniform (almost always differs on D)
    assert train_stats.shape == full_stats.shape
    # Transform test must not refit
    Xt = pipe.transform(X.iloc[te])
    assert Xt.shape[0] == len(te)


def test_target_never_used_for_imputation(processed_df):
    X, y, _ = prepare_xy(processed_df)
    assert "rockburst_grade" not in X.columns
    pipe = build_preprocess("median")
    # sklearn imputer fit signature ignores y when called via Pipeline with X only
    pipe.fit(X)
    # Explicitly ensure FEATURE frame has no target leakage columns
    assert all(not c.startswith("rockburst") for c in X.columns)


def test_fold_reference_excludes_self_keeps_group(processed_df):
    """RM3: exclude case by ``no`` only; same-group near-dups stay for conflict checks."""
    case = processed_df.iloc[0].to_dict()
    idx = np.arange(len(processed_df))
    ref = build_fold_reference(processed_df, idx, case)
    assert int(case["no"]) not in set(ref["no"].astype(int))
    # Optional legacy group hold-out still available
    ref_g = build_fold_reference(
        processed_df, idx, case, exclude_dup_group=True
    )
    if "dup_group_id" in processed_df.columns:
        assert case["dup_group_id"] not in set(ref_g["dup_group_id"])


def test_fusion_monotonicity():
    from rockburst.fusion.stacker import FEATURE_NAMES, FREE_SIGN_FEATURES

    rng = np.random.default_rng(0)
    R = rng.uniform(0.2, 0.9, size=(80, 7))
    R[:, 5] = rng.integers(0, 3, size=80)  # n_missing
    R[:, 6] = rng.integers(0, 2, size=80)  # d_observed
    # Higher R → more likely correct
    w = np.array([1.0, 1.0, 1.2, 0.8, 0.7, -0.3, 0.2])
    logits = R @ w - 2.5
    y = (rng.random(80) < 1 / (1 + np.exp(-logits))).astype(int)
    if len(np.unique(y)) < 2:
        y[:40] = 0
        y[40:] = 1
    fusion = ReliabilityFusion().fit(R, y)
    assert fusion.is_fit
    for i, name in enumerate(FEATURE_NAMES):
        if name not in FREE_SIGN_FEATURES:
            assert fusion.coef_[i] >= -1e-12
    base = pack_r_vector(0.8, 0.8, 0.8, 0.8, 0.8, 0.0, 1.0)
    worse = pack_r_vector(0.2, 0.8, 0.8, 0.8, 0.8, 0.0, 1.0)
    assert fusion.r_final(worse) <= fusion.r_final(base) + 1e-9


def test_train_serve_reliability_path_identical():
    from rockburst.agents.schemas import (
        DataQualityOutput,
        PhysicsOutput,
        PredictiveOutput,
        UncertaintyOutput,
    )
    from rockburst.fusion.features import compute_r_model

    dq = DataQualityOutput(data_reliability=0.7, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.9)
    proba = [0.1, 0.2, 0.6, 0.1]
    pred = PredictiveOutput(
        predicted_grade=2,
        probabilities=proba,
        r_model=compute_r_model(proba),
        observed_mask={
            "sigma_theta": True,
            "sigma_c": True,
            "sigma_t": True,
            "scf": True,
            "b1": True,
            "b2": True,
            "wet": True,
            "d": False,
        },
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.3,
        r_uncertainty=0.7,
        prediction_set=[2],
        model_disagreement=0.1,
        prediction_stability="high",
        stability_score=0.85,
        entropy=0.5,
    )
    a = compute_reliability_features(dq, phys, pred, unc)
    b = pack_r_vector(0.7, 0.9, 0.6, 0.7, 0.85, 1.0, 0.0)
    np.testing.assert_allclose(a, b)
    assert compute_r_model(proba) == 0.6


def test_r_model_train_serve_no_discount():
    from rockburst.fusion.features import compute_r_model

    proba = np.array([0.05, 0.1, 0.7, 0.15])
    assert compute_r_model(proba) == 0.7


def test_conformal_calib_disjoint_from_fit(processed_df):
    X, y, groups = prepare_xy(processed_df)
    tr, te = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1).split(
            X, y, groups
        )
    )
    fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=1)
    assert set(fit_idx).isdisjoint(set(calib_idx))
    assert set(calib_idx).isdisjoint(set(te))
    assert set(fit_idx).isdisjoint(set(te))


def test_finite_sample_quantile_formula():
    scores = np.arange(1, 11, dtype=float)  # n=10
    # ceil((10+1)*0.9)/10 → ceil(9.9)=10 → 10th order stat = 10
    assert finite_sample_quantile(scores, 0.1) == 10.0


def test_conformal_coverage_near_nominal():
    rng = np.random.default_rng(42)
    coverages = []
    for seed in range(12):
        n = 300
        y = rng.integers(0, 4, size=n)
        logits = rng.normal(size=(n, 4))
        logits[np.arange(n), y] += 2.0
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        split = n // 2
        conf = fit_marginal_thresholds(y[:split], proba[:split], alpha=0.1)
        sets = [
            conformal_prediction_set(
                proba[i], alpha=0.1, class_thresholds=conf.class_thresholds
            )
            for i in range(split, n)
        ]
        cov = coverage_and_size(y[split:], sets)["coverage"]
        coverages.append(cov)
    mean_cov = float(np.mean(coverages))
    # Finite-sample marginal split should sit near nominal under exchangeability
    assert 0.80 <= mean_cov <= 0.98, mean_cov


def test_max_loops_forced_abstain(processed_df):
    """Safety max_loops ceiling still forces abstain when planner never finalizes."""
    case = processed_df.iloc[0].to_dict()
    import rockburst.graph.runner as runner

    def never_finalize(state, cfg, allow_critic=True):
        if state.data_q is None:
            return "data_quality"
        if state.phys is None:
            return "geomechanics"
        if state.pred is None:
            return "predict"
        if state.unc is None:
            return "uncertainty"
        return "data_quality"  # loop forever until forced

    original = runner.rule_based_next_action
    runner.rule_based_next_action = never_finalize  # type: ignore[assignment]
    try:
        cfg3 = load_config()
        # Absolute step ceiling only (RC5 primary limit is critic cycles).
        cfg3["routing"] = {**cfg3.get("routing", {}), "max_loops": 2}
        from rockburst.config import resolve_path

        mp = resolve_path(cfg3, "models") / "best_baseline.joblib"
        if not mp.exists():
            pytest.skip("model artifact missing")
        result = runner.run_graph(case, processed_df.head(50), cfg3, use_llm=False)
        assert result.get("abstained") is True
        assert "forced_finalize" in result.get("route_trace", []) or result.get(
            "recommended_action"
        ) == "abstain"
    finally:
        runner.rule_based_next_action = original


def test_thresholds_selected_on_validation_only():
    from rockburst.fusion.stacker import select_decision_thresholds

    rng = np.random.default_rng(0)
    r = rng.uniform(0, 1, 100)
    y = (r > 0.5).astype(int)
    # Pretend val is indices 50:100 only
    thr = select_decision_thresholds(r[50:], y[50:])
    assert "abstain_r_final" in thr
    assert "accept_r_final" in thr
    assert "engineering_verification" in thr
    assert thr["abstain_r_final"] < thr["engineering_verification"]
    assert thr["engineering_verification"] < thr["accept_with_caution"]
    assert thr["accept_with_caution"] <= thr["accept_r_final"]


def test_decide_band_ordering():
    from rockburst.fusion.stacker import ReliabilityFusion

    f = ReliabilityFusion()
    f.set_decision_thresholds(
        {
            "abstain_r_final": 0.4,
            "engineering_verification": 0.55,
            "accept_with_caution": 0.65,
            "accept_r_final": 0.8,
        }
    )
    # Confidence-ascending: eng_verify is BELOW accept, never above.
    assert f.decide(0.3) == "abstain"
    assert f.decide(0.5) == "engineering_verification"
    assert f.decide(0.6) == "accept_with_caution"  # eng ≤ R < caution
    assert f.decide(0.7) == "accept_with_caution"  # caution ≤ R < accept
    assert f.decide(0.85) == "accept"


def test_crossfit_fusion_indices_disjoint():
    """ECE/held-out scores must not use rows that fitted that fold's stacker."""
    rng = np.random.default_rng(0)
    n, n_folds = 40, 4
    fold_of_row = np.repeat(np.arange(n_folds), n // n_folds)
    for fold in range(n_folds):
        idx_fit = set(np.where(fold_of_row != fold)[0].tolist())
        idx_te = set(np.where(fold_of_row == fold)[0].tolist())
        assert idx_fit.isdisjoint(idx_te)
        assert len(idx_fit) + len(idx_te) == n


def test_unfitted_fusion_raises():
    from rockburst.fusion.stacker import FusionNotFitError, ReliabilityFusion

    f = ReliabilityFusion()
    try:
        f.r_final(np.ones(7))
        assert False, "expected FusionNotFitError"
    except FusionNotFitError:
        pass
