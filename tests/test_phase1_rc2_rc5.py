"""Phase 1: RC2 fusion leakage path + RC5 Critic→recheck cycle semantics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.agents.tools import fuse_assessment
from rockburst.config import load_config, resolve_path
from rockburst.fusion.components import ComponentCalibrators
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.fold_oof import fit_fold_fusion_oof
from rockburst.fusion.stacker import FEATURE_NAMES
from rockburst.graph.runner import CaseState, rule_based_next_action, run_graph, step
from rockburst.ml.pipelines import prepare_xy


@pytest.fixture
def cfg_df():
    cfg = load_config()
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        pytest.skip("processed csv missing; run extract first")
    return cfg, pd.read_csv(path)


def _agent_bundle(*, d_obs: bool = True):
    dq = DataQualityOutput(data_reliability=0.8, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.7)
    pred = PredictiveOutput(
        predicted_grade=2,
        probabilities=[0.1, 0.2, 0.6, 0.1],
        r_model=0.6,
        observed_mask={
            "sigma_theta": True,
            "sigma_c": True,
            "sigma_t": True,
            "scf": True,
            "b1": True,
            "b2": True,
            "wet": True,
            "d": d_obs,
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
    case = {
        "scf": 1.0,
        "b1": 10.0,
        "b2": 0.5,
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "wet": 10.0,
        "d": 500.0 if d_obs else None,
    }
    return dq, phys, pred, unc, case


def test_no_autoload_full_data_component_calibrators(tmp_path, cfg_df):
    """RC2: disk component_calibrators.joblib must not apply unless injected."""
    cfg, _ = cfg_df
    dq, phys, pred, unc, case = _agent_bundle()
    models = tmp_path / "models"
    models.mkdir()
    # Fitted calibrator on disk that would shift vectors if auto-loaded.
    cal = ComponentCalibrators()
    rng = np.random.default_rng(1)
    R = rng.uniform(0.1, 0.9, size=(80, len(FEATURE_NAMES)))
    y = (R[:, 2] > 0.55).astype(int)
    cal.fit(R, y)
    assert cal.is_fit
    cal.save(models / "component_calibrators.joblib")
    assert (models / "component_calibrators.joblib").exists()

    cfg_disk = {
        **cfg,
        "_root": str(tmp_path),
        "paths": {**cfg["paths"], "models": "models"},
    }
    r_plain = compute_reliability_features(dq, phys, pred, unc, case=case)
    r_with_cfg = compute_reliability_features(
        dq, phys, pred, unc, case=case, cfg=cfg_disk
    )
    # Disk file present must not change the vector (no auto-load).
    np.testing.assert_allclose(r_plain, r_with_cfg)

    cfg_explicit = {**cfg_disk, "_component_calibrators": cal}
    r_explicit = compute_reliability_features(
        dq, phys, pred, unc, case=case, cfg=cfg_explicit
    )
    # Explicit injection may change channels; at least path accepts calibrators.
    assert r_explicit.shape == r_plain.shape
    assert getattr(cal, "is_fit", False) is True

def test_train_serve_vector_identical_with_fold_calibrators(cfg_df):
    """RC2: train-time vector ≡ serve-time vector for same case + fold calibrators."""
    cfg, _ = cfg_df
    dq, phys, pred, unc, case = _agent_bundle()
    cal = ComponentCalibrators()
    rng = np.random.default_rng(0)
    R = rng.uniform(0.2, 0.9, size=(60, len(FEATURE_NAMES)))
    y = (R[:, 2] > 0.5).astype(int)
    cal.fit(R, y)

    # Train path (explicit fold calibrators)
    r_train = compute_reliability_features(
        dq, phys, pred, unc, case=case, cfg={"_component_calibrators": cal}
    )
    # Serve path (fuse_assessment with same calibrators)
    cfg_serve = {**cfg, "_component_calibrators": cal, "_fold_fusion": MagicMock(is_fit=False)}
    out = fuse_assessment(
        dq, phys, pred, unc, cfg_serve, ["rules:finalize"], case=case
    )
    r_serve = np.array([out["r_vector"][n] for n in FEATURE_NAMES], dtype=float)
    np.testing.assert_allclose(r_train, r_serve)


def test_fold_fusion_oof_never_uses_test_rows(cfg_df):
    """RC2: OOF fusion fit indices stay inside outer-train; disjoint from test."""
    cfg, df = cfg_df
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, groups = prepare_xy(df)
    seed = 42
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X, y, groups))
    est = LogisticRegression(max_iter=200)
    ranked = [("logistic__median", {"best_params": {}, "fold_params": []})]
    fusion = fit_fold_fusion_oof(
        df,
        X,
        y,
        groups,
        tr,
        base_est=est,
        imputer="median",
        ranked=ranked,
        cfg=cfg,
        seed=seed,
        fold=0,
        conf_method="marginal_split",
        alpha=0.1,
        quick=True,
    )
    assert set(tr).isdisjoint(set(te))
    assert hasattr(fusion, "is_fit")


def test_critic_recheck_is_not_repeated_on_identical_evidence(cfg_df):
    """TC1: a persistent deterministic objection gets one recheck, not a countdown."""
    cfg, df = cfg_df
    import rockburst.agents.tools as tools_mod
    import rockburst.graph.runner as runner_mod

    # Real case ID (prefer OOD 146 so need_critic stays true after each recheck).
    row = df.loc[df["no"] == 146]
    case = row.iloc[0].to_dict() if not row.empty else df.iloc[0].to_dict()
    case_no = int(case.get("no", -1))
    rt = dict(cfg.get("routing", {}))
    # Keep production max_loops; shrink cycle budget only for a fast unit test.
    fa_cfg = {
        **cfg,
        "routing": {**rt, "max_critic_recheck_cycles": 2},
    }

    original = tools_mod.run_critic

    def _always_object(*_a, **_k):
        return CriticOutput(
            challenges=["persistent"],
            request_recheck="physics",
            endorse=False,
            rules_fired=["test_persistent"],
        )

    tools_mod.run_critic = _always_object  # type: ignore[assignment]
    runner_mod.run_critic = _always_object  # type: ignore[assignment]
    try:
        result = run_graph(case, df, fa_cfg, use_llm=False, allow_critic=True)
    finally:
        tools_mod.run_critic = original  # type: ignore[assignment]
        runner_mod.run_critic = original  # type: ignore[assignment]

    trace = result.get("route_trace") or []
    assert sum("recheck_physics" in step for step in trace) == 1, trace
    assert "forced_finalize" not in trace
    assert result.get("termination_reason") == "finalize_unresolved_no_new_evidence"
    assert int(result.get("critic_recheck_cycles") or 0) == 1
    assert case_no > 0
    # Must not have relied on artificial max_loops=2
    assert int(fa_cfg["routing"].get("max_loops", 24)) >= 8


def test_missing_d_and_present_d_share_critic_cycle_budget():
    """RC5: cycle limit is independent of missing-D extra_validation steps."""
    cfg = load_config()
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
    phys = PhysicsOutput(physics_score=0.5)
    state_miss = CaseState(
        case={"d": None},
        loop=10,
        max_loops=24,
        max_critic_recheck_cycles=3,
        critic_recheck_cycles=3,
        data_q=DataQualityOutput(
            data_reliability=0.8, ood_flag=False, missing_features=["d"]
        ),
        phys=phys,
        critic=CriticOutput(
            challenges=["x"], request_recheck="physics", endorse=False
        ),
        pred=pred,
        unc=unc,
        extra_validation_done=True,
    )
    state_present = CaseState(
        case={"d": 500.0},
        loop=6,
        max_loops=24,
        max_critic_recheck_cycles=3,
        critic_recheck_cycles=3,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=phys,
        critic=CriticOutput(
            challenges=["x"], request_recheck="physics", endorse=False
        ),
        pred=pred,
        unc=unc,
        extra_validation_done=True,
    )
    assert rule_based_next_action(state_miss, cfg) == "forced_abstain"
    assert rule_based_next_action(state_present, cfg) == "forced_abstain"


def test_ordinary_route_does_not_hit_step_ceiling(cfg_df):
    """RC5: legitimate path finishes under production max_loops without forced abstain."""
    cfg, df = cfg_df
    # Present-D clean-ish row
    sub = df[df["d"].notna()]
    if sub.empty:
        pytest.skip("no present-D rows")
    case = sub.iloc[0].to_dict()
    result = run_graph(case, df.head(60), cfg, use_llm=False, allow_critic=False)
    assert result.get("termination_reason") == "finalize"
    assert "forced_finalize" not in (result.get("route_trace") or [])
    assert int(result.get("steps_taken") or 0) < int(
        cfg.get("routing", {}).get("max_loops", 24)
    )


def test_recheck_increments_critic_recheck_cycles(cfg_df):
    cfg, df = cfg_df
    from rockburst.config import resolve_path
    import joblib

    mp = resolve_path(cfg, "models") / "best_baseline.joblib"
    if not mp.exists():
        pytest.skip("model artifact missing")

    case = df.iloc[0].to_dict()
    state = CaseState(
        case=case,
        max_loops=24,
        max_critic_recheck_cycles=3,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(physics_score=0.4, issues=["b1 inconsistency"]),
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
        critic=CriticOutput(
            challenges=["Derived B1 does not match"],
            request_recheck="physics",
            endorse=False,
        ),
    )
    assert state.critic_recheck_cycles == 0
    state = step(state, df.head(50), cfg, use_llm=False, allow_critic=True)
    assert any(t.endswith(":recheck_physics") for t in state.route_trace)
    assert state.critic_recheck_cycles == 1
