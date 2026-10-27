"""Phase 2: RM1–RM6 unit tests (wipe/skip/near-dup/B2/edge-of-set/OOD escalate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rockburst.agents.rechecks import (
    alt_imputer_reprediction,
    monotonic_tendency_recheck,
    validation_objections_from_rechecks,
)
from rockburst.agents.schemas import (
    CriticOutput,
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.agents.tools import (
    critic_rule_stat_hooks,
    disabled_rules_from_stats,
    fuse_assessment,
    run_critic,
    severe_rules_from_stats,
)
from rockburst.config import load_config, resolve_path
from rockburst.graph.runner import (
    CaseState,
    _merge_physics_recheck,
    _record_critic_rules,
    _resolve_rechecked_objections,
    rule_based_next_action,
    step,
)
from rockburst.physics.criteria import (
    BRITTLENESS_B1_THRESHOLDS,
    HOEK_SCF_THRESHOLDS,
    brittleness_b1_grade,
    consensus_from_grades,
    criteria_grades_for_case,
    hoek_grade,
)
from rockburst.physics.precheck import full_physics_result, is_perfect_physics
from rockburst.physics.relations import check_case, kirsch_tangential_stress_range
from rockburst.quality.checks import _SCALE_BOUNDS, score_data_quality
from rockburst.quality.fold_ref import build_fold_reference


@pytest.fixture
def cfg_df():
    cfg = load_config()
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        pytest.skip("processed csv missing; run extract first")
    return cfg, pd.read_csv(path)


def test_rm1_recheck_preserves_criteria_and_can_change_score():
    """RM1: recheck merge keeps criteria_as_given; score can move."""
    first = PhysicsOutput(
        physics_score=0.8,
        exact_checks={"b1": {"ok": True}},
        empirical_checks={
            "criteria_as_given": {
                "consensus_grade": 2,
                "agreement": 1.0,
                "grades": {"russenes": 2},
            }
        },
        issues=[],
    )
    payload = {
        "physics_score_stated": 0.55,
        "physics_score_tight": 0.4,
        "borderline_relations": ["b1"],
        "issues_stated": ["borderline note"],
        "low_physics": True,
    }
    merged = _merge_physics_recheck(first, payload)
    assert "criteria_as_given" in merged.empirical_checks
    assert merged.empirical_checks["criteria_as_given"]["consensus_grade"] == 2
    assert merged.physics_score < first.physics_score
    assert "recheck_borderline" in merged.exact_checks


def test_rm1_recheck_physics_step_does_not_wipe(cfg_df):
    cfg, df = cfg_df
    case = df.iloc[0].to_dict()
    res = full_physics_result(case, cfg)
    state = CaseState(
        case=case,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(**res.to_dict()),
        pred=PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.1, 0.6, 0.2, 0.1],
            r_model=0.6,
        ),
        unc=UncertaintyOutput(
            uncertainty_score=0.3,
            r_uncertainty=0.7,
            prediction_set=[1],
            model_disagreement=0.1,
            prediction_stability="high",
            stability_score=0.9,
            entropy=0.4,
        ),
        critic=CriticOutput(
            challenges=["x"],
            endorse=False,
            request_recheck="physics",
            rules_fired=["low_physics"],
        ),
        extra_validation_done=True,
    )
    before_crit = (state.phys.empirical_checks or {}).get("criteria_as_given")
    assert before_crit is not None
    state = step(state, df, cfg, use_llm=False, allow_critic=True)
    assert any(t.endswith(":recheck_physics") for t in state.route_trace)
    assert (state.phys.empirical_checks or {}).get("criteria_as_given") is not None
    assert state.recheck_effects
    assert state.recheck_effects[-1].get("preserved_criteria") is True


def test_rm4_skip_retains_criteria(cfg_df):
    """RM4: clean path keeps empirical criteria; R_physics matches full tool."""
    cfg, df = cfg_df
    clean = None
    for _, row in df.iterrows():
        case = row.to_dict()
        res = full_physics_result(case, cfg)
        if is_perfect_physics(res):
            clean = case
            full = res
            break
    if clean is None:
        pytest.skip("no clean physics row")
    state = CaseState(
        case=clean,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
    )
    state = step(state, df, cfg, use_llm=False, allow_critic=False)
    # May need to force geomechanics if planner chose data_quality first
    if state.phys is None:
        state = step(state, df, cfg, use_llm=False, allow_critic=False)
    assert state.phys is not None
    assert (state.phys.empirical_checks or {}).get("criteria_as_given")
    assert abs(float(state.phys.physics_score) - float(full.r_physics)) < 1e-9
    if state.physics_skipped:
        assert "geomechanics_skipped" in " ".join(state.route_trace)


def test_tm2_only_independent_criteria_vote():
    assert BRITTLENESS_B1_THRESHOLDS == (10.0, 14.0, 18.0)
    assert HOEK_SCF_THRESHOLDS == (0.42, 0.56, 0.70)
    assert brittleness_b1_grade(15.0) == 2  # 10/14/18 → ≥14 → 2
    assert hoek_grade(0.60) == 2
    row = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "scf": 2.0,
        "b1": 10.0,
        "b2": 0.82,
        "wet": 5.0,
        "d": 500.0,
    }
    crit = criteria_grades_for_case(row)
    assert set(crit["grades"]) == {"russenes", "kidybinski"}
    assert "hoek" in crit["diagnostic_grades"]
    assert "brittleness_b1" in crit["diagnostic_grades"]
    assert "brittleness_b2" not in crit["grades"]
    assert "brittleness_b2" in crit["diagnostic_grades"]
    assert crit["voting_policy"] == "russenes_plus_kidybinski_equal_weight"
    # Rounded median: {1,2} → 1.5 → 2
    cons, agr, rule = consensus_from_grades({"a": 1, "b": 2})
    assert cons == 2
    assert rule == "rounded_median_tiebreak"


def test_rm2_edge_of_wide_set_not_duplicate_of_large_set():
    cfg = load_config()
    phys = PhysicsOutput(physics_score=0.9)
    pred = PredictiveOutput(
        predicted_grade=0,
        probabilities=[0.4, 0.2, 0.2, 0.2],
        r_model=0.4,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.5,
        r_uncertainty=0.5,
        prediction_set=[0, 1, 2],
        model_disagreement=0.1,
        prediction_stability="low",
        stability_score=0.4,
        entropy=1.0,
    )
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    critic = run_critic(dq, phys, pred, unc, cfg)
    assert "large_prediction_set" in critic.rules_fired
    assert "edge_of_wide_set" in critic.rules_fired
    assert "wide_conformal_edge" not in critic.rules_fired
    # Interior prediction on wide set → large set only, not edge
    pred_mid = PredictiveOutput(
        predicted_grade=1,
        probabilities=[0.2, 0.5, 0.2, 0.1],
        r_model=0.5,
    )
    critic2 = run_critic(dq, phys, pred_mid, unc, cfg)
    assert "large_prediction_set" in critic2.rules_fired
    assert "edge_of_wide_set" not in critic2.rules_fired
    assert critic_rule_stat_hooks()["per_rule_fields"]


def test_rm2_row_shift_not_every_large_b1():
    row = {
        "sigma_theta": 54.2,
        "sigma_c": 134.0,
        "sigma_t": 9.1,
        "scf": 0.4,
        "b1": 0.15,
        "b2": 0.87,
        "wet": 7.1,
        "d": 500.0,
    }
    res = check_case(row)
    assert any("large_b1_identity_gap" in i for i in res.issues)
    assert not any("suspected_row_shift" in i for i in res.issues)


def test_rm3_near_dup_fold_train_no_own_label(cfg_df):
    cfg, df = cfg_df
    # Build synthetic conflicting train near-dups sharing a group
    base = df.iloc[0].to_dict()
    ref = df.iloc[:30].copy()
    for col in ("sigma_theta", "sigma_c", "sigma_t", "wet", "d"):
        ref.loc[ref.index[0], col] = base[col]
        ref.loc[ref.index[1], col] = float(base[col]) * 1.001
    ref.loc[ref.index[0], "rockburst_grade"] = 0
    ref.loc[ref.index[1], "rockburst_grade"] = 3
    gid = 9001
    ref.loc[ref.index[0], "dup_group_id"] = gid
    ref.loc[ref.index[1], "dup_group_id"] = gid
    case = {**base, "no": 99999, "dup_group_id": gid, "rockburst_grade": 1}
    # fold ref keeps same group (RM3)
    fold_ref = build_fold_reference(ref, np.arange(len(ref)), case)
    assert gid in set(fold_ref["dup_group_id"])
    out = score_data_quality(case, fold_ref, near_dup_rel_tol=0.05)
    assert out["conflicting_near_dup_labels"] is True
    # Own label must not appear as the sole reason — train grades conflict
    assert set(out["train_near_dup_grades"]) >= {0, 3}
    assert 1 not in out["train_near_dup_grades"] or len(set(out["train_near_dup_grades"])) > 1


def test_rm6_b2_bound_and_sigma_t_ge_sigma_c():
    assert _SCALE_BOUNDS["b2"] == (0.0, 1.0)
    row = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 60.0,  # ≥ σc
        "scf": 2.0,
        "b1": 50 / 60,
        "b2": 1.5,  # out of (0,1)
        "wet": 5.0,
        "d": 500.0,
    }
    res = check_case(row)
    assert any("sigma_t_ge_sigma_c" in i for i in res.issues)
    assert any("b2_out_of_bounds" in i for i in res.issues)


def test_rm6_ood_and_failed_physics_abstains():
    cfg = load_config()
    from rockburst.fusion.stacker import ReliabilityFusion

    fus = ReliabilityFusion()
    fus.coef_ = np.zeros(7)
    fus.coef_[2] = 8.0
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
    dq = DataQualityOutput(data_reliability=0.4, ood_flag=True)
    phys = PhysicsOutput(physics_score=0.333, issues=["SCF OOD"])
    pred = PredictiveOutput(
        predicted_grade=3,
        probabilities=[0.001, 0.001, 0.001, 0.997],
        r_model=0.997,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.1,
        r_uncertainty=0.9,
        prediction_set=[3],
        model_disagreement=0.05,
        prediction_stability="high",
        stability_score=0.95,
        entropy=0.1,
    )
    critic = run_critic(dq, phys, pred, unc, cfg)
    assert "ood_and_failed_physics" in critic.rules_fired
    assert critic.escalate_abstain is False
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


def test_rm6_monotonic_tendency_helper():
    class _Pipe:
        def predict_proba(self, X):
            # Higher wet → lower grade (violation once wet crosses 5.2)
            wet = float(X.iloc[0]["wet"])
            g = 3 if wet < 5.2 else 0
            p = np.zeros(4)
            p[g] = 1.0
            return np.array([p])

    case = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "scf": 2.0,
        "b1": 10.0,
        "b2": 0.82,
        "wet": 5.0,
        "d": 500.0,
    }
    out = monotonic_tendency_recheck(case, pipeline=_Pipe(), primary_grade=3)
    assert out["monotonic_violation"] is True
    assert "wet" in out["violations"]


def test_rc1_missing_data_cap_mirrors_robustness():
    """RC1: full mode max_per_fold is None (not 24); quick keeps a cap."""
    import inspect

    from rockburst.eval import ablation as abl

    src = inspect.getsource(abl._missing_data_all_folds)
    assert "8 if quick else None" in src
    assert "8 if quick else 24" not in src
    # Robustness already uses the same full-mode pattern.
    rob = inspect.getsource(abl._robustness_all_folds)
    assert "else None" in rob


def test_rc6_imputer_score_emits_mean_delta_r_final(cfg_df):
    """RC6: use_m7=False path fills mean_delta_r_final via fusion (not all null)."""
    from rockburst.eval.ablation import _score_missing_batch
    from rockburst.fusion.stacker import ReliabilityFusion
    from rockburst.ml.pipelines import build_preprocess, get_model_zoo, prepare_xy
    from rockburst.physics.relations import FEATURE_COLS
    from sklearn.pipeline import Pipeline

    cfg, df = cfg_df
    X, y, _ = prepare_xy(df)
    complete = df.dropna(subset=FEATURE_COLS).head(3)
    if len(complete) < 2:
        pytest.skip("need complete rows")
    idxs = complete.index.tolist()
    fit_idx = np.arange(min(40, len(df)))
    zoo = get_model_zoo(quick=True)
    est = next(iter(zoo.values()))
    pipe = Pipeline([("prep", build_preprocess("median")), ("model", est)])
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    fus = ReliabilityFusion()
    fus.coef_ = np.zeros(7)
    fus.coef_[2] = 4.0
    fus.intercept_ = 0.0
    fus.is_fit = True
    fold_cfg = {
        **cfg,
        "_fold_pipeline": pipe,
        "_fold_fusion": fus,
        "_skip_local_explanation": True,
    }
    out = _score_missing_batch(
        df=df,
        indices=idxs,
        mask_cols=("d",),
        pipe=pipe,
        fold_cfg=fold_cfg,
        fit_idx=fit_idx,
        use_m7=False,
        use_physics_recon=False,
    )
    assert "mean_delta_r_final" in out
    assert out["mean_delta_r_final"] is not None
    assert np.isfinite(out["mean_delta_r_final"])


def test_rm1_recheck_model_can_change_prediction_set(cfg_df):
    """RM1: model recheck with disagreement widens prediction_set."""
    from rockburst.ml.pipelines import prepare_xy

    cfg, df = cfg_df
    case = df.iloc[0].to_dict()
    state = CaseState(
        case=case,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(
            physics_score=0.9,
            empirical_checks={"criteria_as_given": {"consensus_grade": 1}},
        ),
        pred=PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.1, 0.7, 0.1, 0.1],
            r_model=0.7,
        ),
        unc=UncertaintyOutput(
            uncertainty_score=0.2,
            r_uncertainty=0.8,
            prediction_set=[1],
            model_disagreement=0.05,
            prediction_stability="high",
            stability_score=0.9,
            entropy=0.3,
        ),
        critic=CriticOutput(
            challenges=["x"],
            endorse=False,
            request_recheck="model",
            rules_fired=["ensemble_disagreement"],
        ),
        extra_validation_done=True,
    )
    X, y, _ = prepare_xy(df)
    cfg = {
        **cfg,
        "_fold_X": X.iloc[:40],
        "_fold_y": y[:40],
        "_fold_model_name": "logistic__median",
        "_fold_cross_pipes": {},
        "_skip_local_explanation": True,
    }
    before_set = list(state.unc.prediction_set)
    state = step(state, df, cfg, use_llm=False, allow_critic=True)
    assert any(t.endswith(":recheck_model") for t in state.route_trace)
    assert state.recheck_effects
    assert "changed_set" in state.recheck_effects[-1]
    if state.recheck_payloads.get("cross_check", {}).get("large_disagreement"):
        assert state.recheck_effects[-1]["changed_set"] is True or list(
            state.unc.prediction_set
        ) != before_set


def test_rm2_critic_rules_union_survives_recheck(cfg_df):
    """RM2: rules from first Critic visit remain in union after recheck clears critic."""
    cfg, df = cfg_df
    case = df.iloc[0].to_dict()
    res = full_physics_result(case, cfg)
    state = CaseState(
        case=case,
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(**res.to_dict()),
        pred=PredictiveOutput(
            predicted_grade=0,
            probabilities=[0.4, 0.2, 0.2, 0.2],
            r_model=0.4,
        ),
        unc=UncertaintyOutput(
            uncertainty_score=0.5,
            r_uncertainty=0.5,
            prediction_set=[0, 1, 2],
            model_disagreement=0.1,
            prediction_stability="low",
            stability_score=0.4,
            entropy=1.0,
        ),
        critic=CriticOutput(
            challenges=["Large set"],
            endorse=False,
            request_recheck="physics",
            rules_fired=["large_prediction_set", "edge_of_wide_set"],
        ),
        critic_rules_union=["large_prediction_set", "edge_of_wide_set"],
        extra_validation_done=True,
    )
    state = step(state, df, cfg, use_llm=False, allow_critic=True)
    assert state.critic is None  # cleared after recheck
    assert "large_prediction_set" in state.critic_rules_union
    assert "edge_of_wide_set" in state.critic_rules_union


def test_rm4_cheap_precheck_skips_criteria_cost(cfg_df):
    """RM4: cheap path has no criteria; clean attach equals full R_physics."""
    from rockburst.physics.precheck import (
        cheap_physics_precheck,
        identities_and_ranges_only,
        physics_for_geomechanics,
    )

    cfg, df = cfg_df
    clean = None
    full = None
    for _, row in df.iterrows():
        case = row.to_dict()
        partial = identities_and_ranges_only(case, cfg)
        if is_perfect_physics(partial):
            clean = case
            full = full_physics_result(case, cfg)
            break
    if clean is None:
        pytest.skip("no clean physics row")
    cheap = cheap_physics_precheck(clean, cfg)
    assert cheap["clean"] is True
    assert "criteria_as_given" not in (cheap.get("empirical_checks") or {})
    routed = physics_for_geomechanics(clean, cfg)
    assert "criteria_as_given" in (routed.empirical_checks or {})
    assert abs(float(routed.r_physics) - float(full.r_physics)) < 1e-9
    # Dirty case still gets full check (criteria_recomputed present).
    dirty = dict(clean)
    dirty["scf"] = float(dirty["sigma_theta"]) / float(dirty["sigma_c"]) + 5.0
    dirty_res = physics_for_geomechanics(dirty, cfg)
    assert "criteria_recomputed" in (dirty_res.empirical_checks or {})


def test_rm5_criterion_label_agreement_table_usable(cfg_df):
    """RM5: agreement table returns consensus precision for severity docs."""
    from rockburst.physics.criteria import criterion_label_agreement_table

    cfg, df = cfg_df
    rows = [df.iloc[i].to_dict() for i in range(min(30, len(df)))]
    labels = [int(df.iloc[i]["rockburst_grade"]) for i in range(min(30, len(df)))]
    table = criterion_label_agreement_table(rows, labels)
    assert table["n"] == len(rows)
    assert "consensus" in table["agreement"]
    assert 0.0 <= table["agreement"]["consensus"] <= 1.0


def test_tc8_runner_stores_local_explanation_per_case(cfg_df):
    """TC8: two cases sharing a fold config retain distinct explanations."""
    from rockburst.graph.runner import _maybe_attach_local_explanation
    from rockburst.ml.pipelines import build_preprocess, get_model_zoo, prepare_xy
    from sklearn.pipeline import Pipeline

    cfg, df = cfg_df
    case = df.iloc[0].to_dict()
    X, y, _ = prepare_xy(df)
    zoo = get_model_zoo(quick=True)
    pipe = Pipeline(
        [("prep", build_preprocess("median")), ("model", next(iter(zoo.values())))]
    )
    pipe.fit(X.iloc[:40], y[:40])
    state = CaseState(
        case=case,
        pred=PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.1, 0.6, 0.2, 0.1],
            r_model=0.6,
        ),
    )
    fold_cfg = {**cfg, "_fold_pipeline": pipe}
    _maybe_attach_local_explanation(state, fold_cfg)
    assert isinstance(state.local_explanation, dict)
    assert set(state.local_explanation) <= {"scf", "wet", "sigma_theta"}
    assert "_local_explanation" not in fold_cfg

    case2 = df.iloc[1].to_dict()
    state2 = CaseState(
        case=case2,
        pred=PredictiveOutput(
            predicted_grade=2,
            probabilities=[0.1, 0.2, 0.6, 0.1],
            r_model=0.6,
        ),
    )
    _maybe_attach_local_explanation(state2, fold_cfg)
    assert isinstance(state2.local_explanation, dict)
    assert state2.local_explanation is not state.local_explanation
    assert "_local_explanation" not in fold_cfg
    # Quick gate
    fold_cfg2 = {**cfg, "_fold_pipeline": pipe, "_skip_local_explanation": True}
    state3 = CaseState(case=case, pred=state.pred)
    _maybe_attach_local_explanation(state3, fold_cfg2)
    assert state3.local_explanation is None


def test_tc8_complete_case_cannot_create_imputation_objection(cfg_df):
    cfg, df = cfg_df
    case = df.iloc[0].to_dict()
    for column in ("sigma_theta", "sigma_c", "sigma_t", "scf", "b1", "b2", "wet", "d"):
        if pd.isna(case[column]):
            case[column] = float(df[column].dropna().median())
    out = alt_imputer_reprediction(
        case,
        cfg,
        X_train=None,
        y_train=None,
        primary_grade=1,
        missing_features=[],
    )
    assert out["skipped"] is True
    assert out["skip_reason"] == "complete_case_no_imputed_features"
    assert out["large_spread"] is False
    assert validation_objections_from_rechecks({"alt_imputer": out}) == []


def test_tc8_alt_imputers_reuse_production_model_family(cfg_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from rockburst.ml.pipelines import build_preprocess, prepare_xy

    cfg, df = cfg_df
    X, y, _ = prepare_xy(df)
    production = Pipeline(
        [
            ("prep", build_preprocess("median")),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    production.fit(X.iloc[:80], y[:80])
    case = df.iloc[0].to_dict()
    case["d"] = np.nan
    out = alt_imputer_reprediction(
        case,
        {**cfg},
        X_train=X.iloc[:80],
        y_train=y[:80],
        primary_imputer="median",
        primary_grade=1,
        primary_pipeline=production,
        missing_features=["d"],
    )
    assert out["skipped"] is False
    assert out["model_family"] == "LogisticRegression"
    assert set(out["imputers"]) == {"median", "knn", "iterative"}


def test_tc1_model_recheck_resolves_structured_objection():
    cfg = load_config()
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    phys = PhysicsOutput(physics_score=0.9)
    pred = PredictiveOutput(
        predicted_grade=2,
        probabilities=[0.05, 0.15, 0.7, 0.1],
        r_model=0.7,
    )
    unc = UncertaintyOutput(
        uncertainty_score=0.4,
        r_uncertainty=0.6,
        prediction_set=[1, 2, 3],
        model_disagreement=0.5,
        prediction_stability="moderate",
        stability_score=0.7,
        entropy=0.5,
    )
    critic = run_critic(dq, phys, pred, unc, cfg)
    state = CaseState(case={}, data_q=dq, phys=phys, pred=pred, unc=unc, critic=critic)
    _record_critic_rules(state)
    assert state.objection_records
    _resolve_rechecked_objections(
        state,
        kind="model",
        payload={
            "primary_grade": 2,
            "cross_grade": 1,
            "soft_disagreement": 0.2,
            "resolution_threshold": 0.35,
        },
    )
    assert "ensemble_disagreement" in state.resolved_critic_rules
    assert any(
        r["rule"] == "ensemble_disagreement" and r["status"] == "resolved"
        for r in state.objection_records
    )
    cleared = run_critic(
        dq,
        phys,
        pred,
        unc,
        cfg,
        resolved_rules=set(state.resolved_critic_rules),
    )
    assert "ensemble_disagreement" not in cleared.rules_fired


def test_tc1_identical_recheck_is_not_repeated():
    state = CaseState(
        case={},
        data_q=DataQualityOutput(data_reliability=0.9, ood_flag=False),
        phys=PhysicsOutput(physics_score=0.9),
        pred=PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.1, 0.7, 0.1, 0.1],
            r_model=0.7,
        ),
        unc=UncertaintyOutput(
            uncertainty_score=0.2,
            r_uncertainty=0.8,
            prediction_set=[0, 1, 2],
            model_disagreement=0.4,
            prediction_stability="moderate",
            stability_score=0.7,
            entropy=0.4,
        ),
        critic=CriticOutput(
            endorse=False,
            request_recheck="model",
            rules_fired=["ensemble_disagreement"],
        ),
        extra_validation_done=True,
        recheck_counts={"physics": 0, "data": 0, "model": 1},
    )
    assert rule_based_next_action(state, load_config()) == "finalize"
    assert state.termination_reason == "finalize_unresolved_no_new_evidence"


def test_tm3_kirsch_range_and_case194_route(cfg_df):
    cfg, df = cfg_df
    lo, hi = kirsch_tangential_stress_range(500.0)
    assert lo == pytest.approx(27.0)
    assert hi == pytest.approx(148.5)
    row = df.loc[df["no"] == 194].iloc[0].to_dict()
    res = check_case(row)
    assert any("tangential_stress_exceeds_strength" in i for i in res.issues)
    critic = run_critic(
        DataQualityOutput(data_reliability=0.9, ood_flag=False),
        PhysicsOutput(**res.to_dict()),
        PredictiveOutput(
            predicted_grade=1,
            probabilities=[0.01, 0.97, 0.01, 0.01],
            r_model=0.97,
        ),
        UncertaintyOutput(
            uncertainty_score=0.1,
            r_uncertainty=0.9,
            prediction_set=[1],
            model_disagreement=0.0,
            prediction_stability="high",
            stability_score=0.95,
            entropy=0.1,
        ),
        cfg,
    )
    assert "tangential_stress_exceeds_strength" in critic.rules_fired
    assert critic.request_recheck == "physics"
    assert critic.disagreement_level in {"Moderate", "High"}
    assert "predictive_agent" in critic.disagreeing_parties


def test_tm1_rule_severity_and_disablement_come_from_intervals():
    stats = {
        "harmful": {
            "auroc_ci95": [0.58, 0.72],
            "error_delta_ci95": [0.04, 0.20],
        },
        "protective": {
            "auroc_ci95": [0.31, 0.47],
            "error_delta_ci95": [-0.30, -0.05],
        },
        "uncertain": {
            "auroc_ci95": [0.45, 0.57],
            "error_delta_ci95": [-0.02, 0.09],
        },
    }
    assert severe_rules_from_stats(stats) == {"harmful"}
    assert disabled_rules_from_stats(stats) == {"protective"}
