"""Phase 4: RM8–RM11 + RC4/RC6 metric hooks (cheap unit tests only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import run_critic
from rockburst.agents.schemas import (
    DataQualityOutput,
    PhysicsOutput,
    PredictiveOutput,
    UncertaintyOutput,
)
from rockburst.eval.metrics import (
    capacity_matched_family_table,
    matched_coverage_metrics,
)
from rockburst.ml.explain import (
    aggregate_fold_importances,
    check_implausible_explanation_directions,
    local_permutation_attributions,
)
from rockburst.ml.ordinal import median_of_distribution
from rockburst.ml.pipelines import build_preprocess
from rockburst.ml.tuning import SEARCH_SPACES_DOC, estimator_from_params
from rockburst.uncertainty.conformal import (
    compare_conformal_methods,
    coverage_size_abstention_tradeoff,
    fit_class_conditional_alpha,
    fit_ordinal_thresholds,
    ordinal_nonconformity,
    set_size_distribution,
)


def test_rm8_ordinal_nonconformity_and_set_size_schema():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=40)
    P = rng.dirichlet(np.ones(4), size=40)
    scores = ordinal_nonconformity(y, P)
    assert scores.shape == (40,)
    assert np.all(scores >= 0)
    conf = fit_ordinal_thresholds(y[:20], P[:20], alpha=0.1)
    assert conf.method == "ordinal_split"
    conf3 = fit_class_conditional_alpha(y[:20], P[:20], alpha=0.1, focus_class=3)
    assert conf3.method == "class_conditional_grade3"
    sets = [[0], [0, 1], [0, 1, 2], [1, 2, 3]] * 10
    dist = set_size_distribution(y, sets, fold=0, large_set_size=3)
    assert "size_hist" in dist and "per_class" in dist
    assert dist["fold"] == 0


def test_rm8_compare_methods_includes_tradeoff():
    rng = np.random.default_rng(1)
    y_cal = rng.integers(0, 4, size=30)
    p_cal = rng.dirichlet(np.ones(4), size=30)
    y_te = rng.integers(0, 4, size=20)
    p_te = rng.dirichlet(np.ones(4), size=20)
    cmp_ = compare_conformal_methods(
        y_cal, p_cal, y_te, p_te, alpha=0.1, cv_plus_scores=np.ones(30) * 0.2
    )
    assert "ordinal_split" in cmp_
    assert "class_conditional_grade3" in cmp_
    assert "tradeoff" in cmp_
    trade = coverage_size_abstention_tradeoff(
        {k: cmp_[k] for k in ("mondrian_split", "ordinal_split") if k in cmp_}
    )
    assert trade["methods"]


def test_rm9_implausible_explanation_critic():
    flags = check_implausible_explanation_directions(
        {"scf": -0.1, "wet": 0.01, "sigma_theta": -0.05},
        predicted_grade=3,
    )
    assert any("scf" in f for f in flags)
    crit = run_critic(
        DataQualityOutput(data_reliability=0.9, ood_flag=False),
        PhysicsOutput(physics_score=0.9),
        PredictiveOutput(
            predicted_grade=3,
            probabilities=[0.05, 0.05, 0.1, 0.8],
            r_model=0.9,
        ),
        UncertaintyOutput(
            uncertainty_score=0.1,
            r_uncertainty=0.9,
            prediction_set=[3],
            model_disagreement=0.0,
            prediction_stability="high",
            stability_score=0.95,
            entropy=0.2,
        ),
        {},
        local_explanation={"scf": -0.08, "wet": 0.0, "sigma_theta": 0.0},
    )
    assert "implausible_explanation_direction" in crit.rules_fired
    assert crit.endorse is False


def test_rm9_local_perm_and_aggregate_keys():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int).to_numpy()
    pipe = Pipeline(
        [("model", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X, y)
    attrib = local_permutation_attributions(
        pipe, X.iloc[[0]], feature_names=list(X.columns), predicted_grade=int(y[0])
    )
    assert set(attrib) >= {"a", "b", "c"}
    agg = aggregate_fold_importances(
        [
            {
                "features": ["a", "b", "c"],
                "importances_mean": [0.2, 0.1, 0.05],
            },
            {
                "features": ["a", "b", "c"],
                "importances_mean": [0.3, 0.05, 0.1],
            },
        ]
    )
    assert "importances_mean" in agg and "ranked" in agg
    assert agg["n_folds"] == 2
    assert agg["ranked"][0]["feature"] == "a"


def test_rm10_capacity_matched_and_median_decode():
    models = {
        "rf__median": {
            "pooled_oof": {"qwk": 0.65},
            "inner_qwk_mean": 0.62,
        },
        "frank_hall_rf__median": {
            "pooled_oof": {"qwk": 0.66},
            "inner_qwk_mean": 0.61,
        },
        "logistic__knn": {
            "pooled_oof": {"qwk": 0.55},
            "inner_qwk_mean": 0.50,
        },
    }
    tab = capacity_matched_family_table(models)
    assert any(r["multiclass"] == "rf__median" for r in tab["pairs"])
    assert all(r.get("ci95_delta_qwk") is None for r in tab["pairs"])
    # With fold OOF vectors, CI must be populated (train path hook).
    rng = np.random.default_rng(1)
    n = 32
    y = rng.integers(0, 4, size=n)
    pm = y.copy()
    po = y.copy()
    pm[::5] = (pm[::5] + 1) % 4
    tab_ci = capacity_matched_family_table(
        models,
        oof_pred_by_model={"rf__median": pm, "frank_hall_rf__median": po},
        y_true=y,
        groups=np.repeat(np.arange(8), 4),
        n_boot=20,
        seed=1,
    )
    row = next(r for r in tab_ci["pairs"] if r["multiclass"] == "rf__median")
    assert row["ci95_delta_qwk"] is not None and len(row["ci95_delta_qwk"]) == 2
    proba = np.array([[0.2, 0.35, 0.4, 0.05]])
    assert int(median_of_distribution(proba)[0]) == 1


def test_rm11_search_space_broadened_and_svm_builds():
    assert "min_samples_leaf" in SEARCH_SPACES_DOC["rf"]
    assert "subsample" in SEARCH_SPACES_DOC["xgboost"]
    assert "l2_leaf_reg" in SEARCH_SPACES_DOC["catboost"]
    rf = estimator_from_params(
        "rf",
        {
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
        },
        seed=0,
    )
    assert rf.min_samples_leaf == 3
    svm = estimator_from_params(
        "svm", {"kernel": "linear", "C": 1.0}, seed=0
    )
    assert svm.__class__.__name__ == "CalibratedClassifierCV"
    # Preprocess still scales before SVM in Pipeline
    prep = build_preprocess("median")
    assert "scaler" in dict(prep.steps)


def test_tc5_catboost_subsample_uses_compatible_bootstrap():
    pytest.importorskip("catboost")
    est = estimator_from_params(
        "catboost",
        {
            "iterations": 4,
            "depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "l2_leaf_reg": 1.0,
        },
        seed=0,
    )
    assert est.get_param("bootstrap_type") == "Bernoulli"
    X = np.arange(64, dtype=float).reshape(16, 4)
    y = np.tile(np.arange(4), 4)
    est.fit(X, y)


def test_tc5_required_candidate_failure_is_fail_fast():
    from rockburst.ml.train import _fit_required_candidate

    class BrokenPipeline:
        def fit(self, X, y):
            raise ValueError("deliberate fit failure")

    with pytest.raises(RuntimeError, match="catboost__median.*outer fold 2"):
        _fit_required_candidate(
            BrokenPipeline(),
            np.zeros((4, 2)),
            np.arange(4),
            candidate="catboost__median",
            fold=2,
        )


def test_rc4_per_system_same_coverage_comparison():
    y = np.array([0, 1, 2, 3, 1, 2])
    systems = {
        "M1_maxprob": {
            "y": y, "pred": y, "abstain": np.zeros(6),
            "confidence": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        },
        "M3": {"y": y, "pred": y, "abstain": np.array([0, 1, 0, 0, 0, 0])},
        "M4": {"y": y, "pred": y, "abstain": np.array([0, 0, 1, 0, 0, 0])},
        "M7": {"y": y, "pred": np.array([0, 1, 2, 0, 1, 2]), "abstain": np.zeros(6)},
    }
    out = matched_coverage_metrics(systems, n_boot=20, seed=0)
    assert out["n_total"] == 6
    assert out["comparisons"]["M3"]["n_accepted"] == 5
    assert out["comparisons"]["M4"]["coverage"] == pytest.approx(5 / 6)
    assert out["comparisons"]["M7"]["system"]["accuracy"] is not None
    assert "accuracy_ci95" in out["comparisons"]["M7"]["delta_system_minus_m1"]


def test_rc6_missing_scenario_keys_documented():
    from rockburst.eval import ablation as abl

    src = open(abl.__file__).read()
    assert "physics_recon_theta_scf_missing" in src
    assert "train_vs_inference_missing_d_protocol" in src
    assert "includes_iterative_imputer" in src
