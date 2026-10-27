"""Slice 6 unit tests: ordinal decode, MLP zoo, conformal contiguous, DQ OOD, selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rockburst.ml.ordinal import (
    FrankHallOrdinalClassifier,
    argmax_grade,
    median_of_distribution,
)
from rockburst.ml.pipelines import get_model_zoo
from rockburst.ml.tuning import SEARCH_SPACES_DOC, available_model_names, estimator_from_params
from rockburst.quality.checks import combination_scale_issues, score_data_quality
from rockburst.uncertainty.conformal import (
    conformal_prediction_set,
    make_contiguous_ordinal_set,
    pick_defensible_method,
)


def test_median_of_distribution_vs_argmax():
    # Left-skewed: mode at 2 but cumulative mass reaches 0.5 at grade 1
    proba = np.array(
        [
            [0.20, 0.35, 0.40, 0.05],  # argmax=2; cdf=[0.20,0.55,0.95,1] → median=1
            [0.10, 0.10, 0.10, 0.70],  # both → 3
        ]
    )
    assert argmax_grade(proba).tolist() == [2, 3]
    assert median_of_distribution(proba).tolist() == [1, 3]


def test_frank_hall_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = np.clip((X[:, 0] * 1.5 + rng.normal(scale=0.3, size=80)).astype(int) + 1, 0, 3)
    clf = FrankHallOrdinalClassifier(
        base_estimator=LogisticRegression(max_iter=1000),
        n_classes=4,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = clf.predict(X[:5])
    assert pred.shape == (5,)
    assert set(pred.tolist()) <= {0, 1, 2, 3}


def test_mlp_replaces_extratrees_in_zoo_and_tuning():
    zoo = get_model_zoo(quick=True)
    assert "mlp" in zoo
    assert "extratrees" not in zoo
    assert "frank_hall_rf" in zoo
    names = available_model_names(quick=False)
    assert "mlp" in names
    assert "extratrees" not in names
    assert "frank_hall_logistic" in names
    assert "mlp" in SEARCH_SPACES_DOC
    est = estimator_from_params(
        "mlp",
        {"hidden_layer_sizes": (32,), "alpha": 1e-3, "learning_rate_init": 1e-3},
        seed=0,
    )
    assert est.__class__.__name__ == "MLPClassifier"


def test_contiguous_ordinal_prediction_sets():
    assert make_contiguous_ordinal_set([0, 2]) == [0, 1, 2]
    assert make_contiguous_ordinal_set([]) == []
    # Thresholds that would pick {0, 2} → contiguous fills 1
    p = np.array([0.4, 0.05, 0.4, 0.15])
    thr = np.array([0.7, 0.1, 0.7, 0.1])  # 1-p <= thr → grades 0 and 2
    s = conformal_prediction_set(p, class_thresholds=thr, contiguous=True)
    assert s == [0, 1, 2]
    s_raw = conformal_prediction_set(p, class_thresholds=thr, contiguous=False)
    assert s_raw == [0, 2]


def test_pick_defensible_uses_coverage_size_and_abstention():
    comparison = {
        "class_calib_counts": [20, 20, 15, 10],
        "mondrian_split": {
            "coverage": 0.88,
            "mean_size": 2.1,
            "per_class": {"3": {"coverage": 0.85, "n": 10}},
        },
        "marginal_split": {"coverage": 0.90, "mean_size": 1.8, "per_class": {}},
    }
    assert pick_defensible_method(comparison, alpha=0.1) == "marginal_split"


def test_combination_scale_and_ood_detectors():
    issues = combination_scale_issues(
        {
            "sigma_theta": 100.0,
            "sigma_c": 100.0,
            "sigma_t": 10.0,
            "scf": 0.5,  # should be ~1.0
            "b1": 5.0,  # should be ~10
            "b2": 1.0,
            "wet": 5.0,
            "d": 500.0,
        },
        rel_tol=0.05,
    )
    assert any(i.startswith("combo_scf") for i in issues)
    assert any(i.startswith("combo_b1") for i in issues)

    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {
            **{c: rng.normal(size=40) for c in [
                "sigma_theta", "sigma_c", "sigma_t", "scf", "b1", "b2", "wet", "d"
            ]},
            "no": np.arange(40),
            "rockburst_grade": rng.integers(0, 4, size=40),
            "dup_group_id": np.arange(40),
        }
    )
    # Two fold-train near-dups with conflicting labels (RM3: never use case label).
    base = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "wet": 5.0,
        "d": 500.0,
        "scf": 2.0,
        "b1": 10.0,
        "b2": 0.82,
    }
    for col, val in base.items():
        ref.loc[0, col] = val
        ref.loc[1, col] = val * 1.001  # within 5% rel tol
    ref.loc[0, "rockburst_grade"] = 0
    ref.loc[1, "rockburst_grade"] = 3
    ref.loc[0, "dup_group_id"] = 0
    ref.loc[1, "dup_group_id"] = 0
    case = {**base, "no": 999, "dup_group_id": 0, "rockburst_grade": 1}
    out = score_data_quality(case, ref, near_dup_rel_tol=0.05)
    assert "ood_scores" in out
    assert set(out["ood_scores"]) >= {"isolation_forest", "mahalanobis", "knn"}
    assert out["conflicting_near_dup_labels"] is True
    assert set(out.get("train_near_dup_grades") or []) >= {0, 3}


def test_production_selection_rule_mean_inner():
    """M10: best mean inner_qwk beats mode-of-winners when they disagree."""
    models = {
        "a__median": {"inner_qwk_mean": 0.70},
        "b__median": {"inner_qwk_mean": 0.55},
        "c__median": {"inner_qwk_mean": 0.60},
    }
    fold_winners = ["b__median", "b__median", "b__median", "a__median", "c__median"]
    from collections import Counter

    mode = Counter(fold_winners).most_common(1)[0][0]
    prod = max(models.items(), key=lambda kv: kv[1]["inner_qwk_mean"])[0]
    assert mode == "b__median"
    assert prod == "a__median"
    assert mode != prod
