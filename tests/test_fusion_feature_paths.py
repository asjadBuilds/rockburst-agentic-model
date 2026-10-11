"""Production / ablation / item4 share one reliability feature builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import (
    CHOSEN_FUSION_FEATURES,
    CHOSEN_FUSION_SET,
    FEATURE_NAMES,
    locked_feature_mask,
)
from rockburst.ml import train as train_mod
from rockburst.eval import ablation as abl_mod


def _load_item4():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "item4_metrics", root / "scripts" / "item4_metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_locked_set_is_maxprob_only():
    assert CHOSEN_FUSION_SET == "A"
    assert CHOSEN_FUSION_FEATURES == ("r_model",)
    mask = locked_feature_mask()
    assert mask.shape == (len(FEATURE_NAMES),)
    assert int(mask.sum()) == 1
    assert FEATURE_NAMES[int(np.flatnonzero(mask)[0])] == "r_model"


def test_three_paths_build_identical_feature_vectors():
    item4 = _load_item4()
    assert train_mod.compute_reliability_features is compute_reliability_features
    assert abl_mod.compute_reliability_features is compute_reliability_features
    assert item4.compute_reliability_features is compute_reliability_features

    dq = SimpleNamespace(data_reliability=0.8)
    phys = SimpleNamespace(physics_score=0.7)
    pred = SimpleNamespace(r_model=0.55, observed_mask=None)
    unc = SimpleNamespace(r_uncertainty=0.6, stability_score=0.5)
    case = {
        "scf": 1.0,
        "b1": 1.0,
        "b2": 1.0,
        "sigma_theta": 1.0,
        "sigma_c": 1.0,
        "sigma_t": 1.0,
        "wet": 1.0,
        "d": 100.0,
    }
    r_shared = compute_reliability_features(dq, phys, pred, unc, case=case)
    r_train = train_mod.compute_reliability_features(dq, phys, pred, unc, case=case)
    r_abl = abl_mod.compute_reliability_features(dq, phys, pred, unc, case=case)
    r_i4 = item4.compute_reliability_features(dq, phys, pred, unc, case=case)
    assert np.allclose(r_shared, r_train)
    assert np.allclose(r_shared, r_abl)
    assert np.allclose(r_shared, r_i4)
    assert r_shared.shape == (len(FEATURE_NAMES),)
