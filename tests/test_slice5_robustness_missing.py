"""Slice 5: physical missing masks, physics reconstruction, robustness helpers."""

from __future__ import annotations

import numpy as np

from rockburst.eval.ablation import _mean_boot_ci
from rockburst.physics.relations import (
    PHYSICAL_MISSING_MASKS,
    apply_missing_mask,
    compute_scf,
    reconstruct_missing_from_physics,
)


def test_physical_masks_cover_linked_features():
    assert "scf" in PHYSICAL_MISSING_MASKS["sigma_theta_scf"]
    assert set(PHYSICAL_MISSING_MASKS["sigma_c_scf_b1_b2"]) >= {
        "sigma_c",
        "scf",
        "b1",
        "b2",
    }
    assert set(PHYSICAL_MISSING_MASKS["sigma_t_b1_b2"]) >= {"sigma_t", "b1", "b2"}


def test_reconstruct_sigma_theta_from_scf_times_sigma_c():
    case = {
        "sigma_theta": float("nan"),
        "sigma_c": 100.0,
        "sigma_t": 10.0,
        "scf": 0.5,
        "b1": float("nan"),
        "b2": float("nan"),
        "wet": 5.0,
        "d": 500.0,
    }
    out = reconstruct_missing_from_physics(case)
    assert abs(out["sigma_theta"] - 50.0) < 1e-9
    assert abs(out["scf"] - compute_scf(out["sigma_theta"], out["sigma_c"])) < 1e-9
    assert abs(out["b1"] - 10.0) < 1e-9


def test_apply_missing_mask_sets_nan():
    case = {"sigma_theta": 1.0, "scf": 2.0, "sigma_c": 3.0}
    out = apply_missing_mask(case, ("sigma_theta", "scf"))
    assert np.isnan(out["sigma_theta"]) and np.isnan(out["scf"])
    assert out["sigma_c"] == 3.0
    assert case["sigma_theta"] == 1.0  # original untouched


def test_mean_boot_ci_shape():
    out = _mean_boot_ci([0.1, 0.2, 0.3, 0.4], n_boot=50, seed=0)
    assert out["n"] == 4
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
