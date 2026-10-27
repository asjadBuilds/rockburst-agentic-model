"""Phase 5 helpers: Rm8 run metadata + RM12/RM13 sensitivity (no full train)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rockburst.eval.run_meta import build_run_metadata
from rockburst.eval.sensitivity import (
    assign_triplet_group_ids,
    drop_rows_315_344_mask,
    earliest_of_conflict_mask,
    label_policy_frames,
    missing_d_vs_grade,
    triplet_inventory,
    triplet_key,
)


def test_run_metadata_fields():
    meta = build_run_metadata(mode="full", command="rockburst evaluate")
    assert meta["mode"] == "full"
    assert meta["command"] == "rockburst evaluate"
    assert "date_utc" in meta and "T" in meta["date_utc"]
    assert meta["rockburst_version"]
    assert "git_dirty" in meta
    assert isinstance(meta["package_versions"], dict)
    assert "numpy" in meta["package_versions"]


def test_triplet_group_ids_share_identical_keys():
    df = pd.DataFrame(
        {
            "no": [1, 2, 3, 4],
            "sigma_c": [100.0, 100.0, 80.0, 100.00001],
            "sigma_t": [5.0, 5.0, 4.0, 5.0],
            "wet": [2.0, 2.0, 1.5, 2.0],
            "rockburst_grade": [1, 2, 0, 1],
        }
    )
    ids = assign_triplet_group_ids(df)
    # rows 0,1,3 share rounded (100,5,2); row 2 alone
    assert ids[0] == ids[1] == ids[3]
    assert ids[2] != ids[0]
    inv = triplet_inventory(df)
    assert inv["n_triplets"] == 2
    assert inv["n_multi_row_triplets"] == 1
    assert inv["n_conflicting_grade_triplets"] == 1


def test_triplet_key_handles_missing():
    row = pd.Series({"sigma_c": 1.0, "sigma_t": np.nan, "wet": 2.0})
    assert None in triplet_key(row)


def test_label_policies_and_missing_d():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "no": np.arange(1, n + 1),
            "sigma_c": rng.uniform(50, 150, n),
            "sigma_t": rng.uniform(2, 10, n),
            "sigma_theta": rng.uniform(20, 80, n),
            "scf": rng.uniform(0.2, 1.0, n),
            "b1": rng.uniform(5, 30, n),
            "b2": rng.uniform(0.5, 0.95, n),
            "wet": rng.uniform(1, 8, n),
            "d": [np.nan if i % 3 == 0 else float(500 + i) for i in range(n)],
            "rockburst_grade": rng.integers(0, 4, n),
        }
    )
    # Force a 315–344 style block via nos
    df.loc[df["no"] >= 30, "no"] = np.arange(315, 315 + (n - 29))

    policies = label_policy_frames(df, rel_tol=0.05)
    assert set(policies) == {
        "as_given",
        "drop_rows_315_344",
        "earliest_of_conflict",
    }
    assert len(policies["as_given"]) == n
    assert len(policies["drop_rows_315_344"]) < n

    keep = drop_rows_315_344_mask(df)
    assert keep.sum() == len(policies["drop_rows_315_344"])

    cmp_ = {
        "as_given": missing_d_vs_grade(df),
        "without_block": missing_d_vs_grade(df.loc[keep]),
    }
    assert "with_d" in cmp_["as_given"] and "missing_d" in cmp_["as_given"]
    assert earliest_of_conflict_mask(df).dtype == bool
