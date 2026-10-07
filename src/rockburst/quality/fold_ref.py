"""Fold-aware reference frames for leakage-safe quality scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_fold_reference(
    df: pd.DataFrame,
    train_idx,
    case: dict[str, Any],
    *,
    group_col: str = "dup_group_id",
) -> pd.DataFrame:
    """Training-fold reference excluding the case and its duplicate group."""
    ref = df.iloc[list(train_idx)].copy()
    case_no = case.get("no")
    if case_no is not None and "no" in ref.columns:
        ref = ref[ref["no"].astype(int) != int(case_no)]
    case_gid = case.get(group_col)
    if case_gid is not None and group_col in ref.columns:
        ref = ref[ref[group_col] != case_gid]
    return ref.reset_index(drop=True)
