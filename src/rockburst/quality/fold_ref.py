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
    exclude_dup_group: bool = False,
) -> pd.DataFrame:
    """Training-fold reference excluding the evaluated case.

    By default keeps same-``dup_group_id`` neighbours so near-duplicate conflict
    checks can fire (RM3). Set ``exclude_dup_group=True`` only when a caller
    explicitly wants group-holdout (legacy OOD isolation).
    """
    ref = df.iloc[list(train_idx)].copy()
    case_no = case.get("no")
    if case_no is not None and "no" in ref.columns:
        ref = ref[ref["no"].astype(int) != int(case_no)]
    if exclude_dup_group:
        # Optional legacy path — not used for near-dup conflict (RM3).
        gid = case.get(group_col)
        if gid is not None and group_col in ref.columns:
            ref = ref[ref[group_col] != gid]
    return ref.reset_index(drop=True)
