"""Dataset audit: physics errors, OOD, duplicates, group ids."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from rockburst.physics.relations import FEATURE_COLS, check_case
from rockburst.quality.near_dup import (
    DEFAULT_REL_TOL,
    RAW_DUP_COLS,
    assign_component_ids,
    connected_components,
    pairwise_near_dup_mask,
)


def assign_dup_groups(
    df: pd.DataFrame,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    cols: tuple[str, ...] = RAW_DUP_COLS,
) -> pd.DataFrame:
    """Assign ``dup_group_id`` via relative-tolerance connected components.

    Matching uses raw measured inputs only (σθ, σc, σt, Wet, D when present).
    Pairs with relative difference ``< rel_tol`` on every overlapping finite
    feature are edges; groups are connected components. Label conflicts are
    retained (never silently resolved).
    """
    out = df.copy()
    # Deterministic row order by case number
    order = np.argsort(out["no"].to_numpy(dtype=int), kind="mergesort")
    ordered = out.iloc[order].reset_index(drop=True)
    feats = ordered[list(cols)].to_numpy(dtype=float)
    nos = ordered["no"].to_numpy(dtype=int)
    adj = pairwise_near_dup_mask(feats, rel_tol=rel_tol)
    comps = connected_components(adj)
    gids = assign_component_ids(comps, order_keys=nos)
    # Map back to original row positions
    inv = np.empty(len(out), dtype=int)
    inv[order] = np.arange(len(out))
    out["dup_group_id"] = gids[inv]
    return out


def _dup_group_stats(df: pd.DataFrame, target: str = "rockburst_grade") -> dict[str, Any]:
    sizes = df.groupby("dup_group_id").size()
    size_dist = Counter(int(s) for s in sizes.tolist())
    multi = sizes[sizes > 1]
    conflicting_pairs: list[dict[str, Any]] = []
    n_conflict_groups = 0
    for gid, gdf in df.groupby("dup_group_id"):
        if len(gdf) < 2:
            continue
        grades = gdf[target].astype(int)
        if grades.nunique() <= 1:
            continue
        n_conflict_groups += 1
        rows = gdf[["no", target]].sort_values("no")
        nos = rows["no"].astype(int).tolist()
        grs = rows[target].astype(int).tolist()
        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                if grs[i] != grs[j]:
                    conflicting_pairs.append(
                        {
                            "dup_group_id": int(gid),
                            "no_a": nos[i],
                            "grade_a": grs[i],
                            "no_b": nos[j],
                            "grade_b": grs[j],
                        }
                    )
    return {
        "n_dup_groups": int(df["dup_group_id"].nunique()),
        "n_groups_with_duplicates": int(len(multi)),
        "group_size_distribution": {str(k): int(v) for k, v in sorted(size_dist.items())},
        "n_conflicting_label_groups": n_conflict_groups,
        "conflicting_label_pairs": conflicting_pairs,
        "label_noise_ceiling_note": (
            "Conflicting labels are kept (not silently resolved). "
            "Groups stay in one fold via StratifiedGroupKFold. "
            f"Conflicting pairs imply a label-noise ceiling on achievable accuracy "
            f"of at least {len(conflicting_pairs)} disagreeing row-pairs."
        ),
    }


def audit_dataframe(df: pd.DataFrame, physics_cfg: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for _, row in df.iterrows():
        res = check_case(
            row.to_dict(),
            rel_tol=physics_cfg.get("rel_tol", 0.05),
            abs_tol=physics_cfg.get("abs_tol", 0.05),
            scf_ood_threshold=physics_cfg.get("scf_ood_threshold", 2.0),
            wet_range=tuple(physics_cfg.get("wet_range", [0.5, 40.0])),
            d_range=tuple(physics_cfg.get("d_range", [50.0, 2500.0])),
        )
        if res.issues:
            issues.append({"no": int(row["no"]), "issues": res.issues, "r_physics": res.r_physics})

    known = {
        "bad_b1_rows": [20, 21, 244],
        "ood_scf_block": list(range(146, 158)),
    }
    dup_stats = (
        _dup_group_stats(df)
        if "dup_group_id" in df.columns
        else {
            "n_dup_groups": None,
            "n_groups_with_duplicates": None,
            "group_size_distribution": {},
            "n_conflicting_label_groups": 0,
            "conflicting_label_pairs": [],
        }
    )
    return {
        "n_rows": len(df),
        "d_missing": int(df["d"].isna().sum()),
        "grade_counts": df["rockburst_grade"].value_counts().sort_index().to_dict(),
        **dup_stats,
        "n_cases_with_physics_issues": len(issues),
        "physics_issue_examples": issues[:15],
        "known_problem_rows": known,
        "near_dup_features": list(RAW_DUP_COLS),
    }


def process_raw(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    q = cfg.get("quality", {})
    rel_tol = float(q.get("near_dup_rel_tol", DEFAULT_REL_TOL))
    processed = assign_dup_groups(df, rel_tol=rel_tol)
    audit = audit_dataframe(processed, cfg.get("physics", {}))
    audit["near_dup_rel_tol"] = rel_tol
    return processed, audit


def dup_sensitivity(
    df: pd.DataFrame,
    tols: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict[str, Any]:
    """Group counts / sizes at several relative tolerances."""
    rows = []
    for tol in tols:
        gdf = assign_dup_groups(df, rel_tol=tol)
        stats = _dup_group_stats(gdf)
        rows.append(
            {
                "rel_tol": tol,
                "n_dup_groups": stats["n_dup_groups"],
                "n_groups_with_duplicates": stats["n_groups_with_duplicates"],
                "group_size_distribution": stats["group_size_distribution"],
                "n_conflicting_label_groups": stats["n_conflicting_label_groups"],
                "n_conflicting_pairs": len(stats["conflicting_label_pairs"]),
            }
        )
    return {"tolerances": rows, "justification": (
        "Default near_dup_rel_tol=0.01 (1%): Table S1 near-copies differ by "
        "~0.1–0.2% on σθ; 1% catches that block while sensitivity at 0.5%/2% "
        "shows grouping stability."
    )}
