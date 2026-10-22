#!/usr/bin/env python3
"""Slice 6 / M13–M14: triplet grouping + label-policy headline metrics.

M13: Sensitivity grouped by identical (σc, σt, Wet) triplets (source/rock series).
M14: Headline M1-style QWK under (a) as-given (b) drop rows 315–344
     (c) earliest-of-conflict; missing-D vs grade with/without that block.

Writes artifacts/baselines/slice6_m13_m14.json. Uses a fixed logistic baseline
(no Optuna) so the script finishes in minutes; full production numbers come
from `rockburst train` after Slice 6 modelling changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import classification_report_dict, quadratic_weighted_kappa
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.quality.near_dup import RAW_DUP_COLS, rows_near_duplicate


def _triplet_key(row: pd.Series) -> tuple:
    def _r(v, nd=4):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return round(float(v), nd)

    return (_r(row.get("sigma_c")), _r(row.get("sigma_t")), _r(row.get("wet")))


def triplet_sensitivity(df: pd.DataFrame) -> dict:
    groups: dict[tuple, list[int]] = {}
    for i, row in df.iterrows():
        key = _triplet_key(row)
        if None in key:
            continue
        groups.setdefault(key, []).append(int(row.get("no", i)))

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    grade_var = []
    conflicting = 0
    for key, nos in multi.items():
        sub = df[df["no"].isin(nos)] if "no" in df.columns else df.loc[nos]
        grades = sub["rockburst_grade"].astype(int).tolist()
        uniq = sorted(set(grades))
        if len(uniq) > 1:
            conflicting += 1
        grade_var.append(
            {
                "triplet": {"sigma_c": key[0], "sigma_t": key[1], "wet": key[2]},
                "n": len(nos),
                "nos": nos[:20],
                "grades": grades,
                "grade_span": int(max(grades) - min(grades)) if grades else 0,
            }
        )
    grade_var.sort(key=lambda r: (-r["grade_span"], -r["n"]))
    return {
        "n_triplets": len(groups),
        "n_multi_row_triplets": len(multi),
        "n_conflicting_grade_triplets": conflicting,
        "top_span_examples": grade_var[:15],
    }


def earliest_of_conflict_mask(df: pd.DataFrame, rel_tol: float = 0.05) -> np.ndarray:
    """Keep earliest row in each conflicting near-dup cluster; drop later conflicts."""
    keep = np.ones(len(df), dtype=bool)
    if "no" not in df.columns:
        return keep
    nos = df["no"].astype(int).to_numpy()
    grades = df["rockburst_grade"].astype(int).to_numpy()
    mat = df[list(RAW_DUP_COLS)].to_numpy(dtype=float)
    order = np.argsort(nos)
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1 :]:
            if not keep[j]:
                continue
            if rows_near_duplicate(mat[i], mat[j], rel_tol=rel_tol):
                if grades[i] != grades[j]:
                    keep[j] = False  # drop later conflicting near-dup
    return keep


def _quick_oof_qwk(df: pd.DataFrame, *, seed: int = 42, n_outer: int = 5) -> dict:
    if len(df) < 20:
        return {"n": int(len(df)), "qwk": None, "accuracy": None, "error": "too_few_rows"}
    X, y, groups = prepare_xy(df)
    n_unique = len(np.unique(groups))
    n_sp = min(n_outer, max(2, n_unique // 2))
    sgkf = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=seed)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), 4), dtype=float)
    for tr, te in sgkf.split(X, y, groups):
        pipe = Pipeline(
            [
                ("prep", build_preprocess("median")),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                ),
            ]
        )
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])
        if proba.shape[1] < 4:
            pad = np.zeros((proba.shape[0], 4))
            pad[:, : proba.shape[1]] = proba
            proba = pad
        oof_proba[te] = proba
        oof_pred[te] = proba.argmax(axis=1)
    mask = oof_pred >= 0
    report = classification_report_dict(y[mask], oof_pred[mask], oof_proba[mask])
    return {
        "n": int(mask.sum()),
        "qwk": report.get("qwk"),
        "accuracy": report.get("accuracy"),
        "ordinal_mae": report.get("ordinal_mae"),
        "severe_misclassification_rate": report.get("severe_misclassification_rate"),
    }


def missing_d_vs_grade(df: pd.DataFrame) -> dict:
    has_d = df["d"].notna() if "d" in df.columns else pd.Series(False, index=df.index)
    g = df["rockburst_grade"].astype(int)
    out = {}
    for label, mask in (("with_d", has_d), ("missing_d", ~has_d)):
        sub = g[mask]
        counts = {str(int(k)): int(v) for k, v in sub.value_counts().sort_index().items()}
        out[label] = {
            "n": int(mask.sum()),
            "grade_counts": counts,
            "mean_grade": float(sub.mean()) if len(sub) else None,
        }
    return out


def main() -> int:
    cfg = load_config()
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        print(f"Missing {path}; run rockburst extract", file=sys.stderr)
        return 1
    df = pd.read_csv(path)
    rel_tol = float(cfg.get("quality", {}).get("near_dup_rel_tol", 0.05))

    # Row block 315–344 (1-based source nos if present; else positional)
    if "no" in df.columns:
        drop_block = (df["no"].astype(int) >= 315) & (df["no"].astype(int) <= 344)
    else:
        drop_block = pd.Series(False, index=df.index)
        drop_block.iloc[314:344] = True

    policies = {
        "as_given": df,
        "drop_rows_315_344": df.loc[~drop_block].copy(),
        "earliest_of_conflict": df.loc[earliest_of_conflict_mask(df, rel_tol=rel_tol)].copy(),
    }

    m14 = {}
    for name, sub in policies.items():
        m14[name] = {
            "n": int(len(sub)),
            "n_dropped_vs_as_given": int(len(df) - len(sub)),
            "headline_logistic_oof": _quick_oof_qwk(sub, seed=int(cfg.get("seed", 42))),
            "missing_d_vs_grade": missing_d_vs_grade(sub),
        }

    # Missing-D vs grade with/without 315–344 block on as-given
    m14["missing_d_comparison"] = {
        "as_given": missing_d_vs_grade(df),
        "without_315_344": missing_d_vs_grade(df.loc[~drop_block]),
    }

    blob = {
        "m13_triplet_sensitivity": triplet_sensitivity(df),
        "m14_label_policies": m14,
        "note": (
            "Headline QWK here uses untuned logistic (fast probe). "
            "Re-run `rockburst train` for production model×imputer under each policy if needed."
        ),
    }
    out = resolve_path(cfg, "baselines") / "slice6_m13_m14.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2))
    print(json.dumps({"wrote": str(out), "m13_multi": blob["m13_triplet_sensitivity"]["n_multi_row_triplets"], "policies": {k: v["n"] for k, v in m14.items() if k != "missing_d_comparison"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
