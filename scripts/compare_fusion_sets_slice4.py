#!/usr/bin/env python3
"""Slice 4 / M4: recompare fusion feature sets A–D after M1/M2/C3 on saved OOF R.

Uses the same pre-registered selection rule as Item 9 (simplest tied with best).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from rockburst.config import load_config, resolve_path
from rockburst.eval.metrics import expected_calibration_error
from rockburst.fusion.components import (
    FUSION_SET_CANDIDATES,
    SIMPLICITY_ORDER,
    mask_for_set,
)
from rockburst.fusion.stacker import (
    CHOSEN_FUSION_SET,
    ReliabilityFusion,
    bootstrap_coef_cis,
    select_decision_thresholds,
)
from rockburst.ml.pipelines import prepare_xy
from rockburst.ml.train import _risk_coverage_curve, _split_fit_calib


def _crossfit_scores(
    R: np.ndarray,
    y_correct: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    seed: int,
    n_outer: int,
) -> tuple[np.ndarray, np.ndarray]:
    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(y), -1, dtype=int)
    for fold, (_tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_of_row[te] = fold
    r_final = np.zeros(len(y), dtype=float)
    for fold in range(n_outer):
        train_mask = (fold_of_row >= 0) & (fold_of_row != fold)
        te_mask = fold_of_row == fold
        if not train_mask.any() or not te_mask.any():
            continue
        idx_fit = np.where(train_mask)[0]
        idx_te = np.where(te_mask)[0]
        _inner, inner_val = _split_fit_calib(y, groups, idx_fit, seed=seed + 100 + fold)
        fus = ReliabilityFusion().fit(
            R[idx_fit], y_correct[idx_fit], feature_mask=mask, select_features=False
        )
        if not fus.is_fit:
            continue
        r_raw = np.array([fus._raw_score(R[i]) for i in idx_fit])
        val_pos = np.isin(idx_fit, inner_val)
        if val_pos.any():
            fus.fit_calibrator(r_raw[val_pos], y_correct[idx_fit][val_pos])
            thr = select_decision_thresholds(
                np.array([fus.r_final(R[i]) for i in idx_fit[val_pos]]),
                y_correct[idx_fit][val_pos],
            )
            fus.set_decision_thresholds(thr)
        for j in idx_te:
            r_final[j] = fus.r_final(R[j])
    return r_final, fold_of_row


def main() -> None:
    cfg = load_config()
    models = resolve_path(cfg, "models")
    R = np.load(models / "fusion_R_oof.npy")
    y_correct = np.load(models / "fusion_y_correct.npy")
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    _X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    n_outer = int(cfg["cv"]["n_outer"])

    set_scores: dict[str, np.ndarray] = {}
    fold_of = None
    rows: dict = {}
    for set_id in SIMPLICITY_ORDER:
        mask = mask_for_set(set_id)
        r_final, fold_of = _crossfit_scores(
            R, y_correct, y, groups, mask, seed, n_outer
        )
        set_scores[set_id] = r_final
        held = fold_of >= 0
        rc = _risk_coverage_curve(r_final[held], y_correct[held])
        ece = expected_calibration_error(
            y_correct[held],
            np.column_stack([1 - r_final[held], r_final[held]]),
        )
        rows[set_id] = {
            "features": list(FUSION_SET_CANDIDATES[set_id]),
            "area": rc.get("area"),
            "ece": ece,
            "brier": float(brier_score_loss(y_correct[held], r_final[held])),
            "mean_r_final": float(r_final[held].mean()),
        }

    best_id = max(rows, key=lambda k: rows[k]["area"] or -1)
    rng = np.random.default_rng(seed)
    held_idx = np.where(fold_of >= 0)[0]
    deltas: dict[str, list[float]] = {s: [] for s in SIMPLICITY_ORDER}
    for _ in range(200):
        boot = rng.choice(held_idx, size=len(held_idx), replace=True)
        a_best = _risk_coverage_curve(
            set_scores[best_id][boot], y_correct[boot]
        )["area"]
        for s in SIMPLICITY_ORDER:
            a_s = _risk_coverage_curve(set_scores[s][boot], y_correct[boot])["area"]
            if a_best is not None and a_s is not None:
                deltas[s].append(float(a_s - a_best))

    tied = []
    for s in SIMPLICITY_ORDER:
        d = deltas[s]
        lo, hi = (
            (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
            if d
            else (0.0, 0.0)
        )
        rows[s]["delta_vs_best_ci"] = [lo, hi]
        if lo <= 0.0 <= hi:
            tied.append(s)
    chosen = best_id
    for s in SIMPLICITY_ORDER:
        if s in tied:
            chosen = s
            break

    mask_prod = mask_for_set(chosen)
    _inner, _ = _split_fit_calib(y, groups, np.arange(len(y)), seed=seed + 99)
    coef_cis = bootstrap_coef_cis(
        R[_inner], y_correct[_inner], n_boot=200, seed=seed, feature_mask=mask_prod
    )

    out = {
        "sets": rows,
        "best_by_area": best_id,
        "tied_with_best": tied,
        "chosen_set": chosen,
        "locked_production_set": CHOSEN_FUSION_SET,
        "selection_rule": (
            "Among sets whose Δ-area CI vs best contains 0, choose simplest "
            f"(order {list(SIMPLICITY_ORDER)})."
        ),
        "agents_add_calibrated_reliability_signal": chosen != "A",
        "finding": (
            "Still set A (r_model only): agent R_* channels add no calibrated "
            "ranking signal under the pre-registered rule."
            if chosen == "A"
            else f"Chosen set {chosen}; multi-channel reliability contributes."
        ),
        "coef_bootstrap_cis": coef_cis,
        "note": (
            "OOF R matrix may predate Slice 3 physics changes; full train refreshes R."
        ),
    }
    path = resolve_path(cfg, "baselines") / "fusion_set_comparison_slice4.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"wrote": str(path), "chosen": chosen, "finding": out["finding"]}, indent=2))


if __name__ == "__main__":
    main()
