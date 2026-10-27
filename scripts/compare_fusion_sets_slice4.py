#!/usr/bin/env python3
"""RM7: recompare fusion feature sets A–D on the final leakage-safe OOF R matrix.

Uses full train-portion OOF calibration (RC3) and reports per-component AUROC +
bootstrap CIs. Phase 5 user job after ``train`` refreshes ``fusion_R_oof.npy``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rockburst.config import load_config, resolve_path
from rockburst.eval.run_meta import build_run_metadata
from rockburst.fusion.set_compare import compare_fusion_sets
from rockburst.ml.pipelines import prepare_xy


def main() -> None:
    cfg = load_config()
    models = resolve_path(cfg, "models")
    r_path = models / "fusion_R_oof.npy"
    y_path = models / "fusion_y_correct.npy"
    if not r_path.exists() or not y_path.exists():
        raise SystemExit(
            f"Missing {r_path.name} / {y_path.name}; run full train first (Phase 5)."
        )
    R = np.load(r_path)
    y_correct = np.load(y_path)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    _X, y, groups = prepare_xy(df)
    if len(y) != len(y_correct):
        raise SystemExit(
            f"Length mismatch: R/y_correct n={len(y_correct)} vs processed n={len(y)}"
        )
    seed = int(cfg.get("seed", 42))
    n_outer = int(cfg["cv"]["n_outer"])
    conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))

    out = compare_fusion_sets(
        R,
        y_correct,
        y,
        groups,
        seed=seed,
        n_outer=n_outer,
        n_boot=200,
        conf_eps=conf_eps,
    )
    out["note"] = (
        "Leakage-safe OOF R from train; calibrator uses full train-portion OOF "
        "raw scores (RC3). Keep set A if it still wins on RC area (RM7)."
    )
    out["run_metadata"] = build_run_metadata(
        mode="full",
        command="scripts/compare_fusion_sets_slice4.py",
        extra={"n_outer": n_outer, "chosen_set": out.get("chosen_set")},
    )
    path = resolve_path(cfg, "baselines") / "fusion_set_comparison_slice4.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps(
            {
                "wrote": str(path),
                "chosen": out["chosen_set"],
                "finding": out["finding"],
                "component_auroc": {
                    k: v.get("auroc")
                    for k, v in (out.get("component_auroc") or {})
                    .get("per_component", {})
                    .items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
