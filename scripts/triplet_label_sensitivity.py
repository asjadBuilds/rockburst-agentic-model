#!/usr/bin/env python3
"""Phase 5 / RM12–RM13: production M1+M7 under triplet groups and label policies.

RM12: Headline M1 and M7 with (σc, σt, Wet) as StratifiedGroupKFold group key,
      reported next to the main (dup_group_id) headlines from ablation_results.

RM13: Production M1 + full M7 under three label policies; missing-D vs grade
      association with/without rows 315–344.

Requires a prior full ``rockburst train`` (reads ``m1_results.json``).
Agent must not run this in full mode — user Phase 5 job.

Usage:
  .venv/bin/python scripts/triplet_label_sensitivity.py          # full
  .venv/bin/python scripts/triplet_label_sensitivity.py --quick  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rockburst.config import load_config, resolve_path
from rockburst.eval.run_meta import build_run_metadata, finalize_run_metadata
from rockburst.eval.sensitivity import (
    assign_triplet_group_ids,
    evaluate_m1_m7,
    label_policy_frames,
    missing_d_vs_grade,
    triplet_inventory,
)
from rockburst.ml.pipelines import prepare_xy


def _load_m1(cfg: dict) -> dict:
    path = resolve_path(cfg, "baselines") / "m1_results.json"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}; run full `.venv/bin/python -m rockburst.cli train` first."
        )
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="2 folds / 1 repeat smoke (not for delivery artefacts)",
    )
    ap.add_argument(
        "--skip-m7-policies",
        action="store_true",
        help="RM13: M1 only under policies (faster debug; not delivery)",
    )
    args = ap.parse_args()

    cfg = load_config()
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        print(f"Missing {path}; run rockburst extract", file=sys.stderr)
        return 1
    df = pd.read_csv(path)
    m1_meta = _load_m1(cfg)
    rel_tol = float(cfg.get("quality", {}).get("near_dup_rel_tol", 0.05))
    mode = "quick" if args.quick else "full"
    cmd = "scripts/triplet_label_sensitivity.py" + (" --quick" if args.quick else "")
    started_metadata = build_run_metadata(
        mode=mode,
        command=cmd,
        config=cfg,
        input_paths=[path],
    )

    # --- RM12: triplet-grouped CV headlines ---
    _X, _y, dup_groups = prepare_xy(df)
    triplet_groups = assign_triplet_group_ids(df)
    print("RM12: evaluating M1+M7 under triplet groups…", flush=True)
    rm12_triplet = evaluate_m1_m7(
        df, triplet_groups, cfg, m1_meta, quick=args.quick
    )
    print("RM12: evaluating M1+M7 under dup_group_id (side-by-side)…", flush=True)
    rm12_dup = evaluate_m1_m7(df, dup_groups, cfg, m1_meta, quick=args.quick)
    print("RM12: evaluating observed-indicator sensitivity…", flush=True)
    no_indicator = evaluate_m1_m7(
        df,
        dup_groups,
        cfg,
        m1_meta,
        quick=args.quick,
        include_observed_indicators=False,
    )

    # --- RM13: three label policies ---
    policies = label_policy_frames(df, rel_tol=rel_tol)
    rm13: dict = {}
    for name, sub in policies.items():
        print(f"RM13: policy={name} n={len(sub)}…", flush=True)
        _Xs, _ys, g_sub = prepare_xy(sub)
        if args.skip_m7_policies:
            # M1-only shortcut: still use evaluate but caller shouldn't ship this
            blob = evaluate_m1_m7(sub, g_sub, cfg, m1_meta, quick=True)
            # Strip M7 to make the shortcut obvious
            if "systems" in blob:
                blob["systems"].pop("M7", None)
            blob["note"] = "M7 skipped (--skip-m7-policies); not for delivery"
        else:
            blob = evaluate_m1_m7(sub, g_sub, cfg, m1_meta, quick=args.quick)
        rm13[name] = {
            "n": int(len(sub)),
            "n_dropped_vs_as_given": int(len(df) - len(sub)),
            "missing_d_vs_grade": missing_d_vs_grade(sub),
            "m1_m7": blob,
        }

    drop_mask = ~((df["no"].astype(int) >= 315) & (df["no"].astype(int) <= 344)) if "no" in df.columns else pd.Series(True, index=df.index)
    missing_cmp = {
        "as_given": missing_d_vs_grade(df),
        "without_315_344": missing_d_vs_grade(df.loc[drop_mask]),
        "finding": (
            "After dropping rows 315–344, compare mean_grade with_d vs missing_d. "
            "Round-2 noted the association disappears (≈1.54 vs 1.55); report plainly."
        ),
    }

    out_blob = {
        "run_metadata": finalize_run_metadata(
            started_metadata,
            mode=mode,
            command=cmd,
            extra={
                "production_model": m1_meta.get("nested_unbiased", {}).get(
                    "production_model"
                )
                or m1_meta.get("best_model"),
            },
            config=cfg,
            input_paths=[path],
        ),
        "rm12_triplet_grouping": {
            "inventory": triplet_inventory(df),
            "group_key": "(sigma_c, sigma_t, wet)",
            "headline_triplet_groups": rm12_triplet,
            "headline_dup_groups": rm12_dup,
            "note": (
                "Report triplet-grouped M1/M7 next to main dup_group_id headlines "
                "(ablation_results.json)."
            ),
        },
        "rm13_label_policies": rm13,
        "missing_d_comparison": missing_cmp,
        "observed_indicator_sensitivity": {
            "with_indicators": rm12_dup,
            "without_indicators": no_indicator,
            "note": (
                "Same production family, folds, and labels; only observed_* "
                "input indicators differ."
            ),
        },
        "replaces": "artifacts/baselines/slice6_m13_m14.json (untuned logistic probe)",
    }

    out = resolve_path(cfg, "baselines") / "triplet_label_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_blob, indent=2, default=str))
    summary = {
        "wrote": str(out),
        "mode": mode,
        "rm12_triplet_M1_qwk": (rm12_triplet.get("systems") or {})
        .get("M1", {})
        .get("qwk"),
        "rm12_triplet_M7_qwk": (rm12_triplet.get("systems") or {})
        .get("M7", {})
        .get("qwk"),
        "rm12_dup_M1_qwk": (rm12_dup.get("systems") or {}).get("M1", {}).get("qwk"),
        "policies": {k: v["n"] for k, v in rm13.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
