#!/usr/bin/env python3
"""Diff headline metrics between two result JSON trees. Prints differences only."""

from __future__ import annotations

import json
import sys
from pathlib import Path


KEYS = [
    ("m1_results.json", ["nested_unbiased", "pooled_oof", "qwk"]),
    ("m1_results.json", ["nested_unbiased", "pooled_oof", "accuracy"]),
    ("m1_results.json", ["fusion_r_final_ece"]),
    ("m1_results.json", ["fusion_mean_r_final"]),
    ("m1_results.json", ["fusion_coef"]),
    ("ablation_results.json", ["ablations", "M1", "qwk", "mean"]),
    ("ablation_results.json", ["ablations", "M7", "abstention_rate", "mean"]),
    ("ablation_results.json", ["agent_value_finding", "m5_m6_m7_qwk_identical"]),
    ("item4_metrics.json", ["risk_coverage", "R_final_beats_maxprob_area"]),
    ("item4_metrics.json", ["critic_error_detection", "overall_flag", "auroc"]),
]


def _get(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: diff_headline_metrics.py DIR_A DIR_B")
        sys.exit(2)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    diffs = 0
    for fname, path in KEYS:
        da = json.loads((a / fname).read_text())
        db = json.loads((b / fname).read_text())
        va, vb = _get(da, path), _get(db, path)
        if va != vb:
            diffs += 1
            print(f"DIFF {fname} {'/'.join(path)}: {va!r} vs {vb!r}")
    if diffs == 0:
        print("no differences in headline keys")
    else:
        print(f"{diffs} differences")


if __name__ == "__main__":
    main()
