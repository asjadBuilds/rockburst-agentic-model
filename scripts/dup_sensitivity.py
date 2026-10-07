#!/usr/bin/env python3
"""Near-duplicate grouping sensitivity at 0.5% / 1% / 2% relative tolerance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rockburst.config import load_config, resolve_path
from rockburst.data.audit import dup_sensitivity, process_raw


def main() -> None:
    cfg = load_config()
    raw = resolve_path(cfg, "raw_csv")
    if not raw.exists():
        raise SystemExit(f"Missing {raw}; run extract first")
    df = pd.read_csv(raw)
    sens = dup_sensitivity(df, tols=(0.005, 0.01, 0.02, 0.05))
    _, audit = process_raw(df, cfg)
    out = {
        **sens,
        "default_rel_tol": cfg.get("quality", {}).get("near_dup_rel_tol", 0.01),
        "default_audit": {
            "n_dup_groups": audit["n_dup_groups"],
            "n_groups_with_duplicates": audit["n_groups_with_duplicates"],
            "group_size_distribution": audit["group_size_distribution"],
            "n_conflicting_label_groups": audit["n_conflicting_label_groups"],
            "conflicting_label_pairs": audit["conflicting_label_pairs"],
            "label_noise_ceiling_note": audit.get("label_noise_ceiling_note"),
        },
    }
    dest = resolve_path(cfg, "baselines") / "dup_sensitivity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
