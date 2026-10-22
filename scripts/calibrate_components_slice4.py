#!/usr/bin/env python3
"""Slice 4 / M4: fit per-channel R_* calibrators from saved OOF reliability matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rockburst.config import load_config, resolve_path
from rockburst.fusion.components import ComponentCalibrators


def main() -> None:
    cfg = load_config()
    models = resolve_path(cfg, "models")
    r_path = models / "fusion_R_oof.npy"
    y_path = models / "fusion_y_correct.npy"
    if not r_path.exists() or not y_path.exists():
        raise SystemExit("Need fusion_R_oof.npy and fusion_y_correct.npy (run train)")
    R = np.load(r_path)
    y = np.load(y_path)
    cal = ComponentCalibrators(
        conf_eps=float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
    ).fit(R, y)
    out = models / "component_calibrators.joblib"
    cal.save(out)
    report = cal.report()
    report_path = resolve_path(cfg, "baselines") / "component_calibration_slice4.json"
    report_path.write_text(json.dumps({"path": str(out), **report}, indent=2))
    print(json.dumps({"wrote": str(out), "methods": report["methods"]}, indent=2))


if __name__ == "__main__":
    main()
