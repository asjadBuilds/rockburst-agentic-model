#!/usr/bin/env python3
"""RC8: archive non-final / superseded baseline artefacts and Oct-11 LLM traces.

Moves (does not delete) known stale files under artifacts/archive/ with a manifest.
Safe to re-run; skips already-archived paths.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rockburst.config import load_config, resolve_path

# Non-final baselines superseded by Phase 5 full-mode artefacts / Round-2 fusion path.
STALE_BASELINES = (
    "item4_metrics.json",
    "item4_case_vectors.npz",
    "item5_dup_label_noise.json",
    "item5_last_oof.npz",
    "fusion_item2_refit.json",
    "fusion_missing_d_drop.json",
    "fusion_feature_set_compare.txt",
    "fusion_slice2_recalibrate.json",
    "component_calibration_slice4.json",
    "threshold_sensitivity.json",
    "dup_sensitivity.json",
    "slice6_m13_m14.json",  # replaced by triplet_label_sensitivity.json
)

# Keep: m1_results, ablation_results, m2_importance, headline_snapshot,
# fusion_set_comparison_slice4, triplet_label_sensitivity, llm_* STALE pointer.


def _archive_file(src: Path, dest_dir: Path, moved: list[dict]) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        # Already archived; remove live copy if still present
        if src.resolve() != dest.resolve():
            src.unlink()
            moved.append({"path": str(src), "action": "removed_duplicate_live", "archived": str(dest)})
        return
    shutil.move(str(src), str(dest))
    moved.append({"path": str(src), "action": "moved", "archived": str(dest)})


def archive_llm_traces(traces: Path, archive_root: Path, moved: list[dict]) -> None:
    r"""Archive *_llm.json traces and any Oct-11 rejected:recheck_* style dumps."""
    dest = archive_root / "traces_llm"
    for p in sorted(traces.glob("*_llm.json")):
        _archive_file(p, dest, moved)
    # Legacy rejected:recheck filenames if any
    for p in sorted(traces.glob("rejected*")):
        _archive_file(p, dest, moved)


def main() -> int:
    cfg = load_config()
    root = resolve_path(cfg, "artifacts")
    archive_root = root / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict] = []

    base = resolve_path(cfg, "baselines")
    for name in STALE_BASELINES:
        _archive_file(base / name, archive_root / "baselines", moved)

    traces = root / "traces"
    if traces.is_dir():
        archive_llm_traces(traces, archive_root, moved)

    # Stale example stubs without case_no / true_grade (pre-RC8)
    ex = resolve_path(cfg, "examples")
    for name in ("case_1.json", "case_329.json"):
        p = ex / name
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if blob.get("case_no") is None and blob.get("true_grade") is None:
            _archive_file(p, archive_root / "examples_stale", moved)

    manifest = {
        "archived_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_actions": len(moved),
        "actions": moved,
        "note": (
            "RC8 archive of non-final artefacts. Final baselines remain under "
            "artifacts/baselines/; regenerate examples/traces via "
            "scripts/generate_examples.py."
        ),
    }
    (archive_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"archived {len(moved)} path(s) → {archive_root}")
    for a in moved[:20]:
        print(" ", a["action"], a["path"])
    if len(moved) > 20:
        print(f"  … +{len(moved) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
