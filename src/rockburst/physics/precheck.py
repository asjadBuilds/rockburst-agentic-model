"""Cheap deterministic physics pre-check (skip decision for geomechanics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.physics.relations import (
    _rel_ok,
    compute_b1,
    compute_b2,
    compute_scf,
)


def cheap_physics_precheck(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return whether full geomechanics can be skipped.

    Skip only when SCF/B1/B2 agree within tolerance and Wet and D are in range.
    Missing D fails the pre-check (do not skip).
    """
    cfg = cfg or {}
    p = cfg.get("physics", {})
    rel_tol = float(p.get("rel_tol", 0.05))
    abs_tol = float(p.get("abs_tol", 0.05))
    wet_range = tuple(p.get("wet_range", [0.5, 40.0]))
    d_range = tuple(p.get("d_range", [50.0, 2500.0]))

    reasons: list[str] = []
    st = float(case["sigma_theta"])
    sc = float(case["sigma_c"])
    stens = float(case["sigma_t"])
    scf_obs = float(case["scf"]) if case.get("scf") is not None else float("nan")
    b1_obs = float(case["b1"]) if case.get("b1") is not None else float("nan")
    b2_obs = float(case["b2"]) if case.get("b2") is not None else float("nan")

    checks = {
        "scf": _rel_ok(scf_obs, compute_scf(st, sc), rel_tol, abs_tol),
        "b1": _rel_ok(b1_obs, compute_b1(sc, stens), rel_tol, abs_tol),
        "b2": _rel_ok(b2_obs, compute_b2(sc, stens), rel_tol, abs_tol),
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name}_mismatch")

    wet = case.get("wet")
    if wet is None or (isinstance(wet, float) and not np.isfinite(wet)):
        checks["wet_in_range"] = False
        reasons.append("wet_missing")
    else:
        wet_f = float(wet)
        checks["wet_in_range"] = wet_range[0] <= wet_f <= wet_range[1]
        if not checks["wet_in_range"]:
            reasons.append("wet_oor")

    depth = case.get("d")
    if depth is None or (isinstance(depth, float) and not np.isfinite(depth)):
        checks["d_in_range"] = False
        reasons.append("d_missing")
    else:
        d_f = float(depth)
        checks["d_in_range"] = d_range[0] <= d_f <= d_range[1]
        if not checks["d_in_range"]:
            reasons.append("d_oor")

    clean = all(checks.values())
    return {"clean": clean, "checks": checks, "reasons": reasons}


def should_skip_geomechanics(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    return bool(cheap_physics_precheck(case, cfg)["clean"])
