"""Physics pre-check: skip geomechanics iff full check_case is perfect."""

from __future__ import annotations

from typing import Any

from rockburst.physics.relations import PhysicsCheckResult, check_case


def _physics_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    from rockburst.physics.relations import DEFAULT_ABS_TOL, DEFAULT_REL_TOL

    p = (cfg or {}).get("physics", {})
    return {
        "rel_tol": float(p.get("rel_tol", DEFAULT_REL_TOL)),
        "abs_tol": float(p.get("abs_tol", DEFAULT_ABS_TOL)),
        "scf_ood_threshold": float(p.get("scf_ood_threshold", 2.0)),
        "wet_range": tuple(p.get("wet_range", [0.5, 40.0])),
        "d_range": tuple(p.get("d_range", [50.0, 2500.0])),
        "b1_implausible_max": float(p.get("b1_implausible_max", 200.0)),
    }


def full_physics_result(
    case: dict[str, Any], cfg: dict[str, Any] | None = None
) -> PhysicsCheckResult:
    """Same scoring path as the geomechanics tool (`check_case`)."""
    return check_case(case, **_physics_kwargs(cfg))


def is_perfect_physics(res: PhysicsCheckResult) -> bool:
    """Skip full geomechanics when identities hold and empirical ranges are clean.

    Criteria disagreement may lower ``r_physics`` below 1.0 without forcing
    a re-run — criteria are already computed inside ``check_case``. Missing or
    out-of-range Wet/D still blocks skip (same as pre–Slice-3 behaviour).
    """
    if res.has_identity_violation or res.issues:
        return False
    emp = res.empirical_checks or {}
    wet = emp.get("wet") or {}
    depth = emp.get("d") or {}
    if wet.get("status") == "missing" or depth.get("status") == "missing":
        return False
    if wet.get("in_range") is False or depth.get("in_range") is False:
        return False
    if emp.get("scf_ood"):
        return False
    return True


def cheap_physics_precheck(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Skip decision via full tool score (no duplicated penalty logic)."""
    res = full_physics_result(case, cfg)
    clean = is_perfect_physics(res)
    return {
        "clean": clean,
        "r_physics": float(res.r_physics),
        "issues": list(res.issues),
        "checks": res.exact_checks,
        "reasons": list(res.issues),
    }


def should_skip_geomechanics(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    return is_perfect_physics(full_physics_result(case, cfg))
