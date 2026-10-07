"""Physics pre-check: skip geomechanics iff full check_case is perfect."""

from __future__ import annotations

from typing import Any

from rockburst.physics.relations import PhysicsCheckResult, check_case


def _physics_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    p = (cfg or {}).get("physics", {})
    return {
        "rel_tol": float(p.get("rel_tol", 0.05)),
        "abs_tol": float(p.get("abs_tol", 0.05)),
        "scf_ood_threshold": float(p.get("scf_ood_threshold", 2.0)),
        "wet_range": tuple(p.get("wet_range", [0.5, 40.0])),
        "d_range": tuple(p.get("d_range", [50.0, 2500.0])),
    }


def full_physics_result(
    case: dict[str, Any], cfg: dict[str, Any] | None = None
) -> PhysicsCheckResult:
    """Same scoring path as the geomechanics tool (`check_case`)."""
    return check_case(case, **_physics_kwargs(cfg))


def is_perfect_physics(res: PhysicsCheckResult) -> bool:
    return float(res.r_physics) == 1.0 and len(res.issues) == 0


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
