"""Deterministic rock-mechanics relations (no LLM arithmetic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


FEATURE_COLS = [
    "sigma_theta",
    "sigma_c",
    "sigma_t",
    "scf",
    "b1",
    "b2",
    "wet",
    "d",
]


def compute_scf(sigma_theta: float, sigma_c: float) -> float:
    if sigma_c == 0 or not np.isfinite(sigma_c):
        return float("nan")
    return float(sigma_theta / sigma_c)


def compute_b1(sigma_c: float, sigma_t: float) -> float:
    if sigma_t == 0 or not np.isfinite(sigma_t):
        return float("nan")
    return float(sigma_c / sigma_t)


def compute_b2(sigma_c: float, sigma_t: float) -> float:
    denom = sigma_c + sigma_t
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((sigma_c - sigma_t) / denom)


def _rel_ok(observed: float, expected: float, rel_tol: float, abs_tol: float) -> bool:
    if not np.isfinite(observed) or not np.isfinite(expected):
        return False
    return abs(observed - expected) <= max(abs_tol, rel_tol * abs(expected))


@dataclass
class PhysicsCheckResult:
    r_physics: float
    exact_checks: dict[str, Any] = field(default_factory=dict)
    empirical_checks: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics_score": self.r_physics,
            "exact_checks": self.exact_checks,
            "empirical_checks": self.empirical_checks,
            "issues": self.issues,
        }


def check_case(
    row: dict[str, Any],
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
    scf_ood_threshold: float = 2.0,
    wet_range: tuple[float, float] = (0.5, 40.0),
    d_range: tuple[float, float] = (50.0, 2500.0),
) -> PhysicsCheckResult:
    """Verify exact derived relations and empirical ranges for one case."""
    issues: list[str] = []
    st = float(row["sigma_theta"])
    sc = float(row["sigma_c"])
    stens = float(row["sigma_t"])
    scf_obs = float(row["scf"]) if row.get("scf") is not None else float("nan")
    b1_obs = float(row["b1"]) if row.get("b1") is not None else float("nan")
    b2_obs = float(row["b2"]) if row.get("b2") is not None else float("nan")
    wet = row.get("wet")
    depth = row.get("d")

    scf_exp = compute_scf(st, sc)
    b1_exp = compute_b1(sc, stens)
    b2_exp = compute_b2(sc, stens)

    exact = {
        "scf": {
            "observed": scf_obs,
            "expected": scf_exp,
            "ok": _rel_ok(scf_obs, scf_exp, rel_tol, abs_tol),
            "relation": "exact: SCF = sigma_theta / sigma_c",
        },
        "b1": {
            "observed": b1_obs,
            "expected": b1_exp,
            "ok": _rel_ok(b1_obs, b1_exp, rel_tol, abs_tol),
            "relation": "exact: B1 = sigma_c / sigma_t",
        },
        "b2": {
            "observed": b2_obs,
            "expected": b2_exp,
            "ok": _rel_ok(b2_obs, b2_exp, rel_tol, abs_tol),
            "relation": "exact: B2 = (sigma_c - sigma_t) / (sigma_c + sigma_t)",
        },
    }
    for name, chk in exact.items():
        if not chk["ok"]:
            issues.append(
                f"{name} inconsistency: observed={chk['observed']:.4g}, "
                f"expected={chk['expected']:.4g}"
            )

    empirical: dict[str, Any] = {}
    if wet is None or (isinstance(wet, float) and not np.isfinite(wet)):
        empirical["wet"] = {"status": "missing", "relation": "empirical tendency"}
        issues.append("Wet missing")
    else:
        wet_f = float(wet)
        in_range = wet_range[0] <= wet_f <= wet_range[1]
        empirical["wet"] = {
            "value": wet_f,
            "in_range": in_range,
            "range": list(wet_range),
            "relation": "empirical tendency",
        }
        if not in_range:
            issues.append(f"Wet out of plausible range: {wet_f}")

    if depth is None or (isinstance(depth, float) and not np.isfinite(depth)):
        empirical["d"] = {"status": "missing", "relation": "empirical tendency"}
        # Missing D is common; mild penalty handled via score, not hard fail text spam
    else:
        d_f = float(depth)
        in_range = d_range[0] <= d_f <= d_range[1]
        empirical["d"] = {
            "value": d_f,
            "in_range": in_range,
            "range": list(d_range),
            "relation": "empirical tendency",
        }
        if not in_range:
            issues.append(f"D out of plausible range: {d_f}")

    if np.isfinite(scf_obs) and scf_obs > scf_ood_threshold:
        issues.append(f"SCF OOD (> {scf_ood_threshold}): {scf_obs:.3g}")
        empirical["scf_ood"] = True

    n_exact = sum(1 for c in exact.values() if c["ok"])
    exact_score = n_exact / 3.0
    emp_penalties = 0.0
    if empirical.get("wet", {}).get("in_range") is False:
        emp_penalties += 0.1
    if empirical.get("d", {}).get("in_range") is False:
        emp_penalties += 0.05
    if empirical.get("d", {}).get("status") == "missing":
        emp_penalties += 0.05
    if empirical.get("scf_ood"):
        emp_penalties += 0.15
    r_physics = float(np.clip(exact_score - emp_penalties, 0.0, 1.0))
    return PhysicsCheckResult(
        r_physics=r_physics,
        exact_checks=exact,
        empirical_checks=empirical,
        issues=issues,
    )
