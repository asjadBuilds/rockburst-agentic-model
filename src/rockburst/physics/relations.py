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

# Primary measured inputs; SCF/B1/B2 are derived identities.
PRIMARY_FEATURES = ["sigma_theta", "sigma_c", "sigma_t", "wet", "d"]
DERIVED_FEATURES = ["scf", "b1", "b2"]

# Source-table rounding: absolute ~0.006 or 1–2% relative (Slice 3 / M2).
DEFAULT_REL_TOL = 0.02
DEFAULT_ABS_TOL = 0.006

# Flag implausible σc/σt (e.g. row 288 listed B1≈23 but σc/σt≈602).
DEFAULT_B1_IMPLAUSIBLE_MAX = 200.0


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


def with_recomputed_derived(case: dict[str, Any]) -> dict[str, Any]:
    """Policy (b): copy case with SCF/B1/B2 recomputed from primary stresses."""
    out = dict(case)
    st, sc, stens = out.get("sigma_theta"), out.get("sigma_c"), out.get("sigma_t")
    if (
        st is not None
        and sc is not None
        and stens is not None
        and np.isfinite(float(st))
        and np.isfinite(float(sc))
        and np.isfinite(float(stens))
    ):
        out["scf"] = compute_scf(float(st), float(sc))
        out["b1"] = compute_b1(float(sc), float(stens))
        out["b2"] = compute_b2(float(sc), float(stens))
    return out


def with_as_given(case: dict[str, Any]) -> dict[str, Any]:
    """Policy (a): shallow copy keeping listed SCF/B1/B2."""
    return dict(case)


# Physically consistent missingness masks (Slice 5 / C7).
PHYSICAL_MISSING_MASKS: dict[str, tuple[str, ...]] = {
    "sigma_theta_scf": ("sigma_theta", "scf"),
    "sigma_c_scf_b1_b2": ("sigma_c", "scf", "b1", "b2"),
    "sigma_t_b1_b2": ("sigma_t", "b1", "b2"),
    "wet": ("wet",),
    "d": ("d",),
}


def _finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def reconstruct_missing_from_physics(case: dict[str, Any]) -> dict[str, Any]:
    """Fill missing primaries/derived via rock-mechanics identities when possible.

    Examples: σθ = SCF·σc when only σθ missing; SCF = σθ/σc; B1/B2 from σc,σt.
    Does not invent values when both sides of an identity are missing.
    """
    out = dict(case)
    st, sc, stens = out.get("sigma_theta"), out.get("sigma_c"), out.get("sigma_t")
    scf, b1, b2 = out.get("scf"), out.get("b1"), out.get("b2")

    # Reconstruct σθ from SCF·σc
    if not _finite(st) and _finite(scf) and _finite(sc):
        out["sigma_theta"] = float(scf) * float(sc)
        st = out["sigma_theta"]
    # Reconstruct σc from σθ/SCF
    if not _finite(sc) and _finite(st) and _finite(scf) and float(scf) != 0:
        out["sigma_c"] = float(st) / float(scf)
        sc = out["sigma_c"]
    # Reconstruct SCF from stresses
    if not _finite(scf) and _finite(st) and _finite(sc):
        out["scf"] = compute_scf(float(st), float(sc))
    # Reconstruct σt from B1·σc if B1 present
    if not _finite(stens) and _finite(b1) and _finite(sc) and float(b1) != 0:
        out["sigma_t"] = float(sc) / float(b1)
        stens = out["sigma_t"]
    # Refresh derived when primaries available
    sc = out.get("sigma_c")
    stens = out.get("sigma_t")
    st = out.get("sigma_theta")
    if _finite(st) and _finite(sc):
        out["scf"] = compute_scf(float(st), float(sc))
    if _finite(sc) and _finite(stens):
        out["b1"] = compute_b1(float(sc), float(stens))
        out["b2"] = compute_b2(float(sc), float(stens))
    return out


def apply_missing_mask(case: dict[str, Any], cols: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Return a copy with listed columns set to NaN."""
    out = dict(case)
    for c in cols:
        out[c] = float("nan")
    return out


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

    @property
    def has_identity_violation(self) -> bool:
        return any(
            isinstance(c, dict) and c.get("ok") is False
            for c in self.exact_checks.values()
        )


def check_case(
    row: dict[str, Any],
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    scf_ood_threshold: float = 2.0,
    wet_range: tuple[float, float] = (0.5, 40.0),
    d_range: tuple[float, float] = (50.0, 2500.0),
    b1_implausible_max: float = DEFAULT_B1_IMPLAUSIBLE_MAX,
) -> PhysicsCheckResult:
    """Verify exact derived relations, empirical ranges, and criteria grades."""
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
            "abs_delta": (
                float(abs(scf_obs - scf_exp))
                if np.isfinite(scf_obs) and np.isfinite(scf_exp)
                else float("nan")
            ),
        },
        "b1": {
            "observed": b1_obs,
            "expected": b1_exp,
            "ok": _rel_ok(b1_obs, b1_exp, rel_tol, abs_tol),
            "relation": "exact: B1 = sigma_c / sigma_t",
            "abs_delta": (
                float(abs(b1_obs - b1_exp))
                if np.isfinite(b1_obs) and np.isfinite(b1_exp)
                else float("nan")
            ),
        },
        "b2": {
            "observed": b2_obs,
            "expected": b2_exp,
            "ok": _rel_ok(b2_obs, b2_exp, rel_tol, abs_tol),
            "relation": "exact: B2 = (sigma_c - sigma_t) / (sigma_c + sigma_t)",
            "abs_delta": (
                float(abs(b2_obs - b2_exp))
                if np.isfinite(b2_obs) and np.isfinite(b2_exp)
                else float("nan")
            ),
        },
    }
    for name, chk in exact.items():
        if not chk["ok"]:
            issues.append(
                f"{name} inconsistency: observed={chk['observed']:.4g}, "
                f"expected={chk['expected']:.4g}"
            )

    # Implausible primary ratio (listed B1 may look fine while σc/σt is absurd).
    if np.isfinite(b1_exp) and b1_exp > float(b1_implausible_max):
        issues.append(
            f"implausible_sigma_c_over_sigma_t: expected_b1={b1_exp:.4g} "
            f"(>{b1_implausible_max:g})"
        )

    # Suspected table row-shift: listed B1≈neighbor expected while own identity fails.
    if (
        not exact["b1"]["ok"]
        and np.isfinite(b1_obs)
        and np.isfinite(b1_exp)
        and abs(b1_obs - b1_exp) > max(abs_tol, rel_tol * abs(b1_exp)) * 5
    ):
        issues.append(
            "suspected_row_shift: listed B1 far from sigma_c/sigma_t "
            "(see rows 244–246 pattern)"
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

    # Criteria (as-given listed values) — main empirical rockburst signal for R_physics.
    from rockburst.physics.criteria import criteria_grades_for_case

    crit_as = criteria_grades_for_case(row, use_recomputed=False)
    crit_re = criteria_grades_for_case(row, use_recomputed=True)
    empirical["criteria_as_given"] = crit_as
    empirical["criteria_recomputed"] = crit_re
    empirical["data_policy"] = {
        "a": "as_given listed SCF/B1/B2",
        "b": "recompute SCF/B1/B2 from primary stresses",
    }

    n_exact = sum(1 for c in exact.values() if c["ok"])
    exact_score = n_exact / 3.0
    # Criteria agreement among Russenes / Kidybinski / brittleness (as-given).
    criteria_score = float(crit_as.get("agreement") or 0.0)
    if int(crit_as.get("n_valid") or 0) < 2:
        criteria_score = 0.5

    emp_penalties = 0.0
    if empirical.get("wet", {}).get("in_range") is False:
        emp_penalties += 0.1
    if empirical.get("d", {}).get("in_range") is False:
        emp_penalties += 0.05
    if empirical.get("d", {}).get("status") == "missing":
        emp_penalties += 0.05
    if empirical.get("scf_ood"):
        emp_penalties += 0.15
    if any("implausible_sigma_c_over_sigma_t" in i for i in issues):
        emp_penalties += 0.2
    if any("suspected_row_shift" in i for i in issues):
        emp_penalties += 0.15

    # Main R_physics: criteria consensus + identity consistency (Slice 3 / M1).
    r_physics = float(
        np.clip(0.55 * criteria_score + 0.45 * exact_score - emp_penalties, 0.0, 1.0)
    )
    return PhysicsCheckResult(
        r_physics=r_physics,
        exact_checks=exact,
        empirical_checks=empirical,
        issues=issues,
    )


def diagnose_row_shift_block(
    df_rows: list[dict[str, Any]],
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict[str, Any]:
    """Diagnose cyclic SCF/B1 shift across consecutive rows (e.g. 244–246)."""
    notes: list[str] = []
    details: list[dict[str, Any]] = []
    for i, row in enumerate(df_rows):
        scf_exp = compute_scf(float(row["sigma_theta"]), float(row["sigma_c"]))
        b1_exp = compute_b1(float(row["sigma_c"]), float(row["sigma_t"]))
        scf_obs = float(row["scf"])
        b1_obs = float(row["b1"])
        match_prev_b1 = False
        if i > 0:
            prev = df_rows[i - 1]
            prev_b1_exp = compute_b1(float(prev["sigma_c"]), float(prev["sigma_t"]))
            match_prev_b1 = _rel_ok(b1_obs, prev_b1_exp, rel_tol=0.05, abs_tol=0.5)
        details.append(
            {
                "no": row.get("no"),
                "scf_ok": _rel_ok(scf_obs, scf_exp, rel_tol, abs_tol),
                "b1_ok": _rel_ok(b1_obs, b1_exp, rel_tol, abs_tol),
                "listed_b1_matches_prev_expected": match_prev_b1,
            }
        )
        if match_prev_b1 and not _rel_ok(b1_obs, b1_exp, rel_tol, abs_tol):
            notes.append(
                f"row {row.get('no')}: listed B1≈previous row expected B1 (shift)"
            )
    return {"notes": notes, "details": details, "suspect": len(notes) >= 2}
