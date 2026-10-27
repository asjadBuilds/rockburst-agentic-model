"""Physics pre-check helpers for the geomechanics routing path (RM4).

Cheap path = identities + Wet/D/SCF-OOD range checks only. If clean, attach
**one** ``criteria_grades_for_case`` so ``R_physics`` matches the full tool.
Dirty cases fall through to full ``check_case``. Never wipe criteria to ``{}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.physics.relations import (
    DEFAULT_ABS_TOL,
    DEFAULT_B1_IMPLAUSIBLE_MAX,
    DEFAULT_K_RATIO_RANGE,
    DEFAULT_OVERBURDEN_MPA_PER_M,
    DEFAULT_REL_TOL,
    PhysicsCheckResult,
    check_case,
    compute_b1,
    compute_b2,
    compute_scf,
    kirsch_tangential_stress_range,
)


def _physics_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    p = (cfg or {}).get("physics", {})
    return {
        "rel_tol": float(p.get("rel_tol", DEFAULT_REL_TOL)),
        "abs_tol": float(p.get("abs_tol", DEFAULT_ABS_TOL)),
        "scf_ood_threshold": float(p.get("scf_ood_threshold", 2.0)),
        "wet_range": tuple(p.get("wet_range", [0.5, 40.0])),
        "d_range": tuple(p.get("d_range", [50.0, 2500.0])),
        "b1_implausible_max": float(p.get("b1_implausible_max", 200.0)),
        "overburden_mpa_per_m": float(p.get("overburden_mpa_per_m", DEFAULT_OVERBURDEN_MPA_PER_M)),
        "k_ratio_range": tuple(p.get("k_ratio_range", list(DEFAULT_K_RATIO_RANGE))),
    }


def _rel_ok(observed: float, expected: float, rel_tol: float, abs_tol: float) -> bool:
    if not (np.isfinite(observed) and np.isfinite(expected)):
        return False
    return abs(observed - expected) <= max(abs_tol, rel_tol * abs(expected))


def identities_and_ranges_only(
    case: dict[str, Any], cfg: dict[str, Any] | None = None
) -> PhysicsCheckResult:
    """RM4 cheap path: identities + Wet/D/SCF-OOD (+ depth/stress) — no criteria."""
    kw = _physics_kwargs(cfg)
    rel_tol = kw["rel_tol"]
    abs_tol = kw["abs_tol"]
    wet_range = kw["wet_range"]
    d_range = kw["d_range"]
    scf_ood_threshold = kw["scf_ood_threshold"]
    b1_implausible_max = kw["b1_implausible_max"]
    overburden_mpa_per_m = kw["overburden_mpa_per_m"]
    k_ratio_range = kw["k_ratio_range"]

    issues: list[str] = []
    st = float(case["sigma_theta"])
    sc = float(case["sigma_c"])
    stens = float(case["sigma_t"])
    scf_obs = float(case["scf"]) if case.get("scf") is not None else float("nan")
    b1_obs = float(case["b1"]) if case.get("b1") is not None else float("nan")
    b2_obs = float(case["b2"]) if case.get("b2") is not None else float("nan")
    wet = case.get("wet")
    depth = case.get("d")

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

    if np.isfinite(b1_exp) and b1_exp > float(b1_implausible_max):
        issues.append(
            f"implausible_sigma_c_over_sigma_t: expected_b1={b1_exp:.4g} "
            f"(>{b1_implausible_max:g})"
        )
    if (
        not exact["b1"]["ok"]
        and np.isfinite(b1_obs)
        and np.isfinite(b1_exp)
        and abs(b1_obs - b1_exp) > max(abs_tol, rel_tol * abs(b1_exp)) * 5
    ):
        issues.append("large_b1_identity_gap: listed B1 far from sigma_c/sigma_t")
    if np.isfinite(stens) and np.isfinite(sc) and stens >= sc and sc > 0:
        issues.append(
            f"sigma_t_ge_sigma_c: sigma_t={stens:.4g} >= sigma_c={sc:.4g}"
        )
    if np.isfinite(st) and np.isfinite(sc) and st >= sc and sc > 0:
        issues.append(
            "tangential_stress_exceeds_strength: "
            f"sigma_theta={st:.4g} >= sigma_c={sc:.4g} (SCF>=1)"
        )
    if np.isfinite(b2_obs) and not (0.0 < b2_obs < 1.0):
        issues.append(f"b2_out_of_bounds: B2={b2_obs:.4g} not in (0,1)")

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
        if np.isfinite(st) and d_f > 0:
            sig_v = float(overburden_mpa_per_m) * d_f
            k_lo, k_hi = float(k_ratio_range[0]), float(k_ratio_range[1])
            lo, hi = kirsch_tangential_stress_range(
                d_f,
                overburden_mpa_per_m=overburden_mpa_per_m,
                k_ratio_range=k_ratio_range,
            )
            ok_depth = lo <= float(st) <= hi
            empirical["depth_stress"] = {
                "sigma_theta": st,
                "sigma_v_overburden": sig_v,
                "k_ratio_range": [k_lo, k_hi],
                "plausible_tangential_stress_range": [lo, hi],
                "ok": ok_depth,
                "citation": (
                    "Kirsch max(3k-1,3-k)·σv; Brown and Hoek (1978); "
                    "σv≈0.027 MPa/m·D"
                ),
            }
            if not ok_depth:
                issues.append(
                    f"depth_stress_implausible: sigma_theta={st:.4g} "
                    f"not in [{lo:.4g},{hi:.4g}] for D={d_f:.4g} m"
                )

    if np.isfinite(scf_obs) and scf_obs > scf_ood_threshold:
        issues.append(f"SCF OOD (> {scf_ood_threshold}): {scf_obs:.3g}")
        empirical["scf_ood"] = True

    # Placeholder score without criteria (refined on clean attach path).
    n_exact = sum(1 for c in exact.values() if c["ok"])
    exact_score = n_exact / 3.0
    r_physics = float(np.clip(0.45 * exact_score, 0.0, 1.0))
    return PhysicsCheckResult(
        r_physics=r_physics,
        exact_checks=exact,
        empirical_checks=empirical,
        issues=issues,
    )


def _attach_criteria_score(
    partial: PhysicsCheckResult, case: dict[str, Any]
) -> PhysicsCheckResult:
    """One ``criteria_grades_for_case``; score matches full ``check_case`` formula."""
    from rockburst.physics.criteria import criteria_grades_for_case

    crit_as = criteria_grades_for_case(case, use_recomputed=False)
    emp = dict(partial.empirical_checks or {})
    emp["criteria_as_given"] = crit_as
    emp["data_policy"] = {
        "a": "as_given listed SCF/B1/B2",
        "b": "recompute skipped on clean RM4 path",
    }

    criteria_score = float(crit_as.get("agreement") or 0.0)
    if int(crit_as.get("n_valid") or 0) < 2:
        criteria_score = 0.5
    n_exact = sum(
        1
        for c in (partial.exact_checks or {}).values()
        if isinstance(c, dict) and c.get("ok")
    )
    exact_score = n_exact / 3.0
    # Clean path ⇒ no empirical penalties (is_perfect_physics already passed).
    r_physics = float(np.clip(0.55 * criteria_score + 0.45 * exact_score, 0.0, 1.0))
    return PhysicsCheckResult(
        r_physics=r_physics,
        exact_checks=dict(partial.exact_checks or {}),
        empirical_checks=emp,
        issues=list(partial.issues),
    )


def full_physics_result(
    case: dict[str, Any], cfg: dict[str, Any] | None = None
) -> PhysicsCheckResult:
    """Same scoring path as the geomechanics tool (`check_case`)."""
    return check_case(case, **_physics_kwargs(cfg))


def is_perfect_physics(res: PhysicsCheckResult) -> bool:
    """True when identities hold and empirical ranges are clean (skip label).

    Criteria disagreement may lower ``r_physics`` below 1.0 without blocking
    the clean label — criteria remain in ``empirical_checks`` (RM4).
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


def physics_for_geomechanics(
    case: dict[str, Any], cfg: dict[str, Any] | None = None
) -> PhysicsCheckResult:
    """RM4 routing entry: cheap identities/ranges; clean→one criteria; dirty→full."""
    partial = identities_and_ranges_only(case, cfg)
    if is_perfect_physics(partial):
        return _attach_criteria_score(partial, case)
    return full_physics_result(case, cfg)


def cheap_physics_precheck(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Clean-label decision via identities + ranges only (no criteria cost)."""
    res = identities_and_ranges_only(case, cfg)
    clean = is_perfect_physics(res)
    return {
        "clean": clean,
        "r_physics": float(res.r_physics),
        "issues": list(res.issues),
        "checks": res.exact_checks,
        "empirical_checks": res.empirical_checks,
        "reasons": list(res.issues),
    }


def should_skip_geomechanics(case: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    """True iff the case would take the clean/skipped geomechanics label (RM4)."""
    return is_perfect_physics(identities_and_ranges_only(case, cfg))
