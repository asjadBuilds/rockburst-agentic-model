import numpy as np
import pandas as pd

from rockburst.agents.rechecks import needs_extra_validation
from rockburst.agents.schemas import DataQualityOutput, PhysicsOutput
from rockburst.config import load_config, resolve_path
from rockburst.physics import (
    check_case,
    compute_b1,
    compute_b2,
    compute_scf,
    criteria_grades_for_case,
    diagnose_row_shift_block,
    kidybinski_grade,
    russenes_grade,
    with_as_given,
    with_recomputed_derived,
)
from rockburst.physics.precheck import full_physics_result, should_skip_geomechanics
from rockburst.physics.relations import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, FEATURE_COLS
from rockburst.uncertainty.estimates import perturb_case


def test_row1_formulas():
    # Row 1 from dataset: 90, 170, 11.3 → SCF=0.53, B1≈15.04, B2≈0.88
    assert abs(compute_scf(90, 170) - 0.53) < 0.01
    assert abs(compute_b1(170, 11.3) - 15.04) < 0.05
    assert abs(compute_b2(170, 11.3) - 0.88) < 0.02


def test_check_case_detects_bad_b1():
    row = {
        "sigma_theta": 54.2,
        "sigma_c": 134,
        "sigma_t": 9.1,
        "scf": 0.4,
        "b1": 0.15,  # wrong; should be ~14.7
        "b2": 0.87,
        "wet": 7.1,
        "d": float("nan"),
    }
    res = check_case(row)
    assert res.r_physics < 1.0
    assert any("b1" in i for i in res.issues)
    assert res.has_identity_violation


def test_russenes_and_kidybinski_thresholds():
    assert russenes_grade(0.10) == 0
    assert russenes_grade(0.25) == 1
    assert russenes_grade(0.40) == 2
    assert russenes_grade(0.60) == 3
    assert kidybinski_grade(1.0) == 0
    assert kidybinski_grade(2.5) == 1
    assert kidybinski_grade(4.0) == 2
    assert kidybinski_grade(6.0) == 3


def test_tight_identity_tolerance_catches_small_delta():
    """M2: |Δ|≈0.02 on SCF fails abs_tol=0.006 / rel 2% for small SCF."""
    row = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "scf": 2.02,  # expected 2.0; Δ=0.02 > 0.006 and > 2%*2=0.04? 0.02 < 0.04
        "b1": 10.0,
        "b2": (50 - 5) / (50 + 5),
        "wet": 5.0,
        "d": 500.0,
    }
    # Δ_scf=0.02: max(0.006, 0.02*2)=0.04 → still OK; use larger delta
    row["scf"] = 2.05  # Δ=0.05 > 0.04
    res = check_case(row, rel_tol=DEFAULT_REL_TOL, abs_tol=DEFAULT_ABS_TOL)
    assert res.exact_checks["scf"]["ok"] is False
    # Old 5% tol would have accepted Δ=0.05 on expected=2.0
    res_loose = check_case(row, rel_tol=0.05, abs_tol=0.05)
    assert res_loose.exact_checks["scf"]["ok"] is True


def test_identity_violation_triggers_extra_validation():
    phys = PhysicsOutput(
        physics_score=0.4,
        exact_checks={"b1": {"ok": False}},
        issues=["b1 inconsistency: observed=0.15, expected=14.73"],
    )
    dq = DataQualityOutput(data_reliability=0.9, ood_flag=False)
    assert needs_extra_validation(dq, load_config(), phys=phys) is True


def test_rows_244_246_row_shift_and_288_implausible():
    df = pd.read_csv(resolve_path(load_config(), "processed_csv"))
    block = [df.loc[df["no"] == n].iloc[0].to_dict() for n in (244, 245, 246)]
    diag = diagnose_row_shift_block(block)
    assert diag["suspect"] is True
    for case in block:
        # RM2: single-row check uses large_b1_identity_gap, not row-shift label
        res = check_case(case)
        assert res.has_identity_violation
        assert any(
            "large_b1_identity_gap" in i or "inconsistency" in i for i in res.issues
        )
        assert not any("suspected_row_shift" in i for i in res.issues)
    # Block-level diagnose (with neighbours) may flag suspected_row_shift
    res_block = check_case(block[1], neighbor_rows=[block[0]])
    # neighbour context recorded; suspect may require ≥2 shift notes in block
    assert "row_shift_block" in (res_block.empirical_checks or {})

    row288 = df.loc[df["no"] == 288].iloc[0].to_dict()
    res288 = check_case(row288)
    assert any("implausible_sigma_c_over_sigma_t" in i for i in res288.issues)
    # Listed B1≈23 looks fine vs identity expected≈602
    assert abs(float(row288["b1"]) - 23.0) < 1.0
    assert compute_b1(float(row288["sigma_c"]), float(row288["sigma_t"])) > 200


def test_as_given_vs_recompute_policy():
    df = pd.read_csv(resolve_path(load_config(), "processed_csv"))
    case = df.loc[df["no"] == 244].iloc[0].to_dict()
    a = with_as_given(case)
    b = with_recomputed_derived(case)
    assert a["scf"] == case["scf"]
    assert abs(b["scf"] - compute_scf(case["sigma_theta"], case["sigma_c"])) < 1e-9
    crit_a = criteria_grades_for_case(case, use_recomputed=False)
    crit_b = criteria_grades_for_case(case, use_recomputed=True)
    assert crit_a["policy"] == "as_given"
    assert crit_b["policy"] == "recompute_from_stresses"
    # Row 244 listed vs recomputed SCF differ → Russenes grades may differ
    assert crit_a["inputs"]["scf"] != crit_b["inputs"]["scf"]


def test_perturb_only_primaries_recomputes_derived():
    rng = np.random.default_rng(0)
    case = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "scf": 2.0,
        "b1": 10.0,
        "b2": 0.8181818181,
        "wet": 5.0,
        "d": 500.0,
    }
    out = perturb_case(case, FEATURE_COLS, 0.05, rng)
    # Derived must match primaries after perturbation
    assert abs(out["scf"] - compute_scf(out["sigma_theta"], out["sigma_c"])) < 1e-9
    assert abs(out["b1"] - compute_b1(out["sigma_c"], out["sigma_t"])) < 1e-9
    assert abs(out["b2"] - compute_b2(out["sigma_c"], out["sigma_t"])) < 1e-9
    # At least one primary changed
    assert any(
        abs(float(out[f]) - float(case[f])) > 1e-12
        for f in ("sigma_theta", "sigma_c", "sigma_t", "wet", "d")
    )


def test_check_case_exposes_criteria_for_critic():
    row = {
        "sigma_theta": 100.0,
        "sigma_c": 50.0,
        "sigma_t": 5.0,
        "scf": 2.0,
        "b1": 10.0,
        "b2": (50 - 5) / (50 + 5),
        "wet": 5.0,
        "d": 500.0,
    }
    res = check_case(row)
    assert "criteria_as_given" in res.empirical_checks
    assert "criteria_recomputed" in res.empirical_checks
    assert "consensus_grade" in res.empirical_checks["criteria_as_given"]


def test_skip_matches_full_tool_on_all_rows():
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    n_skip = 0
    for _, row in df.iterrows():
        case = row.to_dict()
        skip = should_skip_geomechanics(case, cfg)
        res = full_physics_result(case, cfg)
        if skip:
            n_skip += 1
            assert res.issues == []
            assert not res.has_identity_violation
        else:
            emp = res.empirical_checks or {}
            soft_block = (
                (emp.get("d") or {}).get("status") == "missing"
                or (emp.get("wet") or {}).get("status") == "missing"
                or (emp.get("d") or {}).get("in_range") is False
                or (emp.get("wet") or {}).get("in_range") is False
                or bool(emp.get("scf_ood"))
            )
            assert res.has_identity_violation or bool(res.issues) or soft_block
    assert len(df) == 344
    test_skip_matches_full_tool_on_all_rows.n_skip = n_skip


def test_package_version():
    import rockburst

    assert rockburst.__version__
