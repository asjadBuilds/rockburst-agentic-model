from rockburst.physics import check_case, compute_b1, compute_b2, compute_scf


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


def test_package_version():
    import rockburst

    assert rockburst.__version__
