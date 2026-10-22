"""Empirical rockburst intensity criteria (deterministic; no LLM arithmetic).

Sources (thresholds as commonly tabulated for deep mining / rockburst screening):

- Russenes, B. F. (1974). Analysis of rock spalling for tunnels under high
  stress. Uses tangential stress ratio SCF = σθ/σc with bands at 0.20 / 0.30 /
  0.55 (none → light → moderate → strong).
- Kidybinski, A. (1981). Bursting liability indices of coal. Elastic strain
  energy index Wet with bands at 2.0 / 3.5 / 5.0.
- Brittleness: B1 = σc/σt and B2 = (σc−σt)/(σc+σt) as used in Chinese
  rockburst catalogues (e.g. bands B1 at 10 / 15 / 20; B2 at 0.80 / 0.90 / 0.95).

Grades are 0–3 to match the project label space (none/light/moderate/strong).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.physics.relations import compute_b1, compute_b2, compute_scf


# Russenes (1974) SCF = σθ/σc
RUSSENES_SCF_THRESHOLDS = (0.20, 0.30, 0.55)

# Kidybinski (1981) Wet
KIDYBINSKI_WET_THRESHOLDS = (2.0, 3.5, 5.0)

# Brittleness B1 = σc/σt and B2
BRITTLENESS_B1_THRESHOLDS = (10.0, 15.0, 20.0)
BRITTLENESS_B2_THRESHOLDS = (0.80, 0.90, 0.95)


def _grade_from_thresholds(value: float, thresholds: tuple[float, ...]) -> int:
    """Map value to 0..len(thresholds) using ascending cut-points."""
    if not np.isfinite(value):
        return -1
    g = 0
    for t in thresholds:
        if value >= t:
            g += 1
        else:
            break
    return int(g)


def russenes_grade(scf: float, thresholds: tuple[float, ...] = RUSSENES_SCF_THRESHOLDS) -> int:
    """Russenes intensity from SCF = σθ/σc (0–3, or -1 if undefined)."""
    return _grade_from_thresholds(float(scf), thresholds)


def kidybinski_grade(
    wet: float, thresholds: tuple[float, ...] = KIDYBINSKI_WET_THRESHOLDS
) -> int:
    """Kidybinski intensity from Wet (0–3, or -1 if undefined)."""
    return _grade_from_thresholds(float(wet), thresholds)


def brittleness_b1_grade(
    b1: float, thresholds: tuple[float, ...] = BRITTLENESS_B1_THRESHOLDS
) -> int:
    return _grade_from_thresholds(float(b1), thresholds)


def brittleness_b2_grade(
    b2: float, thresholds: tuple[float, ...] = BRITTLENESS_B2_THRESHOLDS
) -> int:
    return _grade_from_thresholds(float(b2), thresholds)


def criteria_grades_for_case(
    row: dict[str, Any],
    *,
    use_recomputed: bool = False,
) -> dict[str, Any]:
    """Compute Russenes / Kidybinski / brittleness grades for one case.

    ``use_recomputed=True`` derives SCF/B1/B2 from primary stresses (policy b);
    otherwise uses listed values when finite (policy a: as-given).
    """
    st = row.get("sigma_theta")
    sc = row.get("sigma_c")
    stens = row.get("sigma_t")
    wet = row.get("wet")

    scf_listed = row.get("scf")
    b1_listed = row.get("b1")
    b2_listed = row.get("b2")

    scf_exp = (
        compute_scf(float(st), float(sc))
        if st is not None and sc is not None and np.isfinite(float(st)) and np.isfinite(float(sc))
        else float("nan")
    )
    b1_exp = (
        compute_b1(float(sc), float(stens))
        if sc is not None
        and stens is not None
        and np.isfinite(float(sc))
        and np.isfinite(float(stens))
        else float("nan")
    )
    b2_exp = (
        compute_b2(float(sc), float(stens))
        if sc is not None
        and stens is not None
        and np.isfinite(float(sc))
        and np.isfinite(float(stens))
        else float("nan")
    )

    def _pick(listed: Any, expected: float) -> float:
        if use_recomputed:
            return float(expected)
        if listed is not None and np.isfinite(float(listed)):
            return float(listed)
        return float(expected)

    scf = _pick(scf_listed, scf_exp)
    b1 = _pick(b1_listed, b1_exp)
    b2 = _pick(b2_listed, b2_exp)
    wet_f = float(wet) if wet is not None and np.isfinite(float(wet)) else float("nan")

    grades = {
        "russenes": russenes_grade(scf),
        "kidybinski": kidybinski_grade(wet_f),
        "brittleness_b1": brittleness_b1_grade(b1),
        "brittleness_b2": brittleness_b2_grade(b2),
    }
    valid = [g for g in grades.values() if g >= 0]
    consensus = int(np.median(valid)) if valid else -1
    if valid:
        agreement = float(sum(1 for g in valid if g == consensus)) / float(len(valid))
    else:
        agreement = 0.0
    return {
        "policy": "recompute_from_stresses" if use_recomputed else "as_given",
        "inputs": {"scf": scf, "b1": b1, "b2": b2, "wet": wet_f},
        "grades": grades,
        "consensus_grade": consensus,
        "agreement": agreement,
        "n_valid": len(valid),
    }


def criteria_agree_with_ml(criteria: dict[str, Any], ml_grade: int) -> bool:
    cons = int(criteria.get("consensus_grade", -1))
    if cons < 0:
        return True
    return int(ml_grade) == cons
