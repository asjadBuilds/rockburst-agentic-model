"""Empirical rockburst intensity criteria (deterministic; no LLM arithmetic).

Sources (thresholds cited; direction: higher index → higher intensity 0–3):

- Russenes, B. F. (1974). Analysis of Rock Spalling for Tunnels in Steep Valley
  Sides. MSc thesis, Norwegian Institute of Technology. Base article Table 4
  adapts SCF = σθ/σc bands at 0.20 / 0.30 / 0.55 (none → light → moderate →
  strong). Caveat: original Russenes uses different stress-ratio framing;
  thresholds here follow the project base article’s Table 4 adaptation.
- Kidybinski, A. (1981). Bursting liability indices of coal. Wet bands at
  2.0 / 3.5 / 5.0. Caveat: derived for coal bursting liability; applied here
  as in the base article’s rockburst catalogue screening.
- Brittleness B1 and B2 are retained as diagnostics only. Round-3 validation
  found B1 uninformative on this catalogue, so neither enters consensus.
- Hoek/Brown stress-ratio bands are retained as diagnostics only because they
  use the same σθ/σc input as Russenes. Brown & Hoek (1980), Underground
  Excavations in Rock, Institution of Mining and Metallurgy, London.

Grades are 0–3 to match the project label space (none/light/moderate/strong).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rockburst.physics.relations import compute_b1, compute_b2, compute_scf


# Russenes (1974) adaptation — base article Table 4 SCF = σθ/σc
RUSSENES_SCF_THRESHOLDS = (0.20, 0.30, 0.55)

# Kidybinski (1981) Wet (coal bursting liability; catalogue use)
KIDYBINSKI_WET_THRESHOLDS = (2.0, 3.5, 5.0)

# Brittleness B1 = σc/σt — base article Table 4 (not 10/15/20)
BRITTLENESS_B1_THRESHOLDS = (10.0, 14.0, 18.0)

# B2 diagnostic bands only (not a consensus voter). Equivalent B1≈9/19/39.
BRITTLENESS_B2_THRESHOLDS = (0.80, 0.90, 0.95)

# E. Hoek criterion — base article Table 4 (σθ/σc)
HOEK_SCF_THRESHOLDS = (0.42, 0.56, 0.70)

THRESHOLD_CITATIONS: dict[str, str] = {
    "russenes": (
        "Russenes (1974) MSc NTH; Table 4 adaptation SCF 0.20/0.30/0.55"
    ),
    "kidybinski": (
        "Kidybinski (1981) coal Wet 2.0/3.5/5.0; catalogue screening caveat"
    ),
    "brittleness_b1": "Base article Table 4 B1=σc/σt cut-offs 10/14/18",
    "brittleness_b2_diagnostic": (
        "Diagnostic only; B2=(σc−σt)/(σc+σt); not a consensus vote (RM5)"
    ),
    "hoek_diagnostic": (
        "Brown & Hoek (1980), Underground Excavations in Rock; base-article "
        "Table 4 σθ/σc cut-offs 0.42/0.56/0.70; diagnostic only"
    ),
}


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
    """Diagnostic B2 grade (not used in consensus — RM5 single brittleness vote)."""
    return _grade_from_thresholds(float(b2), thresholds)


def hoek_grade(scf: float, thresholds: tuple[float, ...] = HOEK_SCF_THRESHOLDS) -> int:
    """Hoek intensity from σθ/σc (0–3, or -1 if undefined)."""
    return _grade_from_thresholds(float(scf), thresholds)


def _round_half_up(x: float) -> int:
    """Explicit half-up rounding (avoids int(median) truncating 1.5 → 1)."""
    return int(np.floor(float(x) + 0.5))


def consensus_from_grades(grades: dict[str, int]) -> tuple[int, float, str]:
    """Agreement-weighted consensus with explicit half-up median fallback.

    Voters with valid grades (≥0) each cast one vote. If a unique plurality
    exists, that grade wins; otherwise the half-up median of valid grades.
    Returns (consensus_grade, agreement, rule_name).
    """
    valid = {k: int(g) for k, g in grades.items() if int(g) >= 0}
    if not valid:
        return -1, 0.0, "none"
    vals = list(valid.values())
    # Plurality (agreement-weighted: each voter weight 1)
    counts: dict[int, int] = {}
    for g in vals:
        counts[g] = counts.get(g, 0) + 1
    best = max(counts.values())
    winners = sorted(g for g, c in counts.items() if c == best)
    if len(winners) == 1:
        cons = int(winners[0])
        rule = "plurality"
    else:
        cons = _round_half_up(float(np.median(vals)))
        rule = "rounded_median_tiebreak"
    agreement = float(sum(1 for g in vals if g == cons)) / float(len(vals))
    return cons, agreement, rule


def criteria_grades_for_case(
    row: dict[str, Any],
    *,
    use_recomputed: bool = False,
) -> dict[str, Any]:
    """Compute independent consensus voters plus diagnostic criteria.

    ``use_recomputed=True`` derives SCF/B1/B2 from primary stresses (policy b);
    otherwise uses listed values when finite (policy a: as-given).

    Consensus uses one stress-ratio voter (Russenes) plus Kidybinski Wet.
    B1/B2 and Hoek remain visible diagnostics but do not double-vote correlated
    quantities (TM2).
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

    # Independent consensus voters: one stress ratio + one energy index.
    grades = {
        "russenes": russenes_grade(scf),
        "kidybinski": kidybinski_grade(wet_f),
    }
    diagnostic = {
        "brittleness_b1": brittleness_b1_grade(b1),
        "brittleness_b2": brittleness_b2_grade(b2),
        "hoek": hoek_grade(scf),
    }
    consensus, agreement, cons_rule = consensus_from_grades(grades)
    n_valid = sum(1 for g in grades.values() if g >= 0)
    return {
        "policy": "recompute_from_stresses" if use_recomputed else "as_given",
        "inputs": {"scf": scf, "b1": b1, "b2": b2, "wet": wet_f},
        "grades": grades,
        "diagnostic_grades": diagnostic,
        "consensus_grade": consensus,
        "agreement": agreement,
        "consensus_rule": cons_rule,
        "n_valid": n_valid,
        "threshold_citations": dict(THRESHOLD_CITATIONS),
        "voting_policy": "russenes_plus_kidybinski_equal_weight",
        "excluded_from_consensus": ["brittleness_b1", "brittleness_b2", "hoek"],
    }


def criteria_agree_with_ml(criteria: dict[str, Any], ml_grade: int) -> bool:
    cons = int(criteria.get("consensus_grade", -1))
    if cons < 0:
        return True
    return int(ml_grade) == cons


def criterion_label_agreement_table(
    rows: list[dict[str, Any]],
    labels: list[int],
    *,
    use_recomputed: bool = False,
) -> dict[str, Any]:
    """Per-criterion agreement with labels (for Phase 5/6 reporting, RM5)."""
    keys = ["russenes", "kidybinski", "brittleness_b1", "hoek", "consensus"]
    hits = {k: 0 for k in keys}
    n = 0
    for row, y in zip(rows, labels):
        crit = criteria_grades_for_case(row, use_recomputed=use_recomputed)
        n += 1
        y_i = int(y)
        for k, g in {
            **(crit.get("grades") or {}),
            **(crit.get("diagnostic_grades") or {}),
        }.items():
            if int(g) >= 0 and int(g) == y_i:
                hits[k] = hits.get(k, 0) + 1
        cons = int(crit.get("consensus_grade", -1))
        if cons >= 0 and cons == y_i:
            hits["consensus"] += 1
    return {
        "n": n,
        "agreement": {k: (hits[k] / n if n else 0.0) for k in hits},
        "hits": hits,
        "note": "Use to set ml_vs_criteria severity from measured precision (RM5).",
    }
