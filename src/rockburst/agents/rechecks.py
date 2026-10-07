"""Recheck tools that differ from first-pass agents (Critic follow-ups)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from rockburst.ml.pipelines import build_preprocess, get_model_zoo
from rockburst.physics.relations import FEATURE_COLS, check_case
from rockburst.uncertainty.estimates import perturb_case, stability_from_preds


def _pad_proba(proba: np.ndarray, n: int = 4) -> np.ndarray:
    proba = np.asarray(proba, dtype=float).ravel()
    if len(proba) < n:
        pad = np.zeros(n)
        pad[: len(proba)] = proba
        return pad
    return proba[:n]


def _case_frame(case: dict[str, Any]):
    import pandas as pd

    row = {c: case.get(c, np.nan) for c in FEATURE_COLS}
    df = pd.DataFrame([row])
    for c in FEATURE_COLS:
        df[f"observed_{c}"] = (~df[c].isna()).astype(float)
    return df


def _predict_with_pipe(pipe: Any, case: dict[str, Any]) -> np.ndarray:
    X = _case_frame(case)
    if hasattr(pipe, "predict_proba"):
        return _pad_proba(pipe.predict_proba(X)[0])
    pred = int(pipe.predict(X)[0])
    proba = np.zeros(4)
    proba[pred] = 1.0
    return proba


def physics_recheck_perturbed_tolerance(
    case: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Re-evaluate exact relations under a tighter/looser tolerance band.

    Returns borderline flags — information the first-pass physics tool does not emit.
    """
    p = cfg.get("physics", {})
    base_rel = float(p.get("rel_tol", 0.05))
    base_abs = float(p.get("abs_tol", 0.05))
    # Band: tighter (0.5x) and stated (1x); borderline = ok at stated, fail at tighter
    tight = check_case(
        case,
        rel_tol=base_rel * 0.5,
        abs_tol=base_abs * 0.5,
        scf_ood_threshold=p.get("scf_ood_threshold", 2.0),
        wet_range=tuple(p.get("wet_range", [0.5, 40.0])),
        d_range=tuple(p.get("d_range", [50.0, 2500.0])),
    )
    stated = check_case(
        case,
        rel_tol=base_rel,
        abs_tol=base_abs,
        scf_ood_threshold=p.get("scf_ood_threshold", 2.0),
        wet_range=tuple(p.get("wet_range", [0.5, 40.0])),
        d_range=tuple(p.get("d_range", [50.0, 2500.0])),
    )
    borderline: list[str] = []
    for name in ("scf", "b1", "b2"):
        ok_stated = bool(stated.exact_checks.get(name, {}).get("ok"))
        ok_tight = bool(tight.exact_checks.get(name, {}).get("ok"))
        if ok_stated and not ok_tight:
            borderline.append(name)
    return {
        "tool": "physics_recheck_perturbed_tolerance",
        "physics_score_stated": stated.r_physics,
        "physics_score_tight": tight.r_physics,
        "borderline_relations": borderline,
        "issues_stated": list(stated.issues),
        "issues_tight": list(tight.issues),
        "low_physics": stated.r_physics < 0.6,
    }


def alt_imputer_reprediction(
    case: dict[str, Any],
    cfg: dict[str, Any],
    *,
    X_train: Any,
    y_train: Any,
    primary_imputer: str = "knn",
    primary_proba: np.ndarray | None = None,
    primary_grade: int | None = None,
) -> dict[str, Any]:
    """Re-predict with the other fold-trained imputers; report spread and grade flips."""
    kinds = ["median", "knn", "iterative"]
    cache = cfg.setdefault("_fold_alt_pipes", {})
    if not cache:
        zoo = get_model_zoo(quick=True)
        est_name = "rf" if "rf" in zoo else next(iter(zoo))
        base = zoo[est_name]
        for kind in kinds:
            pipe = Pipeline(
                [("prep", build_preprocess(kind)), ("model", clone(base))]
            )
            pipe.fit(X_train, y_train)
            cache[kind] = pipe
    probas: dict[str, list[float]] = {}
    grades: dict[str, int] = {}
    for kind in kinds:
        pr = _predict_with_pipe(cache[kind], case)
        probas[kind] = pr.tolist()
        grades[kind] = int(pr.argmax())

    mat = np.array([probas[k] for k in kinds], dtype=float)
    pair_spreads = []
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            pair_spreads.append(0.5 * float(np.abs(mat[i] - mat[j]).sum()))
    prob_spread = float(max(pair_spreads)) if pair_spreads else 0.0
    grade_set = set(grades.values())
    if primary_grade is not None:
        grade_set.add(int(primary_grade))
    grade_flip = len(grade_set) > 1
    large_spread = prob_spread >= 0.15 or grade_flip
    return {
        "tool": "alt_imputer_reprediction",
        "imputers": kinds,
        "probas": probas,
        "grades": grades,
        "prob_spread": prob_spread,
        "grade_flip": grade_flip,
        "large_spread": large_spread,
        "primary_imputer": primary_imputer,
        "primary_grade": primary_grade,
    }


def targeted_perturbation_recheck(
    case: dict[str, Any],
    cfg: dict[str, Any],
    *,
    pipeline: Any,
    flagged_features: list[str] | None = None,
    primary_grade: int,
    primary_proba: np.ndarray,
) -> dict[str, Any]:
    """Perturb only imputed/flagged features; report stability for those features."""
    u = cfg.get("uncertainty", {})
    mild = float(u.get("mild_eps", 0.05))
    strong = float(u.get("strong_eps", 0.15))
    n_pert = int(u.get("n_perturb", 10))
    feats = flagged_features or [
        c
        for c in FEATURE_COLS
        if case.get(c) is None
        or (isinstance(case.get(c), float) and not np.isfinite(float(case[c])))
    ]
    if not feats:
        # Fall back to all measured features if nothing flagged
        feats = list(FEATURE_COLS)
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 17)
    preds, probas = [], []
    for eps in (mild, strong):
        for _ in range(max(1, n_pert // 2)):
            pc = perturb_case(case, feats, eps, rng)
            pp = _predict_with_pipe(pipeline, pc)
            preds.append(int(np.argmax(pp)))
            probas.append(pp)
    stab = stability_from_preds(primary_grade, preds, np.asarray(primary_proba), probas)
    return {
        "tool": "targeted_perturbation_recheck",
        "features": feats,
        "stability_score": stab["stability_score"],
        "prediction_stability": stab["prediction_stability"],
        "flip_rate": stab["flip_rate"],
        "low_stability": stab["stability_score"] < 0.55,
    }


def cross_check_model_reprediction(
    case: dict[str, Any],
    cfg: dict[str, Any],
    *,
    X_train: Any,
    y_train: Any,
    primary_imputer: str = "knn",
    primary_grade: int,
    primary_proba: np.ndarray,
) -> dict[str, Any]:
    """Independent model family (ordinal vs tree) as Critic cross-check."""
    cache_key = f"cross__{primary_imputer}"
    cache = cfg.setdefault("_fold_cross_pipes", {})
    if cache_key not in cache:
        zoo = get_model_zoo(quick=True)
        if "ordinal_logistic" in zoo:
            cross_name = "ordinal_logistic"
            cross_est = zoo["ordinal_logistic"]
        else:
            cross_name = "logistic"
            cross_est = zoo["logistic"]
        pipe = Pipeline(
            [("prep", build_preprocess(primary_imputer)), ("model", clone(cross_est))]
        )
        pipe.fit(X_train, y_train)
        cache[cache_key] = (cross_name, pipe)
    cross_name, pipe = cache[cache_key]
    pr = _predict_with_pipe(pipe, case)
    cross_grade = int(pr.argmax())
    disagree = cross_grade != int(primary_grade)
    soft = 0.5 * float(np.abs(np.asarray(primary_proba) - pr).sum())
    return {
        "tool": "cross_check_model_reprediction",
        "cross_model": cross_name,
        "cross_grade": cross_grade,
        "cross_proba": pr.tolist(),
        "primary_grade": int(primary_grade),
        "grade_disagree": disagree,
        "soft_disagreement": soft,
        "large_disagreement": disagree or soft >= 0.35,
    }


def needs_extra_validation(
    data_q: Any,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Pre-registered extra-validation gate."""
    cfg = cfg or {}
    cut = float(cfg.get("routing", {}).get("extra_validation_r_data", 0.55))
    if data_q is None:
        return False
    missing = list(getattr(data_q, "missing_features", []) or [])
    if "d" in missing or any(m == "d" for m in missing):
        return True
    if missing:
        return True
    if float(getattr(data_q, "data_reliability", 1.0)) < cut:
        return True
    return False


def validation_objections_from_rechecks(rechecks: dict[str, Any]) -> list[str]:
    """Unresolved objections from extra-validation / recheck payloads."""
    objs: list[str] = []
    alt = rechecks.get("alt_imputer")
    if alt and alt.get("large_spread"):
        objs.append("alt_imputer_large_spread")
    pert = rechecks.get("targeted_perturbation")
    if pert and pert.get("low_stability"):
        objs.append("targeted_perturbation_low_stability")
    phys = rechecks.get("physics_tol")
    if phys and (phys.get("borderline_relations") or phys.get("low_physics")):
        objs.append("physics_borderline_or_low")
    cross = rechecks.get("cross_check")
    if cross and cross.get("large_disagreement"):
        objs.append("cross_check_disagreement")
    return objs
