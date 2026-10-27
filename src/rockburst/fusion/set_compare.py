"""RM7: fusion set A–D comparison + per-component AUROC on a leakage-safe R matrix.

Phase 5 runs this on the final OOF R (after RC2 cross-fit). Agent ships the
hooks; does not execute the full train that produces R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from rockburst.fusion.components import (
    FUSION_SET_CANDIDATES,
    SIMPLICITY_ORDER,
    mask_for_set,
)
from rockburst.fusion.stacker import (
    CHOSEN_FUSION_SET,
    FEATURE_NAMES,
    ReliabilityFusion,
    bootstrap_coef_cis,
    decision_band_occupancy,
    decision_utility_sensitivity,
    fit_calibrator_on_crossfit_raw,
    locked_feature_mask,
    select_decision_thresholds,
)


def production_mask_from_comparison(
    comparison: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the declared set rule and filter confidently inverted reliabilities."""
    chosen = str(comparison.get("chosen_set") or "")
    if chosen not in FUSION_SET_CANDIDATES:
        raise ValueError("fusion comparison has no valid chosen_set")
    mask = mask_for_set(chosen)
    excluded: list[str] = []
    component = (comparison.get("component_auroc") or {}).get("per_component") or {}
    # Reliability channels are oriented high=trust. If their AUROC CI is wholly
    # below chance, they are invalid in that orientation and are removed.
    for name in ("r_data", "r_physics", "r_model", "r_uncertainty", "r_stability"):
        block = component.get(name) or {}
        ci = block.get("ci95")
        i = FEATURE_NAMES.index(name)
        if mask[i] and isinstance(ci, (list, tuple)) and len(ci) == 2 and float(ci[1]) < 0.5:
            mask[i] = False
            excluded.append(name)
    if not mask.any():
        raise ValueError("selected fusion set became empty after orientation checks")
    return mask, {
        "selected_set": chosen,
        "active_features": [FEATURE_NAMES[i] for i, active in enumerate(mask) if active],
        "excluded_inverted_reliability_channels": excluded,
        "n_missing_semantics": (
            "risk count, so AUROC<0.5 versus correctness is expected; coefficient is free-signed"
        ),
        "selection_rule": comparison.get("selection_rule"),
    }


def component_aurocs(
    R: np.ndarray,
    y_correct: np.ndarray,
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Univariate AUROC of each reliability channel vs correctness (RM7)."""
    R = np.asarray(R, dtype=float)
    y = np.asarray(y_correct, dtype=int).ravel()
    names = names or list(FEATURE_NAMES)
    out: dict[str, Any] = {}
    if len(np.unique(y)) < 2:
        return {n: None for n in names}
    for i, name in enumerate(names):
        if i >= R.shape[1]:
            out[name] = None
            continue
        x = R[:, i]
        if len(np.unique(np.round(x, 6))) < 2:
            out[name] = None
            continue
        try:
            out[name] = float(roc_auc_score(y, x))
        except ValueError:
            out[name] = None
    return out


def component_auroc_bootstrap(
    R: np.ndarray,
    y_correct: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Point AUROC + grouped bootstrap CI per channel (Phase 5 artefact)."""
    point = component_aurocs(R, y_correct)
    R = np.asarray(R, dtype=float)
    y = np.asarray(y_correct, dtype=int).ravel()
    n = len(y)
    g = np.asarray(groups) if groups is not None else np.arange(n)
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    samples: dict[str, list[float]] = {k: [] for k in FEATURE_NAMES}
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(g, chosen)
        if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
            continue
        boot = component_aurocs(R[mask], y[mask])
        for k, v in boot.items():
            if v is not None and np.isfinite(v):
                samples[k].append(float(v))
    cis: dict[str, Any] = {}
    for k in FEATURE_NAMES:
        s = samples[k]
        if len(s) < 10:
            cis[k] = {"auroc": point.get(k), "ci95": None, "n_boot": len(s)}
        else:
            cis[k] = {
                "auroc": point.get(k),
                "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
                "n_boot": len(s),
            }
    return {
        "per_component": cis,
        "note": (
            "AUROC of raw R_* vs OOF correctness. A reliability channel whose "
            "grouped-bootstrap CI lies wholly below 0.5 is excluded if the "
            "declared set-selection rule would otherwise activate it."
        ),
    }


def _risk_coverage_area(confidence: np.ndarray, y_correct: np.ndarray) -> float | None:
    from rockburst.ml.train import _risk_coverage_curve

    return _risk_coverage_curve(confidence, y_correct).get("area")


def crossfit_r_final_for_mask(
    R: np.ndarray,
    y_correct: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    n_outer: int,
    conf_eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-safe R_final via fold stacker + full train-portion OOF calibrator."""
    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(y), -1, dtype=int)
    for fold, (_tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_of_row[te] = fold
    r_final = np.zeros(len(y), dtype=float)
    for fold in range(n_outer):
        train_mask = (fold_of_row >= 0) & (fold_of_row != fold)
        te_mask = fold_of_row == fold
        if not train_mask.any() or not te_mask.any():
            continue
        idx_fit = np.where(train_mask)[0]
        idx_te = np.where(te_mask)[0]
        fus = ReliabilityFusion()
        fus.conf_eps = conf_eps
        fus.fit(
            R[idx_fit], y_correct[idx_fit], feature_mask=mask, select_features=False
        )
        if not fus.is_fit:
            continue
        # RC3: calibrate on all non-test OOF raw scores (not a tiny inner slice).
        r_raw_fit = np.array([fus._raw_score(R[i]) for i in idx_fit])
        fus.fit_calibrator(r_raw_fit, y_correct[idx_fit])
        thr = select_decision_thresholds(
            np.array([fus.r_final(R[i]) for i in idx_fit]),
            y_correct[idx_fit],
        )
        fus.set_decision_thresholds(thr)
        for j in idx_te:
            r_final[j] = fus.r_final(R[j])
    return r_final, fold_of_row


def compare_fusion_sets(
    R: np.ndarray,
    y_correct: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    n_outer: int = 5,
    n_boot: int = 200,
    conf_eps: float = 1e-3,
) -> dict[str, Any]:
    """Repeat A/B/C/D on a final leakage-safe R matrix with bootstrap CIs (RM7).

    Selection rule unchanged: among sets whose Δ-area CI vs best contains 0,
    choose simplest (A ≺ B ≺ C ≺ D). Does not force multi-channel if A wins.
    """
    from rockburst.eval.metrics import expected_calibration_error, ece_with_ci
    from sklearn.metrics import brier_score_loss

    set_scores: dict[str, np.ndarray] = {}
    fold_of = None
    rows: dict[str, Any] = {}
    for set_id in SIMPLICITY_ORDER:
        mask = mask_for_set(set_id)
        r_final, fold_of = crossfit_r_final_for_mask(
            R,
            y_correct,
            y,
            groups,
            mask,
            seed=seed,
            n_outer=n_outer,
            conf_eps=conf_eps,
        )
        set_scores[set_id] = r_final
        held = fold_of >= 0
        area = _risk_coverage_area(r_final[held], y_correct[held])
        ece = expected_calibration_error(
            y_correct[held],
            np.column_stack([1 - r_final[held], r_final[held]]),
        )
        ece_ci = ece_with_ci(
            y_correct[held],
            r_final[held],
            groups=groups[held] if groups is not None else None,
            n_boot=min(n_boot, 100),
            seed=seed,
        )
        rows[set_id] = {
            "features": list(FUSION_SET_CANDIDATES[set_id]),
            "area": area,
            "ece": ece,
            "ece_ci": ece_ci,
            "brier": float(brier_score_loss(y_correct[held], r_final[held])),
            "mean_r_final": float(r_final[held].mean()),
            "calibration_n_protocol": "full_train_portion_oof_raw",
        }

    best_id = max(rows, key=lambda k: rows[k]["area"] if rows[k]["area"] is not None else -1)
    rng = np.random.default_rng(seed)
    held_idx = np.where(fold_of >= 0)[0]
    deltas: dict[str, list[float]] = {s: [] for s in SIMPLICITY_ORDER}
    for _ in range(n_boot):
        boot = rng.choice(held_idx, size=len(held_idx), replace=True)
        a_best = _risk_coverage_area(set_scores[best_id][boot], y_correct[boot])
        for s in SIMPLICITY_ORDER:
            a_s = _risk_coverage_area(set_scores[s][boot], y_correct[boot])
            if a_best is not None and a_s is not None:
                deltas[s].append(float(a_s - a_best))

    tied: list[str] = []
    for s in SIMPLICITY_ORDER:
        d = deltas[s]
        lo, hi = (
            (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
            if d
            else (0.0, 0.0)
        )
        rows[s]["delta_vs_best_ci"] = [lo, hi]
        if lo <= 0.0 <= hi:
            tied.append(s)
    chosen = best_id
    for s in SIMPLICITY_ORDER:
        if s in tied:
            chosen = s
            break

    mask_prod = mask_for_set(chosen)
    coef_cis = bootstrap_coef_cis(
        R[held_idx],
        y_correct[held_idx],
        n_boot=n_boot,
        seed=seed,
        feature_mask=mask_prod,
    )
    comp = component_auroc_bootstrap(
        R[held_idx], y_correct[held_idx], groups[held_idx], n_boot=n_boot, seed=seed
    )

    # Fold sensitivity of coefficients for the set selected by the declared rule.
    fold_coef_sens: list[dict[str, Any]] = []
    for fold in range(n_outer):
        tr = fold_of != fold
        if tr.sum() < 10 or len(np.unique(y_correct[tr])) < 2:
            continue
        fus = ReliabilityFusion()
        fus.fit(
            R[tr],
            y_correct[tr],
            feature_mask=mask_prod,
            select_features=False,
        )
        if fus.is_fit and fus.coef_ is not None:
            fold_coef_sens.append(
                {
                    "fold": fold,
                    "coef": fus.coef_.tolist(),
                    "intercept": float(fus.intercept_),
                }
            )

    return {
        "sets": rows,
        "best_by_area": best_id,
        "tied_with_best": tied,
        "chosen_set": chosen,
        # Compatibility key retained, but now reflects the evaluated rule.
        "locked_production_set": chosen,
        "production_set_from_rule": chosen,
        "legacy_default_set": CHOSEN_FUSION_SET,
        "selection_rule": (
            "Among sets whose Δ-area CI vs best contains 0, choose simplest "
            f"(order {list(SIMPLICITY_ORDER)}). Keep A if it still wins (RM7)."
        ),
        "agents_add_calibrated_reliability_signal": chosen != "A",
        "finding": (
            "Set A (r_model only) wins or ties under the pre-registered rule; "
            "multi-channel fusion not forced when it does not improve RC."
            if chosen == "A"
            else f"Chosen set {chosen}; multi-channel reliability contributes."
        ),
        "coef_bootstrap_cis": coef_cis,
        "component_auroc": comp,
        "fold_coef_sensitivity": fold_coef_sens,
        "component_constants_justification": (
            f"Production set {chosen} is selected by the declared risk-coverage rule; "
            "r_data/r_physics/r_uncertainty/r_stability are reported via "
            "component AUROC + fold coef sensitivity but do not enter R_final "
            "unless set compare selects B/C/D. Decision-band utilities are "
            "engineering-cost justified in stacker.DECISION_BAND_UTILITIES."
        ),
    }


def production_fusion_from_oof(
    R: np.ndarray,
    y_correct: np.ndarray,
    fold_of_row: np.ndarray,
    *,
    conf_eps: float = 1e-3,
    feature_mask: np.ndarray | None = None,
    method: str | None = None,
    comparison: dict[str, Any] | None = None,
    interval_boot: int = 100,
    interval_groups: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[ReliabilityFusion, dict[str, Any]]:
    """Fit one serving stacker and calibrate the exact scores it emits."""
    selection_report: dict[str, Any]
    if comparison is not None:
        mask, selection_report = production_mask_from_comparison(comparison)
    else:
        mask = feature_mask if feature_mask is not None else locked_feature_mask()
        selection_report = {
            "selected_set": "explicit_mask" if feature_mask is not None else CHOSEN_FUSION_SET,
            "active_features": [FEATURE_NAMES[i] for i, active in enumerate(mask) if active],
            "selection_rule": "explicit mask / legacy fallback",
        }
    fusion = ReliabilityFusion()
    fusion.conf_eps = conf_eps
    fusion.fit(R, y_correct, feature_mask=mask, select_features=False)
    # Diagnostic fold-stacker scores are retained, but never attached to the
    # production stacker. The serving calibrator sees production raw scores.
    diagnostic = ReliabilityFusion()
    diagnostic.conf_eps = conf_eps
    r_raw_crossfit, _ = fit_calibrator_on_crossfit_raw(
        diagnostic, R, y_correct, fold_of_row, feature_mask=mask,
        conf_eps=conf_eps, method=method,
    )
    r_raw_oof = np.array([fusion._raw_score(row) for row in np.asarray(R)])
    fusion.fit_calibrator(r_raw_oof, y_correct, method=method)
    r_final_oof = np.array([fusion.r_final(row) for row in np.asarray(R)])
    held = fold_of_row >= 0
    if held.any() and fusion.calibrator is not None:
        thr = select_decision_thresholds(r_final_oof[held], y_correct[held])
        fusion.set_decision_thresholds(thr)
    fusion.fit_bootstrap_interval(
        R[held],
        y_correct[held],
        groups=(
            np.asarray(interval_groups)[held]
            if interval_groups is not None
            else fold_of_row[held]
        ),
        n_boot=interval_boot,
        seed=seed, method=method,
    )
    occupancy = decision_band_occupancy(r_final_oof[held], fusion.decision_thresholds)
    sensitivity = decision_utility_sensitivity(r_final_oof[held], y_correct[held])
    meta = {
        "calibration_n": int(fusion.calibration_n),
        "calibration_method": fusion.calibration_method,
        "calibration_protocol": "production_stacker_raw_oof_single_serve_path",
        "served_r_final_oof": r_final_oof.tolist(),
        "served_raw_oof": r_raw_oof.tolist(),
        "diagnostic_crossfit_raw_oof": r_raw_crossfit.tolist(),
        "decision_thresholds": dict(fusion.decision_thresholds),
        "band_occupancy": occupancy,
        "utility_sensitivity": sensitivity,
        "interval_method": "grouped_bootstrap_stacker_plus_calibrator",
        "interval_members": len(fusion.interval_members),
        "feature_selection": selection_report,
        "threshold_bands_ordered": True,
        "all_bands_occupied": all(v > 0 for v in occupancy.values()),
    }
    return fusion, meta
