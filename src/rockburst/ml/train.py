"""Train baselines with nested grouped CV, split conformal, and live-R fusion."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.schemas import PredictiveOutput
from rockburst.agents.tools import run_data_quality, run_physics, run_uncertainty
from rockburst.config import resolve_path
from rockburst.eval.metrics import (
    capacity_matched_family_table,
    classification_report_dict,
    quadratic_weighted_kappa,
)
from rockburst.eval.run_meta import build_run_metadata
from rockburst.fusion.features import compute_r_model, compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, select_decision_thresholds
from rockburst.ml.ordinal import argmax_grade, median_of_distribution
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.tuning import (
    DEFAULT_N_TRIALS,
    SEARCH_SPACES_DOC,
    available_model_names,
    estimator_from_params,
    nested_optuna_select,
)
from rockburst.physics.relations import FEATURE_COLS
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    ConformalState,
    compare_conformal_methods,
    conformal_prediction_set,
    coverage_and_size,
    coverage_size_abstention_tradeoff,
    fit_class_conditional_alpha,
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
    pick_defensible_method,
    set_size_distribution,
)


def _repo_relative_artifact(path: Path, cfg: dict[str, Any]) -> str:
    """Serialize generated artefact paths relative to the repository root."""
    root = Path(cfg.get("_root", ".")).resolve()
    return path.resolve().relative_to(root).as_posix()


def _predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] < 4:
            pad = np.zeros((proba.shape[0], 4))
            pad[:, : proba.shape[1]] = proba
            return pad
        return proba
    pred = model.predict(X).astype(int)
    proba = np.zeros((len(pred), 4))
    proba[np.arange(len(pred)), np.clip(pred, 0, 3)] = 1.0
    return proba


def _fit_required_candidate(
    pipe: Any, X_fit: Any, y_fit: Any, *, candidate: str, fold: int
) -> Any:
    """Fit a required candidate or stop; failed fits are never scored."""
    try:
        return pipe.fit(X_fit, y_fit)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"required candidate {candidate!r} failed in outer fold {fold}"
        ) from exc


def _split_fit_calib(
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one group-stratified fold of ``tr`` as conformal calibration."""
    y_tr, g_tr = y[tr], groups[tr]
    n_unique = len(np.unique(g_tr))
    n_sp = min(n_splits, max(2, n_unique))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=seed)
        for fit_rel, calib_rel in sgkf.split(np.zeros(len(tr)), y_tr, g_tr):
            return tr[fit_rel], tr[calib_rel]
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tr))
        cut = max(1, int(0.2 * len(tr)))
        return tr[perm[cut:]], tr[perm[:cut]]
    return tr, tr[:0]


def _mean_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "qwk",
        "ordinal_mae",
        "severe_misclassification_rate",
        "brier",
        "ece",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [f[k] for f in fold_metrics if isinstance(f.get(k), (int, float))]
        out[k] = float(np.mean(vals)) if vals else None
    cov = [f["conformal"]["coverage"] for f in fold_metrics if f.get("conformal", {}).get("coverage") is not None]
    sz = [f["conformal"]["mean_size"] for f in fold_metrics if f.get("conformal", {}).get("mean_size") is not None]
    out["prediction_set_coverage"] = float(np.mean(cov)) if cov else None
    out["prediction_set_mean_size"] = float(np.mean(sz)) if sz else None
    return out


def _fit_conformal(
    method: str,
    y_calib: np.ndarray,
    p_cal: np.ndarray,
    alpha: float,
) -> ConformalState:
    if method == "mondrian_split":
        return fit_mondrian_thresholds(y_calib, p_cal, alpha=alpha)
    if method == "class_conditional_grade3":
        return fit_class_conditional_alpha(y_calib, p_cal, alpha=alpha)
    return fit_marginal_thresholds(y_calib, p_cal, alpha=alpha)


def fit_fold_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    ranked: list[tuple[str, dict[str, Any]]],
    *,
    seed: int,
    fold: int,
    top_k: int = 3,
) -> dict[str, Any]:
    """Fit top-k ensemble members on fold fit indices only (no full-data reuse)."""
    ensemble: dict[str, Any] = {}
    for key, meta in ranked[:top_k]:
        mname, mimp = key.split("__", 1)
        fold_params_list = meta.get("fold_params") or []
        if fold < len(fold_params_list) and fold_params_list[fold]:
            params = fold_params_list[fold]
        else:
            params = meta.get("best_params") or {}
        mest = estimator_from_params(mname, params, seed + fold)
        pipe = Pipeline(
            [("prep", build_preprocess(mimp)), ("model", clone(mest))]
        )
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        ensemble[key] = pipe
    return ensemble


def _risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> dict[str, Any]:
    """Accuracy when keeping top-confidence fraction; area under curve."""
    if len(confidence) == 0:
        return {"points": [], "area": None}
    order = np.argsort(-np.asarray(confidence, dtype=float))
    corr = np.asarray(correct, dtype=float)[order]
    coverages = np.linspace(0.1, 1.0, 10)
    points = []
    accs = []
    for keep in coverages:
        n_keep = max(1, int(round(keep * len(corr))))
        acc = float(corr[:n_keep].mean())
        points.append({"coverage": float(keep), "accuracy": acc})
        accs.append(acc)
    return {"points": points, "area": float(np.trapezoid(accs, coverages))}


def _live_r_for_case(
    case: dict[str, Any],
    *,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    pipe: Any,
    proba: np.ndarray,
    cfg: dict[str, Any],
    conformal: ConformalState | None,
    ensemble: dict[str, Any] | None = None,
) -> np.ndarray:
    ref = build_fold_reference(df, train_idx, case, group_col=cfg.get("group_col", "dup_group_id"))
    dq = run_data_quality(case, ref, cfg)
    phys = run_physics(case, cfg)
    proba_arr = np.asarray(proba, dtype=float)
    observed_mask = {
        c: bool(np.isfinite(float(case[c]))) if case.get(c) is not None else False
        for c in FEATURE_COLS
    }
    decode = str(cfg.get("decode_rule") or "argmax")
    if decode == "median":
        grade = int(median_of_distribution(proba_arr)[0])
    else:
        grade = int(np.argmax(proba_arr))
    pred = PredictiveOutput(
        predicted_grade=grade,
        probabilities=proba_arr.tolist(),
        r_model=compute_r_model(proba_arr),
        observed_mask=observed_mask,
        model_name="oof",
    )
    unc = run_uncertainty(
        case,
        pred,
        cfg,
        pipeline=pipe,
        ensemble=ensemble,
        conformal=conformal,
    )
    return compute_reliability_features(dq, phys, pred, unc, case=case)


def train_baselines(cfg: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run: rockburst extract")
    df = pd.read_csv(path)
    X, y, groups = prepare_xy(df)
    n_outer = 3 if quick else int(cfg["cv"]["n_outer"])
    n_inner = 2 if quick else int(cfg["cv"].get("n_inner", 3))
    n_trials = 3 if quick else int(cfg.get("tuning", {}).get("n_trials", DEFAULT_N_TRIALS))
    alpha = float(cfg.get("uncertainty", {}).get("conformal_alpha", 0.1))
    imputer_kinds = ["median", "knn"] if quick else ["median", "knn", "iterative"]
    model_names = available_model_names(quick=quick)

    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    out_dir = resolve_path(cfg, "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = resolve_path(cfg, "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "seed": seed,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "n_trials": n_trials,
        "quick": bool(quick),
        "mode": "quick" if quick else "full",
        "run_metadata": build_run_metadata(
            mode="quick" if quick else "full",
            command="rockburst train" + (" --quick" if quick else ""),
            extra={"n_outer": n_outer, "n_inner": n_inner, "n_trials": n_trials},
            config=cfg,
            input_paths=[path],
        ),
        "search_spaces": SEARCH_SPACES_DOC,
        "models": {},
        "complete_case_n": int(len(df.dropna(subset=FEATURE_COLS))),
        "d_missing_n": int(df["d"].isna().sum()),
        "conformal": {"alpha": alpha},
        "protocol_note": (
            "selection_biased: argmax pooled outer-OOF QWK over model×imputer. "
            "nested_unbiased: per outer fold pick model×imputer by inner QWK only. "
            "production (M10): model×imputer with best mean inner_qwk across outer folds "
            "(not mode-of-winners); fold-winner instability reported separately. "
            "pooled_oof vs mean_fold are both reported for every candidate. "
            "Decoding (RM10): argmax vs median compared by mean fold QWK on the "
            "same OOF; production decode_rule adopted only if it does not regress "
            "mean-inner selection. Capacity-matched multiclass vs ordinal in artefact."
        ),
    }

    fold_splits = list(sgkf.split(X, y, groups))
    # Store per-fold artefacts for nested selection
    fold_store: dict[str, list[dict[str, Any]]] = {}

    for imp in imputer_kinds:
        for name in model_names:
            key = f"{name}__{imp}"
            oof_pred = np.zeros(len(y), dtype=int)
            oof_proba = np.zeros((len(y), 4), dtype=float)
            fold_metrics = []
            fold_coverage = []
            fold_params_list: list[dict[str, Any]] = []
            fold_inner: list[float] = []
            fold_store[key] = []

            import hashlib
            from pathlib import Path as _Path

            _cfg_hash = hashlib.sha256(
                json.dumps(
                    {
                        "near_dup_rel_tol": cfg.get("quality", {}).get(
                            "near_dup_rel_tol", 0.01
                        ),
                        "n_trials": n_trials,
                        "quick": quick,
                        "seed": seed,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]
            _cache_dir = _Path(cfg.get("_root", ".")) / "artifacts" / "tuning_cache"

            for fold, (tr, te) in enumerate(fold_splits):
                est, params, iq = nested_optuna_select(
                    X.iloc[tr],
                    y[tr],
                    groups[tr],
                    model_name=name,
                    imputer_kind=imp,
                    n_inner=min(n_inner, max(2, len(np.unique(groups[tr])) // 2)),
                    n_trials=n_trials,
                    seed=seed + fold,
                    cache_dir=_cache_dir,
                    near_dup_rel_tol=float(
                        cfg.get("quality", {}).get("near_dup_rel_tol", 0.01)
                    ),
                    config_hash=f"{_cfg_hash}:fold{fold}",
                )
                fold_params_list.append(params)
                fold_inner.append(float(iq))
                fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + fold)
                # Group disjointness check (exchangeability precondition)
                assert set(groups[fit_idx]).isdisjoint(set(groups[calib_idx]))
                assert set(groups[fit_idx]).isdisjoint(set(groups[te]))
                assert set(groups[calib_idx]).isdisjoint(set(groups[te]))

                pipe = Pipeline(
                    [("prep", build_preprocess(imp)), ("model", clone(est))]
                )
                _fit_required_candidate(
                    pipe,
                    X.iloc[fit_idx],
                    y[fit_idx],
                    candidate=key,
                    fold=fold,
                )
                conf: ConformalState | None = None
                p_cal = None
                if len(calib_idx) > 0:
                    p_cal = _predict_proba_safe(pipe, X.iloc[calib_idx])
                    conf = fit_marginal_thresholds(y[calib_idx], p_cal, alpha=alpha)
                    conf.n_fit = int(len(fit_idx))
                proba = _predict_proba_safe(pipe, X.iloc[te])
                pred = argmax_grade(proba)
                med_fold = median_of_distribution(proba)
                oof_pred[te] = pred
                oof_proba[te] = proba
                sets = [
                    conformal_prediction_set(
                        proba[i],
                        alpha=alpha,
                        class_thresholds=None if conf is None else conf.class_thresholds,
                        contiguous=True,
                    )
                    for i in range(len(te))
                ]
                cov = coverage_and_size(y[te], sets)
                size_dist = set_size_distribution(y[te], sets, fold=fold)
                fold_coverage.append(cov)
                fm = {
                    "fold": fold,
                    **classification_report_dict(y[te], pred, proba),
                    "median_decode_qwk": float(
                        quadratic_weighted_kappa(y[te], med_fold)
                    ),
                    "conformal": cov,
                    "set_size_distribution": size_dist,
                    "n_fit": int(len(fit_idx)),
                    "n_calib": int(len(calib_idx)),
                    "n_test": int(len(te)),
                    "inner_qwk": float(iq),
                }
                fold_metrics.append(fm)
                fold_store[key].append(
                    {
                        "fold": fold,
                        "inner_qwk": float(iq),
                        "params": params,
                        "oof_pred": pred.tolist(),
                        "oof_proba": proba.tolist(),
                        "te": te.tolist(),
                        "fit_idx": fit_idx.tolist(),
                        "calib_idx": calib_idx.tolist(),
                        "y_calib": y[calib_idx].tolist() if len(calib_idx) else [],
                        "p_calib": p_cal.tolist() if p_cal is not None else [],
                        "y_te": y[te].tolist(),
                    }
                )

            pooled = classification_report_dict(y, oof_pred, oof_proba)
            # M8: median-of-distribution vs argmax on the same OOF probabilities
            med_pred = median_of_distribution(oof_proba)
            pooled_median = classification_report_dict(y, med_pred, oof_proba)
            cov_vals = [c["coverage"] for c in fold_coverage if c.get("coverage") is not None]
            size_vals = [c["mean_size"] for c in fold_coverage if c.get("mean_size") is not None]
            pooled["prediction_set_coverage"] = float(np.mean(cov_vals)) if cov_vals else None
            pooled["prediction_set_mean_size"] = float(np.mean(size_vals)) if size_vals else None
            mean_fold = _mean_fold_metrics(fold_metrics)
            med_fold_qwks = [
                f["median_decode_qwk"]
                for f in fold_metrics
                if isinstance(f.get("median_decode_qwk"), (int, float))
            ]
            results["models"][key] = {
                "pooled_oof": pooled,
                "pooled_oof_median_decode": pooled_median,
                "decode_comparison": {
                    "argmax_qwk": pooled.get("qwk"),
                    "median_qwk": pooled_median.get("qwk"),
                    "argmax_ordinal_mae": pooled.get("ordinal_mae"),
                    "median_ordinal_mae": pooled_median.get("ordinal_mae"),
                    "mean_fold_argmax_qwk": mean_fold.get("qwk"),
                    "mean_fold_median_qwk": (
                        float(np.mean(med_fold_qwks)) if med_fold_qwks else None
                    ),
                },
                "mean_fold": mean_fold,
                "overall": pooled,  # backward-compat alias = pooled
                "folds": fold_metrics,
                "best_params": fold_params_list[-1] if fold_params_list else {},
                "fold_params": fold_params_list,
                "inner_qwk_mean": float(np.nanmean(fold_inner)) if fold_inner else None,
                # RM8 / F12 schema: per-fold set-size distributions
                "set_size_distributions": [
                    f.get("set_size_distribution")
                    for f in fold_metrics
                    if f.get("set_size_distribution")
                ],
            }

    # --- Selection-biased: argmax pooled outer OOF QWK ---
    candidate_table = []
    for key, meta in results["models"].items():
        fold_qwks = [
            f.get("qwk")
            for f in (meta.get("folds") or [])
            if isinstance(f.get("qwk"), (int, float))
        ]
        qwk_ci = None
        if len(fold_qwks) >= 2:
            arr = np.asarray(fold_qwks, dtype=float)
            qwk_ci = [
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            ]
        candidate_table.append(
            {
                "model": key,
                "pooled_oof_qwk": meta["pooled_oof"].get("qwk"),
                "pooled_oof_accuracy": meta["pooled_oof"].get("accuracy"),
                "mean_fold_qwk": meta["mean_fold"].get("qwk"),
                "mean_fold_qwk_ci95": qwk_ci,
                "mean_fold_accuracy": meta["mean_fold"].get("accuracy"),
                "inner_qwk_mean": meta.get("inner_qwk_mean"),
                "is_ordinal_family": key.startswith(("frank_hall", "ordinal_logistic")),
            }
        )
    candidate_table.sort(key=lambda r: r["pooled_oof_qwk"] or -1, reverse=True)
    results["candidate_table"] = candidate_table
    results["candidate_table_note"] = (
        "F8: all model×imputer candidates with mean_fold_qwk_ci95 across outer folds."
    )

    biased_name = candidate_table[0]["model"] if candidate_table else None
    results["selection_biased"] = {
        "best_model": biased_name,
        "pooled_oof": results["models"][biased_name]["pooled_oof"] if biased_name else None,
        "mean_fold": results["models"][biased_name]["mean_fold"] if biased_name else None,
        "note": "Optimistic: model×imputer chosen using the same outer OOF scores reported.",
    }

    # --- Nested unbiased: per outer fold pick by inner QWK only ---
    nested_oof_pred = np.zeros(len(y), dtype=int)
    nested_oof_proba = np.zeros((len(y), 4), dtype=float)
    nested_fold_winners: list[str] = []
    nested_fold_metrics: list[dict[str, Any]] = []
    for fold in range(n_outer):
        best_key = None
        best_iq = -1e9
        for key, folds in fold_store.items():
            if fold >= len(folds):
                continue
            iq = folds[fold].get("inner_qwk")
            if iq is None or folds[fold].get("error"):
                continue
            if float(iq) > best_iq:
                best_iq = float(iq)
                best_key = key
        if best_key is None:
            continue
        nested_fold_winners.append(best_key)
        block = fold_store[best_key][fold]
        te = np.asarray(block["te"], dtype=int)
        nested_oof_pred[te] = np.asarray(block["oof_pred"], dtype=int)
        nested_oof_proba[te] = np.asarray(block["oof_proba"], dtype=float)
        nested_fold_metrics.append(
            {
                "fold": fold,
                "winner": best_key,
                "inner_qwk": best_iq,
                **classification_report_dict(
                    y[te], nested_oof_pred[te], nested_oof_proba[te]
                ),
            }
        )

    nested_pooled = classification_report_dict(y, nested_oof_pred, nested_oof_proba)
    winner_counts = Counter(nested_fold_winners)
    mode_winner = (
        winner_counts.most_common(1)[0][0] if nested_fold_winners else biased_name
    )
    mode_count = int(winner_counts.most_common(1)[0][1]) if nested_fold_winners else 0

    # M10: production = best mean inner_qwk across folds (pre-stated), not mode-of-winners
    prod_by_mean_inner = None
    best_mean_iq = -1e9
    for key, meta in results["models"].items():
        iq_m = meta.get("inner_qwk_mean")
        if iq_m is None or not np.isfinite(float(iq_m)):
            continue
        if float(iq_m) > best_mean_iq:
            best_mean_iq = float(iq_m)
            prod_by_mean_inner = key

    results["nested_unbiased"] = {
        "pooled_oof": nested_pooled,
        "mean_fold": _mean_fold_metrics(
            [{**m, "conformal": {}} for m in nested_fold_metrics]
        ),
        "fold_winners": nested_fold_winners,
        "mode_of_winners": mode_winner,
        "production_model": prod_by_mean_inner or mode_winner,
        "note": (
            "Unbiased OOF: per outer fold pick by inner QWK. "
            "Production artefact uses best mean inner_qwk (M10), not mode-of-winners."
        ),
    }
    results["fold_winner_instability"] = {
        "fold_winners": nested_fold_winners,
        "unique_winners": len(set(nested_fold_winners)),
        "mode_of_winners": mode_winner,
        "mode_count": mode_count,
        "mode_fraction": (
            float(mode_count) / len(nested_fold_winners) if nested_fold_winners else None
        ),
        "winner_counts": dict(winner_counts),
        "production_rule": "best_mean_inner_qwk",
        "production_model": prod_by_mean_inner,
        "production_inner_qwk_mean": best_mean_iq if prod_by_mean_inner else None,
        "mode_differs_from_production": bool(
            mode_winner and prod_by_mean_inner and mode_winner != prod_by_mean_inner
        ),
    }

    # Headline / production model = best mean inner (M10); keep biased for comparison
    best_name = prod_by_mean_inner or mode_winner or biased_name
    results["best_model"] = best_name
    results["best_qwk"] = (
        results["models"][best_name]["pooled_oof"].get("qwk")
        if best_name and best_name in results["models"]
        else (nested_pooled.get("qwk") if nested_pooled else None)
    )
    results["best_qwk_selection_biased"] = (
        results["selection_biased"]["pooled_oof"]["qwk"] if biased_name else None
    )
    results["best_qwk_nested_unbiased"] = nested_pooled.get("qwk")
    # M8 / RM10 summary table across all candidates
    results["ordinal_decode_table"] = [
        {
            "model": key,
            **(meta.get("decode_comparison") or {}),
            "is_frank_hall": key.startswith("frank_hall"),
            "is_ordinal_logistic": key.startswith("ordinal_logistic"),
        }
        for key, meta in results["models"].items()
    ]
    # RM10: rebuild pooled OOF preds from fold_store for delta-QWK bootstrap CIs.
    oof_pred_by_model: dict[str, np.ndarray] = {}
    for key, folds in fold_store.items():
        pred = np.full(len(y), -1, dtype=int)
        ok = True
        for block in folds:
            if "oof_pred" not in block or "te" not in block:
                ok = False
                break
            te = np.asarray(block["te"], dtype=int)
            pred[te] = np.asarray(block["oof_pred"], dtype=int)
        if ok and len(pred) and bool((pred >= 0).all()):
            oof_pred_by_model[key] = pred
    results["capacity_matched_multiclass_vs_ordinal"] = capacity_matched_family_table(
        results["models"],
        oof_pred_by_model=oof_pred_by_model,
        y_true=y,
        groups=groups,
        n_boot=50 if quick else 200,
        seed=seed,
    )

    # RM10: adopt median or justify argmax using mean-fold QWK (same spirit as
    # mean-inner selection — no silent M1 regression).
    decode_rule = "argmax"
    decode_justification = (
        "Default argmax retained pending production-model fold comparison."
    )
    if best_name and best_name in results["models"]:
        dc = results["models"][best_name].get("decode_comparison") or {}
        arg_mf = dc.get("mean_fold_argmax_qwk")
        med_mf = dc.get("mean_fold_median_qwk")
        if (
            arg_mf is not None
            and med_mf is not None
            and float(med_mf) > float(arg_mf) + 1e-6
        ):
            decode_rule = "median"
            decode_justification = (
                f"Adopted median decode: mean_fold QWK {med_mf:.4f} > "
                f"argmax {arg_mf:.4f} on production model {best_name} "
                "(RM10; selection still best mean inner QWK)."
            )
        elif arg_mf is not None and med_mf is not None:
            decode_rule = "argmax"
            decode_justification = (
                f"Retained argmax: mean_fold QWK {arg_mf:.4f} ≥ median "
                f"{med_mf:.4f} on {best_name}; avoids silent M1 regression (RM10)."
            )
    results["decode_rule"] = decode_rule
    results["decode_rule_justification"] = decode_justification
    best_params = (
        results["models"][best_name]["fold_params"][0]
        if best_name and results["models"].get(best_name, {}).get("fold_params")
        else {}
    )
    # Prefer last fold params of production model for refit artefact
    if best_name and results["models"].get(best_name, {}).get("fold_params"):
        best_params = results["models"][best_name]["fold_params"][-1]
    results["best_params"] = best_params
    best_fold_params = (
        results["models"][best_name]["fold_params"] if best_name else [{}] * n_outer
    )
    best_imp = best_name.split("__", 1)[1] if best_name else "median"
    name = best_name.split("__", 1)[0] if best_name else "logistic"

    # --- Conformal method comparison on selection-biased best (same folds) ---
    if biased_name:
        method_names = (
            "mondrian_split",
            "marginal_split",
            "cv_plus_grouped",
            "ordinal_split",
            "class_conditional_grade3",
        )
        method_cov: dict[str, list[float]] = {k: [] for k in method_names}
        method_size: dict[str, list[float]] = {k: [] for k in method_names}
        method_frac_large: dict[str, list[float]] = {k: [] for k in method_names}
        per_class_cov: dict[str, list[dict]] = {k: [] for k in method_names}
        set_size_by_fold: list[dict[str, Any]] = []
        all_calib_scores: list[float] = []
        fold_comparisons = []
        for fold_block in fold_store[biased_name]:
            if fold_block.get("error") or not fold_block.get("p_calib"):
                continue
            y_cal = np.asarray(fold_block["y_calib"], dtype=int)
            p_cal = np.asarray(fold_block["p_calib"], dtype=float)
            y_te = np.asarray(fold_block["y_te"], dtype=int)
            p_te = np.asarray(fold_block["oof_proba"], dtype=float)
            scores = 1.0 - p_cal[np.arange(len(y_cal)), np.clip(y_cal, 0, 3)]
            all_calib_scores.extend(scores.tolist())
            cmp_ = compare_conformal_methods(
                y_cal,
                p_cal,
                y_te,
                p_te,
                alpha=alpha,
                cv_plus_scores=np.asarray(all_calib_scores, dtype=float),
            )
            fold_comparisons.append(cmp_)
            for m in method_names:
                if m in cmp_ and cmp_[m].get("coverage") is not None:
                    method_cov[m].append(float(cmp_[m]["coverage"]))
                    method_size[m].append(float(cmp_[m]["mean_size"]))
                    per_class_cov[m].append(cmp_[m].get("per_class", {}))
                    if cmp_[m].get("frac_size_ge_large") is not None:
                        method_frac_large[m].append(float(cmp_[m]["frac_size_ge_large"]))
            # Production-method set-size dist for this fold (filled after pick)
            set_size_by_fold.append(
                {
                    "fold": fold_block.get("fold"),
                    "methods": {
                        m: cmp_[m].get("set_size_distribution")
                        for m in method_names
                        if m in cmp_
                    },
                }
            )

        # Aggregate comparison (include per-class for grade-3 Mondrian decision)
        agg_cmp: dict[str, Any] = {}
        for m in method_names:
            if not method_cov[m]:
                continue
            # Merge per-class coverage across folds (mean)
            per_class_agg: dict[str, Any] = {}
            for pc in per_class_cov[m]:
                for c, block in (pc or {}).items():
                    per_class_agg.setdefault(c, {"coverage": [], "mean_size": [], "n": []})
                    if block.get("coverage") is not None:
                        per_class_agg[c]["coverage"].append(float(block["coverage"]))
                    if block.get("mean_size") is not None:
                        per_class_agg[c]["mean_size"].append(float(block["mean_size"]))
                    if block.get("n") is not None:
                        per_class_agg[c]["n"].append(int(block["n"]))
            per_class_out = {
                c: {
                    "coverage": float(np.mean(v["coverage"])) if v["coverage"] else None,
                    "mean_size": float(np.mean(v["mean_size"])) if v["mean_size"] else None,
                    "n_mean": float(np.mean(v["n"])) if v["n"] else None,
                }
                for c, v in per_class_agg.items()
            }
            agg_cmp[m] = {
                "coverage": float(np.mean(method_cov[m])),
                "mean_size": float(np.mean(method_size[m])),
                "coverage_std": float(np.std(method_cov[m], ddof=1)) if len(method_cov[m]) > 1 else 0.0,
                "frac_size_ge_large": (
                    float(np.mean(method_frac_large[m])) if method_frac_large[m] else None
                ),
                "per_class": per_class_out,
            }
        class_counts_ex = (
            fold_comparisons[0].get("class_calib_counts") if fold_comparisons else []
        )
        pick_payload = {
            **agg_cmp,
            "class_calib_counts": class_counts_ex,
        }
        chosen = pick_defensible_method(pick_payload, alpha=alpha)
        tradeoff = coverage_size_abstention_tradeoff(agg_cmp, alpha=alpha)
        results["conformal"] = {
            "alpha": alpha,
            "method": chosen,
            "comparison": agg_cmp,
            "coverage_size_abstention_tradeoff": tradeoff,
            "set_size_distribution_by_fold": set_size_by_fold,
            "class_calib_counts_example": class_counts_ex,
            "contiguous_ordinal_sets": True,
            "selection_note": (
                "M11/RM8: among methods meeting overall and grade-3 coverage, "
                "choose lowest wide-set abstention then smallest mean set; "
                "compare marginal, CV+, ordinal nonconformity, class-conditional "
                "grade-3 α and report the coverage–size–abstention trade-off. "
                "Prediction sets forced contiguous on the ordinal grade scale."
            ),
        }
        conformal_method = chosen
    else:
        conformal_method = "mondrian_split"

    # Recompute biased & nested headline coverage with chosen method
    def _recompute_coverage(key: str) -> dict[str, Any]:
        covs, sizes = [], []
        per_rep: list[dict] = []
        for fold_block in fold_store.get(key, []):
            if fold_block.get("error") or not fold_block.get("p_calib"):
                continue
            conf = _fit_conformal(
                conformal_method,
                np.asarray(fold_block["y_calib"], dtype=int),
                np.asarray(fold_block["p_calib"], dtype=float),
                alpha,
            )
            sets = [
                conformal_prediction_set(
                    np.asarray(fold_block["oof_proba"][i], dtype=float),
                    alpha=alpha,
                    class_thresholds=conf.class_thresholds,
                )
                for i in range(len(fold_block["y_te"]))
            ]
            cov = coverage_and_size(np.asarray(fold_block["y_te"], dtype=int), sets)
            covs.append(cov["coverage"])
            sizes.append(cov["mean_size"])
            per_rep.append(cov.get("per_class", {}))
        return {
            "coverage": float(np.mean(covs)) if covs else None,
            "mean_size": float(np.mean(sizes)) if sizes else None,
            "per_class_fold_examples": per_rep[:3],
            "method": conformal_method,
        }

    if biased_name:
        results["selection_biased"]["conformal"] = _recompute_coverage(biased_name)
        results["models"][biased_name]["pooled_oof"]["prediction_set_coverage"] = results[
            "selection_biased"
        ]["conformal"]["coverage"]
        results["models"][biased_name]["pooled_oof"]["prediction_set_mean_size"] = results[
            "selection_biased"
        ]["conformal"]["mean_size"]

    if best_name:
        est = estimator_from_params(name, best_params, seed)
        fit_idx, calib_idx = _split_fit_calib(y, groups, np.arange(len(y)), seed=seed)
        best_pipe = Pipeline(
            [("prep", build_preprocess(best_imp)), ("model", clone(est))]
        )
        best_pipe.fit(X.iloc[fit_idx], y[fit_idx])
        conf_final = _fit_conformal(
            conformal_method,
            y[calib_idx],
            _predict_proba_safe(best_pipe, X.iloc[calib_idx]),
            alpha,
        )
        conf_final.n_fit = int(len(fit_idx))
        prod_pipe = Pipeline(
            [("prep", build_preprocess(best_imp)), ("model", clone(est))]
        )
        prod_pipe.fit(X, y)
        model_path = models_dir / "best_baseline.joblib"
        joblib.dump(
            {
                "pipeline": prod_pipe,
                "name": best_name,
                "features": FEATURE_COLS,
                "conformal": conf_final.to_dict(),
                "params": best_params,
                "decode_rule": decode_rule,
                "fit_idx_for_conformal_note": (
                    "Thresholds fitted on calib holdout; production pipe refit on all rows."
                ),
            },
            model_path,
        )
        results["model_path"] = _repo_relative_artifact(model_path, cfg)
        results["conformal_state"] = conf_final.to_dict()

        ensemble = {}
        ranked = sorted(
            results["models"].items(),
            key=lambda kv: kv[1]["pooled_oof"].get("qwk", -1),
            reverse=True,
        )
        for key, meta in ranked[:3]:
            mname, mimp = key.split("__", 1)
            mest = estimator_from_params(mname, meta.get("best_params") or {}, seed)
            pipe = Pipeline(
                [("prep", build_preprocess(mimp)), ("model", clone(mest))]
            )
            pipe.fit(X, y)
            ensemble[key] = pipe
        ens_path = models_dir / "ensemble.joblib"
        joblib.dump(ensemble, ens_path)
        results["ensemble_path"] = _repo_relative_artifact(ens_path, cfg)

        # Live-agent OOF R using production model family fold params
        from rockburst.fusion.stacker import FEATURE_NAMES, bootstrap_coef_cis

        n_feat = len(FEATURE_NAMES)
        R = np.zeros((len(y), n_feat), dtype=float)
        live_cfg = {
            **cfg,
            "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4 if quick else 8},
        }
        best_oof_pred = np.zeros(len(y), dtype=int)
        best_oof_proba = np.zeros((len(y), 4), dtype=float)
        fold_of_row = np.full(len(y), -1, dtype=int)
        for fold, (tr, te) in enumerate(fold_splits):
            params = best_fold_params[fold] if fold < len(best_fold_params) else best_params
            est_f = estimator_from_params(name, params, seed + fold)
            fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + fold)
            pipe = Pipeline(
                [("prep", build_preprocess(best_imp)), ("model", clone(est_f))]
            )
            pipe.fit(X.iloc[fit_idx], y[fit_idx])
            conf = None
            if len(calib_idx):
                conf = _fit_conformal(
                    conformal_method,
                    y[calib_idx],
                    _predict_proba_safe(pipe, X.iloc[calib_idx]),
                    alpha,
                )
                conf.n_fit = int(len(fit_idx))
            proba_te = _predict_proba_safe(pipe, X.iloc[te])
            if decode_rule == "median":
                best_oof_pred[te] = median_of_distribution(proba_te)
            else:
                best_oof_pred[te] = proba_te.argmax(axis=1)
            best_oof_proba[te] = proba_te
            fold_of_row[te] = fold
            fold_ensemble = fit_fold_ensemble(
                X, y, fit_idx, ranked, seed=seed, fold=fold, top_k=3
            )
            live_cfg_fold = {**live_cfg, "decode_rule": decode_rule}
            for k, j in enumerate(te):
                case = df.iloc[int(j)].to_dict()
                R[j] = _live_r_for_case(
                    case,
                    df=df,
                    train_idx=fit_idx,
                    pipe=pipe,
                    proba=proba_te[k],
                    cfg=live_cfg_fold,
                    conformal=conf,
                    ensemble=fold_ensemble,
                )

        y_correct = (best_oof_pred == y).astype(int)
        # Apply the declared A-D selection rule before fitting any production
        # fusion. The same mask is used by fold diagnostics and serving.
        from rockburst.fusion.set_compare import (
            compare_fusion_sets,
            production_fusion_from_oof,
            production_mask_from_comparison,
        )

        fusion_comparison = compare_fusion_sets(
            R,
            y_correct,
            y,
            groups,
            seed=seed,
            n_outer=n_outer,
            n_boot=20 if quick else 200,
            conf_eps=float(cfg.get("fusion", {}).get("conf_eps", 1e-3)),
        )
        mask, feat_report = production_mask_from_comparison(fusion_comparison)
        (out_dir / "fusion_set_comparison_slice4.json").write_text(
            json.dumps(fusion_comparison, indent=2, default=str)
        )
        results["fusion_set_comparison"] = fusion_comparison
        # Cross-fit stacker/calibrator/thresholds: fit on other folds' OOF, score held-out.
        r_final_oof = np.zeros(len(y), dtype=float)
        crossfit_fit_sets: list[set[int]] = []
        crossfit_te_sets: list[set[int]] = []
        for fold, (_tr, te) in enumerate(fold_splits):
            train_mask = (fold_of_row >= 0) & (fold_of_row != fold)
            te_mask = fold_of_row == fold
            if not train_mask.any() or not te_mask.any():
                continue
            idx_fit = np.where(train_mask)[0]
            idx_te = np.where(te_mask)[0]
            crossfit_fit_sets.append(set(int(i) for i in idx_fit))
            crossfit_te_sets.append(set(int(i) for i in idx_te))
            assert crossfit_fit_sets[-1].isdisjoint(crossfit_te_sets[-1])
            fold_fusion = ReliabilityFusion()
            fold_fusion.conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
            fold_fusion.fit(
                R[idx_fit], y_correct[idx_fit], feature_mask=mask, select_features=False
            )
            fold_fusion.feature_report = feat_report
            if fold_fusion.is_fit:
                # RC3: calibrate on all non-test OOF raw scores (~275), not ~70 inner_val.
                r_raw_fit = np.array(
                    [fold_fusion._raw_score(R[i]) for i in idx_fit]
                )
                fold_fusion.fit_calibrator(r_raw_fit, y_correct[idx_fit])
                r_val = np.array([fold_fusion.r_final(R[i]) for i in idx_fit])
                thr = select_decision_thresholds(r_val, y_correct[idx_fit])
                fold_fusion.set_decision_thresholds(thr)
                for j in idx_te:
                    r_final_oof[j] = fold_fusion.r_final(R[j])
            else:
                r_final_oof[idx_te] = 0.0
        results["fusion_crossfit_n_folds"] = len(crossfit_te_sets)
        results["fusion_crossfit_disjoint_ok"] = all(
            fs.isdisjoint(ts) for fs, ts in zip(crossfit_fit_sets, crossfit_te_sets)
        )
        r_final_crossfit_diagnostic = r_final_oof.copy()

        # Production fusion: one stacker/calibrator path for fit reporting + serve.
        conf_eps = float(cfg.get("fusion", {}).get("conf_eps", 1e-3))
        fusion, prod_meta = production_fusion_from_oof(
            R,
            y_correct,
            fold_of_row,
            conf_eps=conf_eps,
            method=cfg.get("fusion", {}).get("calibration_method"),
            comparison=fusion_comparison,
            interval_boot=20 if quick else 100,
            interval_groups=groups,
            seed=seed,
        )
        fusion.feature_report = feat_report
        # Report the exact score vector regenerated by the serialized serving
        # stacker/calibrator. Fold-stacker scores remain diagnostic metadata.
        r_final_oof = np.asarray(prod_meta["served_r_final_oof"], dtype=float)
        thresholds = (
            prod_meta.get("decision_thresholds") or fusion.decision_thresholds
        )
        _inner_fit, _inner_val = _split_fit_calib(
            y, groups, np.arange(len(y)), seed=seed + 99
        )

        fpath = models_dir / "fusion.joblib"
        fusion.save(fpath)
        results["fusion_path"] = _repo_relative_artifact(fpath, cfg)
        results["fusion_protocol"] = (
            "declared A-D set selected by risk-coverage rule; production stacker "
            "and calibrator fitted and reported on the identical served raw-score "
            "path; fold-stacker outputs retained only as diagnostics (RC3)"
        )
        results["fusion_calibration_protocol"] = prod_meta.get(
            "calibration_protocol", "crossfit_raw_full_oof"
        )
        results["fusion_coef"] = fusion.coef_.tolist() if fusion.is_fit else None
        results["fusion_intercept"] = (
            float(fusion.intercept_) if fusion.is_fit else None
        )
        results["fusion_feature_report"] = fusion.feature_report
        results["fusion_band_occupancy"] = prod_meta.get("band_occupancy")
        results["fusion_utility_sensitivity"] = prod_meta.get("utility_sensitivity")
        results["fusion_interval"] = {
            "method": prod_meta.get("interval_method"),
            "members": prod_meta.get("interval_members"),
        }
        results["fusion_zero_weight_flags"] = fusion.feature_report.get(
            "zero_weight_flags", []
        )
        results["fusion_coef_bootstrap"] = (
            bootstrap_coef_cis(
                R[_inner_fit],
                y_correct[_inner_fit],
                n_boot=50 if quick else 200,
                seed=seed,
                feature_mask=fusion.feature_mask,
            )
            if fusion.is_fit
            else None
        )
        results["decision_thresholds"] = thresholds
        from sklearn.metrics import brier_score_loss

        held = fold_of_row >= 0
        from rockburst.eval.metrics import ece_binary, ece_with_ci, reliability_diagram_bins
        from rockburst.fusion.set_compare import component_auroc_bootstrap

        results["fusion_r_final_ece"] = ece_binary(y_correct[held], r_final_oof[held])
        results["fusion_r_final_ece_crossfit"] = ece_binary(
            y_correct[held], r_final_crossfit_diagnostic[held]
        )
        results["fusion_r_final_ece_ci"] = ece_with_ci(
            y_correct[held],
            r_final_oof[held],
            groups=groups[held],
            n_boot=50 if quick else 200,
            seed=seed,
        )
        results["fusion_brier"] = float(brier_score_loss(y_correct[held], r_final_oof[held]))
        results["fusion_mean_r_final"] = float(r_final_oof[held].mean())
        results["fusion_oof_accuracy_of_base"] = float(y_correct[held].mean())
        results["fusion_calibrated"] = fusion.calibrator is not None
        results["fusion_calibration_method"] = fusion.calibration_method
        results["fusion_calibration_n"] = int(fusion.calibration_n)
        results["fusion_crossfit_disjoint"] = True
        results["fusion_component_auroc"] = component_auroc_bootstrap(
            R[held],
            y_correct[held],
            groups[held],
            n_boot=50 if quick else 200,
            seed=seed,
        )
        results["decision_thresholds_nonempty"] = True
        from rockburst.ml.calibration import temperature_calibration_report

        results["base_probability_calibration"] = temperature_calibration_report(
            best_oof_proba[held], y[held]
        )
        results["missing_d_confidence_note"] = (
            "See artifacts/baselines/fusion_missing_d_drop.json (Item 2)."
        )
        # C4: risk–coverage max-prob vs calibrated R_final (held-out OOF)
        max_prob = best_oof_proba[held].max(axis=1)
        rc_max = _risk_coverage_curve(max_prob, y_correct[held])
        rc_rf = _risk_coverage_curve(r_final_oof[held], y_correct[held])
        results["risk_coverage_max_prob"] = rc_max
        results["risk_coverage_r_final"] = rc_rf
        a_max = rc_max.get("area")
        a_rf = rc_rf.get("area")
        results["risk_coverage_comparison"] = {
            "max_prob_area": a_max,
            "r_final_area": a_rf,
            "r_final_minus_max_prob": (
                None if a_max is None or a_rf is None else float(a_rf - a_max)
            ),
            "note": (
                "R_final uses the feature set selected by the declared A-D rule. "
                "The comparison is computed from the exact served score vector."
            ),
        }

        results["fusion_reliability_diagram"] = reliability_diagram_bins(
            y_correct[held], r_final_oof[held]
        )
        np.save(models_dir / "fusion_R_oof.npy", R)
        np.save(models_dir / "fusion_y_correct.npy", y_correct)
        np.save(
            models_dir / "fusion_r_final_crossfit.npy",
            r_final_crossfit_diagnostic,
        )
        np.save(models_dir / "fusion_r_final_served.npy", r_final_oof)

        # Nested production conformal on nested OOF using fold calib of winners
        nested_cov = _recompute_coverage(best_name)
        results["nested_unbiased"]["conformal"] = nested_cov

    # RM5: criterion↔label agreement for severity documentation (cheap CSV pass).
    try:
        from rockburst.physics.criteria import criterion_label_agreement_table

        _rows = [df.iloc[i].to_dict() for i in range(len(df))]
        _ylabels = [int(v) for v in df["rockburst_grade"].tolist()]
        results["criterion_label_agreement"] = criterion_label_agreement_table(
            _rows, _ylabels
        )
    except Exception as exc:  # noqa: BLE001
        results["criterion_label_agreement"] = {"error": str(exc)}

    # Refresh Rm8 metadata at write time (train can take hours).
    results["run_metadata"] = build_run_metadata(
        mode="quick" if quick else "full",
        command="rockburst train" + (" --quick" if quick else ""),
        extra={
            "n_outer": n_outer,
            "n_inner": n_inner,
            "n_trials": n_trials,
            "best_model": results.get("best_model"),
        },
        config=cfg,
        input_paths=[path],
    )
    (out_dir / "m1_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results
