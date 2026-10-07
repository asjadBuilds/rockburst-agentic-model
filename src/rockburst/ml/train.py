"""Train baselines with nested grouped CV, split conformal, and live-R fusion."""

from __future__ import annotations

import json
from collections import Counter
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
from rockburst.eval.metrics import classification_report_dict, expected_calibration_error
from rockburst.fusion.features import compute_reliability_features
from rockburst.fusion.stacker import ReliabilityFusion, select_decision_thresholds
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
    fit_marginal_thresholds,
    fit_mondrian_thresholds,
    pick_defensible_method,
)


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
    return fit_marginal_thresholds(y_calib, p_cal, alpha=alpha)


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
    pred = PredictiveOutput(
        predicted_grade=int(np.argmax(proba)),
        probabilities=np.asarray(proba, dtype=float).tolist(),
        r_model=float(np.clip(float(np.max(proba)) * (
            1 - 0.07 * sum(
                1 for c in FEATURE_COLS
                if case.get(c) is None or not np.isfinite(float(case[c]))
            )
        ), 0, 1)),
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
    return compute_reliability_features(dq, phys, pred, unc)


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
        "search_spaces": SEARCH_SPACES_DOC,
        "models": {},
        "complete_case_n": int(len(df.dropna(subset=FEATURE_COLS))),
        "d_missing_n": int(df["d"].isna().sum()),
        "conformal": {"alpha": alpha},
        "protocol_note": (
            "selection_biased: argmax pooled outer-OOF QWK over model×imputer. "
            "nested_unbiased: per outer fold pick model×imputer by inner QWK only. "
            "pooled_oof vs mean_fold are both reported for every candidate."
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
                try:
                    pipe.fit(X.iloc[fit_idx], y[fit_idx])
                except Exception as exc:  # noqa: BLE001
                    fold_metrics.append({"fold": fold, "error": str(exc)})
                    fold_store[key].append(
                        {"fold": fold, "inner_qwk": float(iq), "error": str(exc)}
                    )
                    continue
                conf: ConformalState | None = None
                p_cal = None
                if len(calib_idx) > 0:
                    p_cal = _predict_proba_safe(pipe, X.iloc[calib_idx])
                    conf = fit_marginal_thresholds(y[calib_idx], p_cal, alpha=alpha)
                    conf.n_fit = int(len(fit_idx))
                proba = _predict_proba_safe(pipe, X.iloc[te])
                pred = proba.argmax(axis=1)
                oof_pred[te] = pred
                oof_proba[te] = proba
                sets = [
                    conformal_prediction_set(
                        proba[i],
                        alpha=alpha,
                        class_thresholds=None if conf is None else conf.class_thresholds,
                    )
                    for i in range(len(te))
                ]
                cov = coverage_and_size(y[te], sets)
                fold_coverage.append(cov)
                fm = {
                    "fold": fold,
                    **classification_report_dict(y[te], pred, proba),
                    "conformal": cov,
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
            cov_vals = [c["coverage"] for c in fold_coverage if c.get("coverage") is not None]
            size_vals = [c["mean_size"] for c in fold_coverage if c.get("mean_size") is not None]
            pooled["prediction_set_coverage"] = float(np.mean(cov_vals)) if cov_vals else None
            pooled["prediction_set_mean_size"] = float(np.mean(size_vals)) if size_vals else None
            mean_fold = _mean_fold_metrics(fold_metrics)
            results["models"][key] = {
                "pooled_oof": pooled,
                "mean_fold": mean_fold,
                "overall": pooled,  # backward-compat alias = pooled
                "folds": fold_metrics,
                "best_params": fold_params_list[-1] if fold_params_list else {},
                "fold_params": fold_params_list,
                "inner_qwk_mean": float(np.nanmean(fold_inner)) if fold_inner else None,
            }

    # --- Selection-biased: argmax pooled outer OOF QWK ---
    candidate_table = []
    for key, meta in results["models"].items():
        candidate_table.append(
            {
                "model": key,
                "pooled_oof_qwk": meta["pooled_oof"].get("qwk"),
                "pooled_oof_accuracy": meta["pooled_oof"].get("accuracy"),
                "mean_fold_qwk": meta["mean_fold"].get("qwk"),
                "mean_fold_accuracy": meta["mean_fold"].get("accuracy"),
                "inner_qwk_mean": meta.get("inner_qwk_mean"),
            }
        )
    candidate_table.sort(key=lambda r: r["pooled_oof_qwk"] or -1, reverse=True)
    results["candidate_table"] = candidate_table

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
    prod_from_nested = (
        Counter(nested_fold_winners).most_common(1)[0][0]
        if nested_fold_winners
        else biased_name
    )
    results["nested_unbiased"] = {
        "pooled_oof": nested_pooled,
        "mean_fold": _mean_fold_metrics(
            [{**m, "conformal": {}} for m in nested_fold_metrics]
        ),
        "fold_winners": nested_fold_winners,
        "production_model": prod_from_nested,
        "note": "Unbiased: model×imputer selected inside each outer fold on inner QWK only.",
    }

    # Headline / production model = nested mode (honest); keep biased for comparison
    best_name = prod_from_nested or biased_name
    results["best_model"] = best_name
    results["best_qwk"] = (
        nested_pooled.get("qwk") if nested_pooled else results["selection_biased"]["pooled_oof"]["qwk"]
    )
    results["best_qwk_selection_biased"] = (
        results["selection_biased"]["pooled_oof"]["qwk"] if biased_name else None
    )
    results["best_qwk_nested_unbiased"] = nested_pooled.get("qwk")
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
        method_cov: dict[str, list[float]] = {
            "mondrian_split": [],
            "marginal_split": [],
            "cv_plus_grouped": [],
        }
        method_size: dict[str, list[float]] = {k: [] for k in method_cov}
        per_class_cov: dict[str, list[dict]] = {k: [] for k in method_cov}
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
            for m in method_cov:
                if m in cmp_ and cmp_[m].get("coverage") is not None:
                    method_cov[m].append(float(cmp_[m]["coverage"]))
                    method_size[m].append(float(cmp_[m]["mean_size"]))
                    per_class_cov[m].append(cmp_[m].get("per_class", {}))

        # Aggregate comparison
        agg_cmp: dict[str, Any] = {}
        for m in method_cov:
            if not method_cov[m]:
                continue
            agg_cmp[m] = {
                "coverage": float(np.mean(method_cov[m])),
                "mean_size": float(np.mean(method_size[m])),
                "coverage_std": float(np.std(method_cov[m], ddof=1)) if len(method_cov[m]) > 1 else 0.0,
            }
        chosen = "marginal_split"  # a priori (preregistration); not picked by closeness to 0.90
        results["conformal"] = {
            "alpha": alpha,
            "method": chosen,
            "comparison": agg_cmp,
            "class_calib_counts_example": (
                fold_comparisons[0].get("class_calib_counts") if fold_comparisons else []
            ),
            "selection_note": (
                "A priori marginal_split (simplest finite-sample guarantee). "
                "Mondrian / CV+ reported in comparison; not selected by test coverage."
            ),
        }
        conformal_method = chosen
    else:
        conformal_method = "marginal_split"

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
                "fit_idx_for_conformal_note": (
                    "Thresholds fitted on calib holdout; production pipe refit on all rows."
                ),
            },
            model_path,
        )
        results["model_path"] = str(model_path)
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
        results["ensemble_path"] = str(ens_path)

        # Live-agent OOF R using production model family fold params
        R = np.zeros((len(y), 5), dtype=float)
        live_cfg = {
            **cfg,
            "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4 if quick else 8},
        }
        best_oof_pred = np.zeros(len(y), dtype=int)
        best_oof_proba = np.zeros((len(y), 4), dtype=float)
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
            best_oof_pred[te] = proba_te.argmax(axis=1)
            best_oof_proba[te] = proba_te
            for k, j in enumerate(te):
                case = df.iloc[int(j)].to_dict()
                R[j] = _live_r_for_case(
                    case,
                    df=df,
                    train_idx=fit_idx,
                    pipe=pipe,
                    proba=proba_te[k],
                    cfg=live_cfg,
                    conformal=conf,
                    ensemble=None,
                )

        y_correct = (best_oof_pred == y).astype(int)
        # Inner split for feature selection / calibration / thresholds (not outer test)
        _inner_fit, inner_val = _split_fit_calib(
            y, groups, np.arange(len(y)), seed=seed + 99
        )
        from rockburst.fusion.stacker import choose_feature_mask, bootstrap_coef_cis

        mask, feat_report = choose_feature_mask(R[_inner_fit], y_correct[_inner_fit])
        fusion = ReliabilityFusion().fit(
            R, y_correct, feature_mask=mask, select_features=False
        )
        fusion.feature_report = feat_report
        # Raw scores; calibrate on inner val only
        r_raw_oof = np.array(
            [
                fusion._raw_score(R[i]) if fusion.is_fit else 0.0
                for i in range(len(y))
            ]
        )
        fusion.fit_calibrator(r_raw_oof[inner_val], y_correct[inner_val])
        r_final_oof = np.array(
            [fusion.r_final(R[i]) if fusion.is_fit else 0.0 for i in range(len(y))]
        )
        thresholds = select_decision_thresholds(
            r_final_oof[inner_val], y_correct[inner_val]
        )
        fusion.set_decision_thresholds(thresholds)
        fusion.decision_thresholds.update(thresholds)

        fpath = models_dir / "fusion.joblib"
        fusion.save(fpath)
        results["fusion_path"] = str(fpath)
        results["fusion_coef"] = fusion.coef_.tolist() if fusion.is_fit else None
        results["fusion_intercept"] = (
            float(fusion.intercept_) if fusion.is_fit else None
        )
        results["fusion_feature_report"] = fusion.feature_report
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

        results["fusion_r_final_ece"] = expected_calibration_error(
            y_correct, np.column_stack([1 - r_final_oof, r_final_oof])
        )
        results["fusion_r_final_ece_inner_val"] = expected_calibration_error(
            y_correct[inner_val],
            np.column_stack(
                [1 - r_final_oof[inner_val], r_final_oof[inner_val]]
            ),
        )
        results["fusion_brier"] = float(brier_score_loss(y_correct, r_final_oof))
        results["fusion_mean_r_final"] = float(r_final_oof.mean())
        results["fusion_oof_accuracy_of_base"] = float(y_correct.mean())
        results["fusion_calibrated"] = fusion.calibrator is not None
        results["missing_d_confidence_note"] = (
            "See artifacts/baselines/fusion_missing_d_drop.json (Item 2)."
        )

        bins = np.linspace(0, 1, 11)
        diagram = []
        for i in range(10):
            m = (r_final_oof > bins[i]) & (r_final_oof <= bins[i + 1])
            if m.sum() == 0:
                continue
            diagram.append(
                {
                    "bin": [float(bins[i]), float(bins[i + 1])],
                    "mean_r_final": float(r_final_oof[m].mean()),
                    "empirical_correct": float(y_correct[m].mean()),
                    "n": int(m.sum()),
                }
            )
        results["fusion_reliability_diagram"] = diagram
        np.save(models_dir / "fusion_R_oof.npy", R)
        np.save(models_dir / "fusion_y_correct.npy", y_correct)

        # Nested production conformal on nested OOF using fold calib of winners
        nested_cov = _recompute_coverage(best_name)
        results["nested_unbiased"]["conformal"] = nested_cov

    (out_dir / "m1_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results
