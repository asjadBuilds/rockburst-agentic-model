"""RM12 / RM13: triplet-grouped and label-policy M1+M7 headlines (Phase 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import run_predict
from rockburst.eval.metrics import (
    classification_report_dict,
    quadratic_weighted_kappa,
    severe_misclassification_rate,
)
from rockburst.fusion.fold_oof import fit_fold_fusion_oof
from rockburst.graph.runner import run_graph
from rockburst.ml.pipelines import build_preprocess, get_model_zoo, prepare_xy
from rockburst.ml.train import (
    _predict_proba_safe,
    _split_fit_calib,
    fit_fold_ensemble,
)
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.quality.near_dup import RAW_DUP_COLS, rows_near_duplicate
from rockburst.uncertainty.conformal import fit_marginal_thresholds, fit_mondrian_thresholds


def _round_key(v: Any, nd: int = 4) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, nd)


def triplet_key(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Rock-type / source key: (σc, σt, Wet)."""
    return (
        _round_key(row.get("sigma_c")),
        _round_key(row.get("sigma_t")),
        _round_key(row.get("wet")),
    )


def assign_triplet_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Integer group ids for StratifiedGroupKFold on (σc, σt, Wet)."""
    mapping: dict[tuple, int] = {}
    ids = np.empty(len(df), dtype=int)
    next_id = 0
    for i, (_, row) in enumerate(df.iterrows()):
        key = triplet_key(row)
        if None in key:
            # Incomplete triplet → singleton (no cross-fold leak via shared key)
            gid = next_id
            next_id += 1
        else:
            if key not in mapping:
                mapping[key] = next_id
                next_id += 1
            gid = mapping[key]
        ids[i] = gid
    return ids


def triplet_inventory(df: pd.DataFrame) -> dict[str, Any]:
    """Descriptive counts (legacy slice6 M13 fields)."""
    groups: dict[tuple, list[int]] = {}
    for _, row in df.iterrows():
        key = triplet_key(row)
        if None in key:
            continue
        groups.setdefault(key, []).append(int(row.get("no", -1)))
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    conflicting = 0
    top: list[dict[str, Any]] = []
    for key, nos in multi.items():
        sub = df[df["no"].isin(nos)] if "no" in df.columns else df.iloc[[]]
        grades = sub["rockburst_grade"].astype(int).tolist() if len(sub) else []
        uniq = sorted(set(grades))
        if len(uniq) > 1:
            conflicting += 1
        top.append(
            {
                "triplet": {"sigma_c": key[0], "sigma_t": key[1], "wet": key[2]},
                "n": len(nos),
                "nos": nos[:20],
                "grades": grades,
                "grade_span": int(max(grades) - min(grades)) if grades else 0,
            }
        )
    top.sort(key=lambda r: (-r["grade_span"], -r["n"]))
    return {
        "n_triplets": len(groups),
        "n_multi_row_triplets": len(multi),
        "n_conflicting_grade_triplets": conflicting,
        "top_span_examples": top[:15],
    }


def earliest_of_conflict_mask(df: pd.DataFrame, rel_tol: float = 0.05) -> np.ndarray:
    """Keep earliest row in each conflicting near-dup cluster; drop later conflicts."""
    keep = np.ones(len(df), dtype=bool)
    if "no" not in df.columns:
        return keep
    nos = df["no"].astype(int).to_numpy()
    grades = df["rockburst_grade"].astype(int).to_numpy()
    mat = df[list(RAW_DUP_COLS)].to_numpy(dtype=float)
    order = np.argsort(nos)
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1 :]:
            if not keep[j]:
                continue
            if rows_near_duplicate(mat[i], mat[j], rel_tol=rel_tol):
                if grades[i] != grades[j]:
                    keep[j] = False
    return keep


def drop_rows_315_344_mask(df: pd.DataFrame) -> np.ndarray:
    """True = keep. Prefer source ``no``; else positional 314:344."""
    if "no" in df.columns:
        nos = df["no"].astype(int)
        return ~((nos >= 315) & (nos <= 344)).to_numpy()
    keep = np.ones(len(df), dtype=bool)
    keep[314:344] = False
    return keep


def label_policy_frames(
    df: pd.DataFrame, *, rel_tol: float = 0.05
) -> dict[str, pd.DataFrame]:
    """Three Round-2 / M14 label policies."""
    return {
        "as_given": df.copy(),
        "drop_rows_315_344": df.loc[drop_rows_315_344_mask(df)].copy(),
        "earliest_of_conflict": df.loc[
            earliest_of_conflict_mask(df, rel_tol=rel_tol)
        ].copy(),
    }


def missing_d_vs_grade(df: pd.DataFrame) -> dict[str, Any]:
    has_d = df["d"].notna() if "d" in df.columns else pd.Series(False, index=df.index)
    g = df["rockburst_grade"].astype(int)
    out: dict[str, Any] = {}
    for label, mask in (("with_d", has_d), ("missing_d", ~has_d)):
        sub = g[mask]
        counts = {str(int(k)): int(v) for k, v in sub.value_counts().sort_index().items()}
        out[label] = {
            "n": int(mask.sum()),
            "grade_counts": counts,
            "mean_grade": float(sub.mean()) if len(sub) else None,
        }
    return out


def _fit_conf(method: str, y_cal, p_cal, alpha: float):
    if method.startswith("mondrian"):
        return fit_mondrian_thresholds(y_cal, p_cal, alpha=alpha)
    return fit_marginal_thresholds(y_cal, p_cal, alpha=alpha)


def _summarise_system(
    y: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    abstain: np.ndarray,
) -> dict[str, Any]:
    mask = pred >= 0
    if not mask.any():
        return {"n": 0, "qwk": None, "accuracy": None}
    report = classification_report_dict(y[mask], pred[mask], proba[mask])
    accepted = abstain[mask] == 0
    acc_acc = (
        float((y[mask][accepted] == pred[mask][accepted]).mean())
        if accepted.any()
        else None
    )
    return {
        "n": int(mask.sum()),
        "qwk": report.get("qwk"),
        "accuracy": report.get("accuracy"),
        "ordinal_mae": report.get("ordinal_mae"),
        "severe_misclassification_rate": report.get("severe_misclassification_rate"),
        "abstention_rate": float(abstain[mask].mean()),
        "accepted_accuracy": acc_acc,
        "accepted_n": int(accepted.sum()),
    }


def evaluate_m1_m7(
    df: pd.DataFrame,
    groups: np.ndarray,
    cfg: dict[str, Any],
    m1_meta: dict[str, Any],
    *,
    quick: bool = False,
    n_boot: int = 200,
    include_observed_indicators: bool = True,
) -> dict[str, Any]:
    """Held-out M1 + full M7 under a supplied group key (dup or triplet).

    Uses production model params from ``m1_meta`` (no Optuna retune).
    """
    if len(df) < 20:
        return {"n": int(len(df)), "error": "too_few_rows"}

    X, y, _ = prepare_xy(
        df, include_observed_indicators=include_observed_indicators
    )
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("groups length must match dataframe rows")

    seed = int(cfg.get("seed", 42))
    n_outer = 2 if quick else int(cfg["cv"]["n_outer"])
    n_repeats = 1 if quick else int(cfg["cv"].get("n_repeats", 5))
    # Cap folds if too few unique groups
    n_unique = int(len(np.unique(groups)))
    n_outer = min(n_outer, max(2, n_unique // 2)) if n_unique >= 2 else 2
    alpha = float(cfg.get("uncertainty", {}).get("conformal_alpha", 0.1))
    conf_method = (m1_meta.get("conformal") or {}).get("method", "marginal_split")

    best_name = (
        m1_meta.get("nested_unbiased", {}).get("production_model")
        or m1_meta.get("best_model")
        or "logistic__median"
    )
    name, imp = best_name.split("__", 1)
    params = m1_meta.get("best_params") or {}
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:  # noqa: BLE001
        zoo = get_model_zoo(quick=True)
        base_est = zoo[name] if name in zoo else next(iter(zoo.values()))

    models_meta = m1_meta.get("models") or {}
    ranked = sorted(
        models_meta.items(),
        key=lambda kv: (kv[1].get("pooled_oof") or {}).get("qwk", -1),
        reverse=True,
    )
    if not ranked:
        ranked = [(best_name, {"best_params": params, "fold_params": []})]

    store: dict[str, dict[str, list]] = {
        s: {"y": [], "pred": [], "proba": [], "abstain": [], "groups": []}
        for s in ("M1", "M7")
    }

    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_outer, shuffle=True, random_state=seed + rep
        )
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + rep + fold)
            pipe = Pipeline(
                [("prep", build_preprocess(imp)), ("model", clone(base_est))]
            )
            pipe.fit(X.iloc[fit_idx], y[fit_idx])
            conf = None
            if len(calib_idx):
                conf = _fit_conf(
                    conf_method,
                    y[calib_idx],
                    _predict_proba_safe(pipe, X.iloc[calib_idx]),
                    alpha,
                )
                conf.n_fit = int(len(fit_idx))

            fold_ensemble = fit_fold_ensemble(
                X, y, fit_idx, ranked, seed=seed + rep, fold=fold, top_k=3
            )
            fold_cfg_base = {
                **cfg,
                "uncertainty": {**cfg.get("uncertainty", {}), "n_perturb": 4},
                "_fold_ensemble": fold_ensemble,
            }
            fusion = fit_fold_fusion_oof(
                df,
                X,
                y,
                groups,
                tr,
                base_est=base_est,
                imputer=imp,
                ranked=ranked,
                cfg=fold_cfg_base,
                seed=seed + rep,
                fold=fold,
                conf_method=conf_method,
                alpha=alpha,
                quick=quick,
            )
            fold_cfg = {
                **fold_cfg_base,
                "_fold_pipeline": pipe,
                "_fold_ensemble": fold_ensemble,
                "_fold_fusion": fusion,
                "_fold_model_name": best_name,
                "_fold_X": X.iloc[fit_idx],
                "_fold_y": y[fit_idx],
                "_fold_alt_pipes": {},
                "_fold_cross_pipes": {},
                "_conformal_state": conf,
                "decode_rule": m1_meta.get("decode_rule")
                or (cfg.get("ml") or {}).get("decode_rule")
                or "argmax",
            }

            te_df = df.iloc[te]
            g_te = groups[te]
            for local_i, (_, row) in enumerate(te_df.iterrows()):
                case = row.to_dict()
                ref = build_fold_reference(df, fit_idx, case)
                pred = run_predict(case, fold_cfg, pipeline=pipe, model_name=best_name)
                y_true = int(row["rockburst_grade"])
                gid = int(g_te[local_i])

                store["M1"]["y"].append(y_true)
                store["M1"]["pred"].append(pred.predicted_grade)
                store["M1"]["proba"].append(pred.probabilities)
                store["M1"]["abstain"].append(0)
                store["M1"]["groups"].append(gid)

                m7 = run_graph(case, ref, fold_cfg, use_llm=False, allow_critic=True)
                g7 = m7.get("predicted_grade")
                g7i = int(g7) if g7 is not None else pred.predicted_grade
                store["M7"]["y"].append(y_true)
                store["M7"]["pred"].append(g7i)
                store["M7"]["proba"].append(m7.get("probabilities") or pred.probabilities)
                store["M7"]["abstain"].append(int(bool(m7.get("abstained"))))
                store["M7"]["groups"].append(gid)

    out: dict[str, Any] = {
        "production_model": best_name,
        "include_observed_indicators": bool(include_observed_indicators),
        "n_outer": n_outer,
        "n_repeats": n_repeats,
        "n_rows": int(len(df)),
        "n_groups": int(len(np.unique(groups))),
        "mode": "quick" if quick else "full",
        "systems": {},
    }
    for s in ("M1", "M7"):
        y_s = np.asarray(store[s]["y"], dtype=int)
        p_s = np.asarray(store[s]["pred"], dtype=int)
        pr_s = np.asarray(store[s]["proba"], dtype=float)
        ab_s = np.asarray(store[s]["abstain"], dtype=int)
        g_s = np.asarray(store[s]["groups"])
        summary = _summarise_system(y_s, p_s, pr_s, ab_s)
        # Grouped bootstrap CI on QWK
        uniq = np.unique(g_s)
        rng = np.random.default_rng(seed)
        qwk_boots: list[float] = []
        for _ in range(n_boot if not quick else 40):
            sg = rng.choice(uniq, size=len(uniq), replace=True)
            m = np.isin(g_s, sg)
            if m.sum() < 5:
                continue
            qwk_boots.append(quadratic_weighted_kappa(y_s[m], p_s[m]))
        if qwk_boots:
            summary["qwk_ci95"] = [
                float(np.quantile(qwk_boots, 0.025)),
                float(np.quantile(qwk_boots, 0.975)),
            ]
        else:
            summary["qwk_ci95"] = None
        summary["severe_rate"] = severe_misclassification_rate(y_s, p_s)
        out["systems"][s] = summary
    return out
