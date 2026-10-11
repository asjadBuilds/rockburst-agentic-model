#!/usr/bin/env python3
"""Item 5: near-dup sensitivity of headlines + label-noise analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.config import load_config, resolve_path
from rockburst.data.audit import assign_dup_groups, process_raw
from rockburst.eval.metrics import quadratic_weighted_kappa
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.near_dup import RAW_DUP_COLS
from rockburst.uncertainty.conformal import fit_marginal_thresholds, coverage_and_size


def _rel_dist(a: pd.Series, b: pd.Series) -> float:
    vals = []
    for c in RAW_DUP_COLS:
        va, vb = a.get(c), b.get(c)
        if va is None or vb is None:
            continue
        if isinstance(va, float) and not np.isfinite(va):
            continue
        if isinstance(vb, float) and not np.isfinite(vb):
            continue
        va, vb = float(va), float(vb)
        denom = max(abs(va), abs(vb), 1e-12)
        vals.append(abs(va - vb) / denom)
    return float(max(vals)) if vals else float("inf")


def _headline_at_tol(
    df_raw: pd.DataFrame, cfg: dict, tol: float, m1: dict, *, n_repeats: int = 5
) -> dict:
    """Fixed catboost__knn (no re-tune); >=5 grouped-CV repeats; grouped-bootstrap CI."""
    from rockburst.eval.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_metric
    from rockburst.uncertainty.conformal import conformal_prediction_set

    cfg2 = json.loads(json.dumps(cfg))
    cfg2.setdefault("quality", {})["near_dup_rel_tol"] = tol
    processed, audit = process_raw(df_raw, cfg2)
    X, y, groups = prepare_xy(processed)
    best_name = "catboost__knn"
    name, imp = best_name.split("__", 1)
    params = m1.get("best_params") or {}
    seed = int(cfg.get("seed", 42))
    try:
        base_est = estimator_from_params(name, params, seed)
    except Exception:
        from rockburst.ml.pipelines import get_model_zoo

        base_est = get_model_zoo(quick=True)[name]

    qwks, accs, covs = [], [], []
    # Pooled last-repeat vectors for bootstrap CI
    last_y = last_pred = last_correct = last_groups = last_covered = None
    for rep in range(n_repeats):
        oof_pred = np.zeros(len(y), dtype=int)
        sets: list = [None] * len(y)
        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed + 17 * rep
        )
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            fit_idx, calib_idx = _split_fit_calib(
                y, groups, tr, seed=seed + fold + 100 * rep
            )
            pipe = Pipeline(
                [("prep", build_preprocess(imp)), ("model", clone(base_est))]
            )
            pipe.fit(X.iloc[fit_idx], y[fit_idx])
            proba = _predict_proba_safe(pipe, X.iloc[te])
            oof_pred[te] = proba.argmax(axis=1)
            if len(calib_idx):
                conf = fit_marginal_thresholds(
                    y[calib_idx],
                    _predict_proba_safe(pipe, X.iloc[calib_idx]),
                    alpha=0.1,
                )
                for k, j in enumerate(te):
                    sets[j] = conformal_prediction_set(
                        proba[k], alpha=0.1, class_thresholds=conf.class_thresholds
                    )
        qwks.append(quadratic_weighted_kappa(y, oof_pred))
        accs.append(float((y == oof_pred).mean()))
        cov = coverage_and_size(y, sets)
        covs.append(cov["coverage"])
        last_y = np.asarray(y, dtype=int)
        last_pred = np.asarray(oof_pred, dtype=int)
        last_correct = (y == oof_pred).astype(float)
        last_groups = np.asarray(groups)
        last_covered = np.asarray(
            [1.0 if s is not None and y[i] in s else 0.0 for i, s in enumerate(sets)]
        )

    def _ci(vals_case, g):
        return grouped_bootstrap_ci(vals_case, g, n_boot=200, seed=seed)

    qwk_ci = (
        grouped_bootstrap_metric(
            last_y,
            last_pred,
            last_groups,
            quadratic_weighted_kappa,
            n_boot=200,
            seed=seed,
        )
        if last_y is not None
        else None
    )
    return {
        "tol": tol,
        "n_groups": audit["n_dup_groups"],
        "n_multi": audit["n_groups_with_duplicates"],
        "n_conflict_groups": audit["n_conflicting_label_groups"],
        "n_repeats": n_repeats,
        "model": "catboost__knn_fixed",
        "M1_qwk_mean": float(np.mean(qwks)),
        "M1_accuracy_mean": float(np.mean(accs)),
        "conformal_coverage_mean": float(np.mean(covs)),
        "M1_qwk_ci": qwk_ci,
        "M1_accuracy_ci": _ci(last_correct, last_groups) if last_correct is not None else None,
        "conformal_coverage_ci": _ci(last_covered, last_groups)
        if last_covered is not None
        else None,
        "M1_qwk": float(np.mean(qwks)),
        "M1_accuracy": float(np.mean(accs)),
        "conformal_coverage": float(np.mean(covs)),
        "_last_oof": {
            "y": last_y,
            "pred": last_pred,
            "groups": last_groups,
            "covered": last_covered,
        },
        "note": (
            "Fixed catboost__knn, no re-tune; >=5 grouped-CV repeats; "
            "QWK CI = grouped bootstrap of QWK per resample; "
            "acc/coverage CIs = grouped bootstrap of case vectors."
        ),
    }

def main() -> None:
    cfg = load_config()
    raw = pd.read_csv(resolve_path(cfg, "raw_csv"))
    m1 = json.loads((resolve_path(cfg, "baselines") / "m1_results.json").read_text())
    abl = json.loads((resolve_path(cfg, "baselines") / "ablation_results.json").read_text())

    tols = [0.005, 0.01, 0.02, 0.05]
    headlines = [_headline_at_tol(raw, cfg, t, m1) for t in tols]

    # Pair checks: which tol first groups them
    processed_default, audit_default = process_raw(raw, cfg)
    pairs = [(19, 342), (20, 328), (9, 337)]
    pair_tol = []
    for a, b in pairs:
        ra = raw.loc[raw["no"] == a].iloc[0]
        rb = raw.loc[raw["no"] == b].iloc[0]
        dist = _rel_dist(ra, rb)
        first = None
        for t in tols:
            cfg2 = json.loads(json.dumps(cfg))
            cfg2.setdefault("quality", {})["near_dup_rel_tol"] = t
            proc, _ = process_raw(raw, cfg2)
            ga = int(proc.loc[proc["no"] == a, "dup_group_id"].iloc[0])
            gb = int(proc.loc[proc["no"] == b, "dup_group_id"].iloc[0])
            if ga == gb and first is None:
                first = t
        pair_tol.append(
            {"pair": [a, b], "max_rel_dist": dist, "first_grouped_tol": first}
        )

    # NN distance gap for tolerance choice
    # For each row, min rel dist to any other row
    nn = []
    for i, row in raw.iterrows():
        best = float("inf")
        for j, other in raw.iterrows():
            if int(row["no"]) == int(other["no"]):
                continue
            d = _rel_dist(row, other)
            if d < best:
                best = d
        if np.isfinite(best):
            nn.append(best)
    nn = np.asarray(nn)
    # Look for gap near 1%
    hist_edges = [0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 1.0]
    hist = {
        f"{hist_edges[i]}-{hist_edges[i+1]}": int(
            ((nn >= hist_edges[i]) & (nn < hist_edges[i + 1])).sum()
        )
        for i in range(len(hist_edges) - 1)
    }
    # Choice: keep 1% — clear mass below 0.5–1% for Table S1 copies; 1.5% pairs are
    # borderline measurement noise; pre-register as domain choice from gap (mass <1% vs sparse 1–5%)
    chosen_tol = 0.05
    choice_note = (
        "Primary near_dup_rel_tol=0.05 decided after seeing that 1% left known near-copies "
        "ungrouped (19–342, 20–328 first group at 2%; 9–337 at 5%). Earlier headline "
        "numbers used 1%. Keep 1%/2% as sensitivity. Not QWK-tuned."
    )

    # Label-noise analysis on default 1% conflicts
    conflicts = audit_default.get("conflicting_label_pairs") or []
    dirs = []
    block_315 = []
    for pair in conflicts:
        if isinstance(pair, dict) and "no_a" in pair:
            nos = [int(pair["no_a"]), int(pair["no_b"])]
            grades = [int(pair["grade_a"]), int(pair["grade_b"])]
        elif isinstance(pair, dict):
            nos = pair.get("nos") or pair.get("pair") or []
            grades = pair.get("grades") or []
        else:
            nos = list(pair)
            grades = [
                int(raw.loc[raw["no"] == n, "rockburst_grade"].iloc[0]) for n in nos
            ]
        if len(nos) < 2:
            continue
        order = np.argsort(nos)
        nos = [nos[i] for i in order]
        grades = [grades[i] for i in order]
        delta = int(grades[-1] - grades[0])
        dirs.append({"nos": nos, "grades": grades, "delta": delta})
        if any(315 <= n <= 344 for n in nos):
            block_315.append(delta)

    deltas = [d["delta"] for d in dirs]
    counts = {
        "+1": int(sum(1 for d in deltas if d == 1)),
        "-1": int(sum(1 for d in deltas if d == -1)),
        "larger_abs": int(sum(1 for d in deltas if abs(d) > 1)),
        "zero": int(sum(1 for d in deltas if d == 0)),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "n_pairs": len(deltas),
    }
    block_finding = {
        "n_conflict_pairs_touching_315_344": len(block_315),
        "mean_delta_later_minus_earlier": float(np.mean(block_315)) if block_315 else None,
        "frac_plus_one": float(np.mean([d == 1 for d in block_315])) if block_315 else None,
        "statement": (
            "Among conflicting pairs involving rows 315–344, later grades are often "
            "+1 vs earlier counterparts; do not over-interpret as systematic relabeling "
            "without source confirmation."
        ),
    }

    # Three policies: keep all / keep one (lowest no) / drop conflicting groups
    X, y, groups = prepare_xy(processed_default)
    conflict_groups = set()
    for d in dirs:
        g = processed_default.loc[
            processed_default["no"].isin(d["nos"]), "dup_group_id"
        ].unique()
        conflict_groups.update(int(x) for x in g)

    def _oof_metrics(mask):
        Xm, ym, gm = X.loc[mask], y[mask], groups[mask]
        if len(ym) < 40:
            return {"n": int(mask.sum()), "qwk": None, "accuracy": None}
        best_name = m1.get("nested_unbiased", {}).get("production_model") or "catboost__knn"
        name, imp = best_name.split("__", 1)
        params = m1.get("best_params") or {}
        seed = int(cfg.get("seed", 42))
        try:
            base_est = estimator_from_params(name, params, seed)
        except Exception:
            from rockburst.ml.pipelines import get_model_zoo

            base_est = get_model_zoo(quick=True)[name]
        oof = np.zeros(len(ym), dtype=int)
        # remap groups
        sgkf = StratifiedGroupKFold(
            n_splits=min(5, len(np.unique(gm))), shuffle=True, random_state=seed
        )
        try:
            splits = list(sgkf.split(Xm, ym, gm))
        except ValueError:
            return {"n": int(mask.sum()), "qwk": None, "accuracy": None, "error": "split_failed"}
        for fold, (tr, te) in enumerate(splits):
            pipe = Pipeline(
                [("prep", build_preprocess(imp)), ("model", clone(base_est))]
            )
            pipe.fit(Xm.iloc[tr], ym[tr])
            oof[te] = _predict_proba_safe(pipe, Xm.iloc[te]).argmax(axis=1)
        return {
            "n": int(mask.sum()),
            "qwk": quadratic_weighted_kappa(ym, oof),
            "accuracy": float((ym == oof).mean()),
        }

    keep_all = np.ones(len(y), dtype=bool)
    # keep one copy per conflicting group: keep lowest no
    drop_extra = np.ones(len(y), dtype=bool)
    for gid in conflict_groups:
        idx = np.where(groups == gid)[0]
        nos = processed_default.iloc[idx]["no"].to_numpy()
        keep_i = idx[np.argmin(nos)]
        for i in idx:
            if i != keep_i:
                drop_extra[i] = False
    drop_conf = ~np.isin(groups, list(conflict_groups))

    policies = {
        "keep_all": _oof_metrics(keep_all),
        "keep_one_lowest_no_per_conflict_group": _oof_metrics(drop_extra),
        "drop_conflicting_groups": _oof_metrics(drop_conf),
    }
    # Ceiling: if all conflict pairs were noise, accuracy ceiling related to conflict rate
    n_conflict_rows = int((~drop_conf).sum())
    ceiling = {
        "n_conflict_rows": n_conflict_rows,
        "n_conflict_pairs": counts["n_pairs"],
        "note": (
            "Label conflicts imply an accuracy/QWK ceiling below 1.0 even with a perfect "
            "feature→label map; dropping conflict groups raises empirical scores but "
            "shrinks n — report all three policies."
        ),
        "M7_abstention_default_tol": abl["ablations"]["M7"]["abstention_rate"]["mean"],
    }

    # Persist last-repeat OOF for CI recompute; strip from JSON payload
    oof_path = resolve_path(cfg, "baselines") / "item5_last_oof.npz"
    save_kw: dict = {}
    clean_headlines = []
    for h in headlines:
        blob = h.pop("_last_oof", None)
        if blob and blob.get("y") is not None:
            t = h["tol"]
            save_kw[f"y_{t}"] = np.asarray(blob["y"])
            save_kw[f"pred_{t}"] = np.asarray(blob["pred"])
            save_kw[f"groups_{t}"] = np.asarray(blob["groups"])
            if blob.get("covered") is not None:
                save_kw[f"covered_{t}"] = np.asarray(blob["covered"])
        clean_headlines.append(h)
    if save_kw:
        np.savez_compressed(oof_path, **save_kw)

    out = {
        "headlines_by_tol": clean_headlines,
        "pair_checks": pair_tol,
        "nn_hist": hist,
        "nn_median": float(np.median(nn)),
        "chosen_tol": chosen_tol,
        "choice_note": choice_note,
        "label_noise_direction": counts,
        "block_315_344": block_finding,
        "policies": policies,
        "ceiling": ceiling,
    }
    dest = resolve_path(cfg, "baselines") / "item5_dup_label_noise.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "headlines": clean_headlines,
        "pairs": pair_tol,
        "chosen_tol": chosen_tol,
        "direction": counts,
        "block": block_finding,
        "policies": policies,
    }, indent=2, default=str)[:5000])


if __name__ == "__main__":
    main()
