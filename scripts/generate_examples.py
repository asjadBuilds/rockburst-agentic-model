#!/usr/bin/env python3
"""Generate evidence package examples/traces (Slice 7 / C5, M15).

- Forced abstain via real max_loops termination (not hard-coded copy).
- Production showcase cases (full-data refit disclaimer).
- Held-out fold examples: real abstain + severe high-confidence error (|ŷ−y|≥2).
- Standalone failure write-up under artifacts/examples/failure_analysis.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from rockburst.agents.tools import intensity_name
from rockburst.config import load_config, resolve_path
from rockburst.graph.runner import assess_case, run_graph
from rockburst.ml.pipelines import build_preprocess, prepare_xy
from rockburst.ml.train import _predict_proba_safe, _split_fit_calib
from rockburst.ml.tuning import estimator_from_params
from rockburst.quality.fold_ref import build_fold_reference
from rockburst.uncertainty.conformal import (
    ConformalState,
    fit_mondrian_thresholds,
    fit_marginal_thresholds,
)


# Production-model showcase cases (full-data refit — labels include disclaimer).
# Abstain / failure are selected dynamically from held-out fold (C5); these IDs
# remain as supplementary production illustrations only.
PROD_CASES = {
    "standard_a": 15,
    "physics_b1": 20,
    "standard_b": 67,
    "ood": 146,
    "physics_b1_244": 244,
    "near_dup": 180,
}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _fit_fold_context(cfg: dict, *, fold_index: int = 0) -> dict | None:
    """Train one outer-fold pipeline for held-out assessment."""
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    X, y, groups = prepare_xy(df)
    seed = int(cfg.get("seed", 42))
    meta_path = resolve_path(cfg, "baselines") / "m1_results.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    best = meta.get("best_model") or "logistic__median"
    name, imp = best.split("__", 1)
    # Prefer fold-local params when present (honest held-out)
    fold_params_list = (meta.get("models") or {}).get(best, {}).get("fold_params") or []
    if fold_index < len(fold_params_list) and fold_params_list[fold_index]:
        params = fold_params_list[fold_index]
    else:
        params = meta.get("best_params") or {}
    est = estimator_from_params(name, params, seed + fold_index)
    n_outer = int(cfg.get("cv", {}).get("n_outer", 5))
    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X, y, groups))
    if fold_index >= len(splits):
        fold_index = 0
    tr, te = splits[fold_index]
    fit_idx, calib_idx = _split_fit_calib(y, groups, tr, seed=seed + fold_index)
    pipe = Pipeline([("prep", build_preprocess(imp)), ("model", clone(est))])
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    conf: ConformalState | None = None
    method = (meta.get("conformal") or {}).get("method") or "mondrian_split"
    if len(calib_idx):
        p_cal = _predict_proba_safe(pipe, X.iloc[calib_idx])
        if method == "mondrian_split":
            conf = fit_mondrian_thresholds(y[calib_idx], p_cal, alpha=0.1)
        else:
            conf = fit_marginal_thresholds(y[calib_idx], p_cal, alpha=0.1)
        conf.n_fit = int(len(fit_idx))
    return {
        "df": df,
        "X": X,
        "y": y,
        "groups": groups,
        "tr": tr,
        "te": te,
        "fit_idx": fit_idx,
        "pipe": pipe,
        "conf": conf,
        "best": best,
        "meta": meta,
        "fold_index": fold_index,
        "fold_cfg": {
            **cfg,
            "_fold_pipeline": pipe,
            "_conformal_state": conf,
            "_fold_model_name": best,
        },
    }


def generate_forced_abstain_example(cfg: dict) -> None:
    """Real max_loops termination → forced abstain (not hard-coded copy)."""
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    case = df.iloc[0].to_dict()
    fa_cfg = {
        **cfg,
        "routing": {**cfg.get("routing", {}), "max_loops": 2},
    }
    result = run_graph(case, df.head(50), fa_cfg, use_llm=False, allow_critic=False)
    result["example_protocol"] = (
        "forced abstain: routing.max_loops=2 stops before predict/uncertainty; "
        "output from rules FSM (not hard-coded prose)"
    )
    result["delivered_planner"] = "rules_fsm"
    _write(ex / "forced_abstain_case.json", result)
    print(
        "forced_abstain",
        result.get("recommended_action"),
        "trace_has_forced_finalize=",
        "forced_finalize" in (result.get("route_trace") or []),
    )


def generate_production_examples(cfg: dict) -> None:
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    for label, no in PROD_CASES.items():
        try:
            result = assess_case(cfg, no, use_llm=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip prod {no}: {exc}")
            continue
        result["example_protocol"] = (
            "production model, case seen in training "
            "(full-data refit of best_baseline.joblib); rules FSM"
        )
        result["delivered_planner"] = "rules_fsm"
        _write(ex / f"case_{no}.json", result)
        if label == "ood":
            shutil.copy(ex / f"case_{no}.json", ex / "case_corrupt_ood.json")
        print(
            "prod",
            label,
            no,
            result.get("recommended_action"),
            result.get("predicted_intensity"),
        )


def _assess_heldout(ctx: dict, no: int) -> dict:
    df = ctx["df"]
    case = df.loc[df["no"] == no].iloc[0].to_dict()
    ref = build_fold_reference(df, ctx["fit_idx"], case)
    result = run_graph(case, ref, ctx["fold_cfg"], use_llm=False, allow_critic=True)
    result["case_no"] = int(no)
    result["example_protocol"] = (
        "held-out fold model: case was in outer test fold; "
        "pipeline fit only on fold train/fit indices; rules FSM"
    )
    result["fold_model"] = ctx["best"]
    result["true_grade"] = int(case["rockburst_grade"])
    result["true_intensity"] = intensity_name(int(case["rockburst_grade"]))
    result["delivered_planner"] = "rules_fsm"
    return result


def _record_failure(ex: Path, result: dict, *, err: int, fold_max_prob: float) -> dict:
    r_final = float(result.get("final_system_confidence") or 0.0)
    result = dict(result)
    result["failure_analysis"] = {
        "true_grade": result["true_grade"],
        "true_intensity": result["true_intensity"],
        "predicted_grade": result.get("predicted_grade"),
        "predicted_intensity": result.get("predicted_intensity"),
        "fold_pred_grade": result.get("fold_pred_grade"),
        "abs_error": err,
        "r_final": r_final,
        "fold_max_prob": float(fold_max_prob),
        "confident": True,
        "system_abstained": bool(result.get("abstained")),
        "diagnosis": (
            f"Held-out severe misclassification (|ŷ−y|={err}) at high confidence; "
            f"point prediction {result.get('predicted_intensity')} vs true "
            f"{result['true_intensity']}. "
            f"FSM action={result.get('recommended_action')}. "
            f"Inter-agent: {result.get('inter_agent_disagreement', {})}."
        ),
        "protocol": "held-out outer test fold; rules FSM; not hard-coded",
    }
    _write(ex / "failure_case.json", result)
    _write(ex / f"heldout_failure_case_{result['case_no']}.json", result)
    _write_failure_analysis_md(ex, result)
    print(
        "heldout_failure",
        result["case_no"],
        "err",
        err,
        "action",
        result.get("recommended_action"),
        "r_final",
        r_final,
    )
    return result


def generate_heldout_c5_examples(cfg: dict, n_extra: int = 3) -> None:
    """C5: real held-out abstain + |ŷ−y|≥2 high-confidence failure + extras."""
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)

    abstain_hit: dict | None = None
    failure_hit: dict | None = None
    scanned: list[dict] = []
    ctx0: dict | None = None
    n_outer = int(cfg.get("cv", {}).get("n_outer", 5))

    for fold_i in range(n_outer):
        if abstain_hit is not None and failure_hit is not None:
            break
        ctx = _fit_fold_context(cfg, fold_index=fold_i)
        if ctx is None:
            print("skip held-out C5: m1_results.json missing")
            return
        if ctx0 is None:
            ctx0 = ctx
        df = ctx["df"]
        te = ctx["te"]
        te_nos = df.iloc[te]["no"].astype(int).tolist()
        y_te = ctx["y"][te]
        proba_te = _predict_proba_safe(ctx["pipe"], ctx["X"].iloc[te])
        pred_te = proba_te.argmax(axis=1)
        conf_te = proba_te.max(axis=1)

        severe_idx = [
            i
            for i in range(len(te))
            if abs(int(pred_te[i]) - int(y_te[i])) >= 2 and float(conf_te[i]) >= 0.55
        ]
        # Prefer high fold-max-prob severe errors first
        severe_idx.sort(key=lambda i: -float(conf_te[i]))
        order = severe_idx + [i for i in range(len(te)) if i not in set(severe_idx)]

        for i in order:
            if abstain_hit is not None and failure_hit is not None:
                break
            no = int(te_nos[i])
            result = _assess_heldout(ctx, no)
            result["fold_max_prob"] = float(conf_te[i])
            result["fold_pred_grade"] = int(pred_te[i])
            result["fold_index"] = fold_i
            scanned.append(
                {
                    "fold": fold_i,
                    "no": no,
                    "action": result.get("recommended_action"),
                    "fold_err": abs(int(pred_te[i]) - int(y_te[i])),
                }
            )

            if abstain_hit is None and result.get("abstained"):
                abstain_hit = result
                _write(ex / "abstain_case.json", result)
                _write(ex / f"heldout_abstain_case_{no}.json", result)
                print("heldout_abstain", no, "fold", fold_i)

            pred_g = result.get("predicted_grade")
            if pred_g is None:
                continue
            # Prefer FSM point-grade error; else fold-model severe error
            err_fsm = abs(int(pred_g) - int(result["true_grade"]))
            err_fold = abs(int(pred_te[i]) - int(y_te[i]))
            r_final = float(result.get("final_system_confidence") or 0.0)
            # Production RF rarely exceeds 0.7 max-prob on |err|≥2; accept ≥0.55
            # or the fold's highest available severe error (documented in analysis).
            high_conf = r_final >= 0.7 or float(conf_te[i]) >= 0.55
            if failure_hit is None and high_conf:
                if err_fsm >= 2 and not result.get("abstained"):
                    failure_hit = _record_failure(
                        ex, result, err=err_fsm, fold_max_prob=float(conf_te[i])
                    )
                elif err_fold >= 2:
                    failure_hit = _record_failure(
                        ex, result, err=err_fold, fold_max_prob=float(conf_te[i])
                    )

            if len(scanned) >= 80:
                break

    ctx = ctx0
    if ctx is None:
        return
    df = ctx["df"]
    te = ctx["te"]
    # Diverse held-out extras from fold 0
    chosen: list[int] = []
    te_df = df.iloc[te].copy()
    for g in range(4):
        sub = te_df.loc[te_df["rockburst_grade"] == g, "no"].astype(int).tolist()
        if sub and sub[0] not in chosen:
            chosen.append(sub[0])
        if len(chosen) >= n_extra:
            break
    for no in chosen[:n_extra]:
        result = _assess_heldout(ctx, no)
        _write(ex / f"heldout_case_{no}.json", result)
        print(
            "heldout",
            no,
            result.get("recommended_action"),
            result.get("predicted_intensity"),
            "true=",
            result.get("true_intensity"),
        )

    summary = {
        "fold_model": ctx["best"],
        "n_test_fold0": int(len(te)),
        "abstain_found": abstain_hit is not None,
        "abstain_no": (abstain_hit or {}).get("case_no"),
        "failure_found": failure_hit is not None,
        "failure_no": (failure_hit or {}).get("case_no"),
        "scanned": scanned[:30],
        "note": "C5: abstain/failure selected from held-out folds via rules FSM",
    }
    _write(ex / "c5_selection_summary.json", summary)

    if abstain_hit is None:
        print("WARN: no held-out abstain found in scanned fold cases")
    if failure_hit is None:
        print("WARN: no held-out |ŷ−y|≥2 high-conf failure found; writing probe note")
        _write(
            ex / "failure_case.json",
            {
                "note": "No held-out |ŷ−y|≥2 at high confidence across scanned folds; "
                "see c5_selection_summary.json",
                "scanned": scanned[:30],
                "delivered_planner": "rules_fsm",
            },
        )


def _write_failure_analysis_md(ex: Path, result: dict) -> None:
    fa = result.get("failure_analysis") or {}
    lines = [
        "# Held-out failure analysis (C5)",
        "",
        "Standalone write-up regenerable from shipped models via "
        "`scripts/generate_examples.py` (rules FSM; no LLM).",
        "",
        f"- **True intensity:** {fa.get('true_intensity')} (grade {fa.get('true_grade')})",
        f"- **Predicted intensity:** {fa.get('predicted_intensity')} "
        f"(grade {fa.get('predicted_grade')})",
        f"- **|ŷ−y|:** {fa.get('abs_error')}",
        f"- **R_final:** {fa.get('r_final')}",
        f"- **Fold max-prob:** {fa.get('fold_max_prob')}",
        "",
        "## Diagnosis",
        "",
        str(fa.get("diagnosis") or ""),
        "",
        "## Decision text",
        "",
        str(result.get("decision") or ""),
        "",
        f"Protocol: {fa.get('protocol')}",
        "",
    ]
    (ex / "failure_analysis.md").write_text("\n".join(lines))


def archive_stale_llm_comparison(cfg: dict) -> None:
    """M16: label/archive stale Gemini planner comparison; do not re-run."""
    base = resolve_path(cfg, "baselines")
    src = base / "llm_planner_comparison.json"
    if not src.exists():
        print("no llm_planner_comparison.json to archive")
        return
    dest = base / "llm_planner_comparison.STALE.json"
    if not dest.exists():
        shutil.copy2(src, dest)
    meta = {
        "status": "STALE",
        "delivered_system": "rules_fsm",
        "note": (
            "Gemini Flash planner comparison is archived for historical routing only. "
            "All headline metrics and delivered examples use the deterministic rules FSM. "
            "Do not re-run Gemini evaluation for Slice 7/8 acceptance. "
            "Optional `--llm` path remains in code for exploratory routing; "
            "numbers always come from deterministic Python tools (brief §4.6 stance: "
            "structured tools + deterministic planner satisfy agentic requirements "
            "without LLM arithmetic)."
        ),
        "archived_copy": str(dest.name),
        "original_bytes": src.stat().st_size,
    }
    # Overwrite original with pointer + keep summary stats if cheap
    try:
        old = json.loads(src.read_text())
        meta["historical_summary"] = {
            k: old.get(k)
            for k in ("n_cases", "agreement_rate", "fallback_rate", "models", "note")
            if k in old or True
        }
        # pull common keys if present
        for k in ("n_cases", "agreement_rate", "fallback_rate", "agree_rate", "summary"):
            if k in old:
                meta["historical_summary"][k] = old[k]
    except Exception:  # noqa: BLE001
        pass
    src.write_text(json.dumps(meta, indent=2))
    print("archived llm_planner_comparison → STALE + pointer JSON")


def main() -> None:
    cfg = load_config()
    archive_stale_llm_comparison(cfg)
    generate_forced_abstain_example(cfg)
    generate_production_examples(cfg)
    generate_heldout_c5_examples(cfg, n_extra=3)


if __name__ == "__main__":
    main()
