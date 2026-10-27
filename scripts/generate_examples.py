#!/usr/bin/env python3
"""Generate evidence package examples/traces (Phase 6 / RC7–RC8).

- Forced abstain via real production Critic→recheck cycle limit (RC5).
- Production showcase cases (full-data refit disclaimer).
- Held-out: correct abstain (separate) + accepted-wrong |ŷ−y|≥2 failure (RC7).
- case_corrupt_ood: actually perturbed input (RC8); no shutil.copy of case_146.
- Relative paths only; case_no / true_grade / fold / model on every example.
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
from rockburst.config import ROOT, load_config, resolve_path
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

_REPO_ABS = str(ROOT.resolve())


# Production-model showcase cases (full-data refit — labels include disclaimer).
PROD_CASES = {
    "standard_a": 15,
    "physics_b1": 20,
    "standard_b": 67,
    "ood": 146,
    "physics_b1_244": 244,
    "near_dup": 180,
}

# Prefer this case class for accepted-wrong failure write-up (RC7).
PREFERRED_FAILURE_CASE_NO = 42

# Fields every shipped example JSON must carry (regen-check / RC8).
REQUIRED_EXAMPLE_FIELDS = (
    "case_no",
    "true_grade",
    "fold_index",
    "fold_model",
    "example_protocol",
    "delivered_planner",
)


def _repo_rel(path: Path | str) -> str:
    """RC8: path relative to repo root (never /home/...)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        s = str(path)
        if s.startswith(_REPO_ABS):
            return s[len(_REPO_ABS) :].lstrip("/")
        for marker in ("artifacts/", "data/", "notebooks/", "reports/"):
            idx = s.find(marker)
            if idx >= 0:
                return s[idx:]
        return s


def _strip_abs_paths(obj: object) -> object:
    """Recursively rewrite absolute repo /home paths to repo-relative strings."""
    if isinstance(obj, dict):
        return {k: _strip_abs_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_abs_paths(v) for v in obj]
    if isinstance(obj, str) and ("/home/" in obj or obj.startswith(_REPO_ABS)):
        return _repo_rel(obj)
    return obj


def _write(path: Path, payload: dict) -> None:
    clean = _strip_abs_paths(payload)
    assert isinstance(clean, dict)
    path.write_text(json.dumps(clean, indent=2, default=str))


def _rel_protocol(kind: str, **extra: object) -> str:
    bits = [kind, "rules FSM", "relative paths only"]
    for k, v in extra.items():
        bits.append(f"{k}={v}")
    return "; ".join(bits)


def _prune_stale_heldout(ex: Path, *, keep_nos: set[int]) -> None:
    """Remove heldout_* siblings that are not part of the current selection (RC7)."""
    for p in ex.glob("heldout_*.json"):
        # heldout_case_N / heldout_abstain_case_N / heldout_failure_case_N
        stem = p.stem
        try:
            no = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if no not in keep_nos:
            p.unlink(missing_ok=True)
            print("pruned_stale", p.name)


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
    """Drive a real case to the production Critic→recheck cycle limit (RC5)."""
    import rockburst.agents.tools as tools_mod
    import rockburst.graph.runner as runner_mod
    from rockburst.agents.schemas import CriticOutput

    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    case_no = 146
    row = df.loc[df["no"] == case_no]
    if row.empty:
        case = df.iloc[0].to_dict()
        case_no = int(case.get("no", 0))
    else:
        case = row.iloc[0].to_dict()

    rt = dict(cfg.get("routing", {}))
    max_cycles = int(rt.get("max_critic_recheck_cycles", 3))
    fa_cfg = {**cfg, "routing": rt}

    original_critic = tools_mod.run_critic

    def _always_object(*args, **kwargs):
        return CriticOutput(
            challenges=["Forced non-convergence demo: persistent Critic objection"],
            request_recheck="physics",
            endorse=False,
            rules_fired=["demo_persistent_objection"],
            escalate_abstain=False,
        )

    tools_mod.run_critic = _always_object  # type: ignore[assignment]
    runner_mod.run_critic = _always_object  # type: ignore[assignment]
    try:
        result = run_graph(case, df, fa_cfg, use_llm=False, allow_critic=True)
    finally:
        tools_mod.run_critic = original_critic  # type: ignore[assignment]
        runner_mod.run_critic = original_critic  # type: ignore[assignment]

    result["case_no"] = int(case_no)
    result["true_grade"] = int(case.get("rockburst_grade", -1))
    result["true_intensity"] = intensity_name(int(case.get("rockburst_grade", -1)))
    result["fold_index"] = None
    result["fold_model"] = "production_full_refit"
    result["example_protocol"] = _rel_protocol(
        "forced_abstain",
        case_no=case_no,
        max_critic_recheck_cycles=max_cycles,
    )
    result["delivered_planner"] = "rules_fsm"
    result["production_max_critic_recheck_cycles"] = max_cycles
    result["production_max_loops"] = int(rt.get("max_loops", 24))
    _write(ex / "forced_abstain_case.json", result)
    print(
        "forced_abstain",
        "case",
        case_no,
        result.get("recommended_action"),
        "termination=",
        result.get("termination_reason"),
        "cycles=",
        result.get("critic_recheck_cycles"),
    )


def generate_production_examples(cfg: dict) -> None:
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    for label, no in PROD_CASES.items():
        try:
            result = assess_case(cfg, no, use_llm=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip prod {no}: {exc}")
            continue
        row = df.loc[df["no"] == no]
        true_g = int(row.iloc[0]["rockburst_grade"]) if not row.empty else None
        result["case_no"] = int(no)
        result["true_grade"] = true_g
        result["true_intensity"] = intensity_name(true_g) if true_g is not None else None
        result["fold_index"] = None
        result["fold_model"] = "production_full_refit"
        result["example_protocol"] = _rel_protocol(
            "production_showcase",
            note="case_seen_in_full_data_refit",
        )
        result["delivered_planner"] = "rules_fsm"
        result["showcase_label"] = label
        _write(ex / f"case_{no}.json", result)
        print(
            "prod",
            label,
            no,
            result.get("recommended_action"),
            result.get("predicted_intensity"),
        )


def generate_corrupt_ood_example(cfg: dict) -> None:
    """RC8: actually perturb inputs (do not copy case_146)."""
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    base_no = 146
    row = df.loc[df["no"] == base_no]
    if row.empty:
        row = df.iloc[[0]]
        base_no = int(row.iloc[0]["no"])
    case = row.iloc[0].to_dict()
    true_g = int(case["rockburst_grade"])

    # Recorded physical corruption: extreme stress + Wet + depth OOD.
    perturbation = {
        "base_case_no": int(base_no),
        "ops": [
            {"field": "sigma_theta", "op": "multiply", "factor": 8.0},
            {"field": "wet", "op": "set", "value": 99.0},
            {"field": "d", "op": "set", "value": 1.0},
            {"field": "scf", "op": "recompute_note", "value": "left_as_listed_then_physics_flags"},
        ],
    }
    corrupted = dict(case)
    corrupted["sigma_theta"] = float(case["sigma_theta"]) * 8.0
    corrupted["wet"] = 99.0
    corrupted["d"] = 1.0
    # Keep listed SCF inconsistent with new σθ so physics / OOD fire.
    result = run_graph(corrupted, df, cfg, use_llm=False, allow_critic=True)
    result["case_no"] = int(base_no)
    result["true_grade"] = true_g
    result["true_intensity"] = intensity_name(true_g)
    result["fold_index"] = None
    result["fold_model"] = "production_full_refit"
    result["example_protocol"] = _rel_protocol(
        "corrupt_ood",
        base_case=base_no,
        note="inputs_actually_perturbed",
    )
    result["delivered_planner"] = "rules_fsm"
    result["perturbation"] = perturbation
    result["corrupted_inputs"] = {
        "sigma_theta": corrupted["sigma_theta"],
        "wet": corrupted["wet"],
        "d": corrupted["d"],
        "sigma_c": corrupted.get("sigma_c"),
        "sigma_t": corrupted.get("sigma_t"),
        "scf_listed": corrupted.get("scf"),
    }
    # Drop any absolute trace_path if a prior assess left one; writers strip /home/.
    result.pop("trace_path", None)
    out_path = ex / "case_corrupt_ood.json"
    _write(out_path, result)
    twin = ex / f"case_{base_no}.json"
    if twin.exists() and out_path.read_bytes() == twin.read_bytes():
        raise AssertionError(
            f"RC8: case_corrupt_ood.json must not be a byte-copy of case_{base_no}.json"
        )
    if "perturbation" not in json.loads(out_path.read_text()):
        raise AssertionError("RC8: case_corrupt_ood.json must record perturbation")
    print(
        "corrupt_ood",
        "base",
        base_no,
        result.get("recommended_action"),
        "ood=",
        result.get("ood_status"),
        "perturbs≠copy",
    )


def _assess_heldout(ctx: dict, no: int) -> dict:
    df = ctx["df"]
    case = df.loc[df["no"] == no].iloc[0].to_dict()
    ref = build_fold_reference(df, ctx["fit_idx"], case)
    result = run_graph(case, ref, ctx["fold_cfg"], use_llm=False, allow_critic=True)
    result["case_no"] = int(no)
    result["example_protocol"] = _rel_protocol(
        "held_out_fold",
        fold=ctx["fold_index"],
        model=ctx["best"],
    )
    result["fold_model"] = ctx["best"]
    result["fold_index"] = int(ctx["fold_index"])
    result["true_grade"] = int(case["rockburst_grade"])
    result["true_intensity"] = intensity_name(int(case["rockburst_grade"]))
    result["delivered_planner"] = "rules_fsm"
    result["inputs"] = {
        k: case.get(k)
        for k in (
            "no",
            "sigma_theta",
            "sigma_c",
            "sigma_t",
            "scf",
            "b1",
            "b2",
            "wet",
            "d",
            "rockburst_grade",
        )
    }
    return result


def _agent_snapshot(result: dict) -> dict:
    """Compact per-agent outputs / flags for failure write-up (RC7)."""
    return {
        "route_trace": result.get("route_trace"),
        "termination_reason": result.get("termination_reason"),
        "critic_recheck_cycles": result.get("critic_recheck_cycles"),
        "ood_status": result.get("ood_status"),
        "data_reliability": result.get("data_reliability"),
        "physics_score": result.get("physics_score"),
        "r_model": result.get("r_model"),
        "prediction_set": result.get("prediction_set"),
        "prediction_set_intensities": result.get("prediction_set_intensities"),
        "model_disagreement": result.get("model_disagreement"),
        "critic_endorse": result.get("critic_endorse"),
        "critic_rules_fired": result.get("critic_rules_fired")
        or (result.get("inter_agent_disagreement") or {}).get("critic_rules_fired"),
        "inter_agent_disagreement": result.get("inter_agent_disagreement"),
        "checks_skipped": result.get("physics_skipped") or result.get("skipped_checks"),
        "decision_override_reason": result.get("decision_override_reason"),
        "recommended_action": result.get("recommended_action"),
        "abstained": result.get("abstained"),
        "final_system_confidence": result.get("final_system_confidence"),
    }


def _would_have_caught(result: dict, err: int) -> list[str]:
    """Which checks could have caught this accepted wrong prediction (RC7)."""
    ideas: list[str] = []
    rules = list(
        result.get("critic_rules_fired")
        or (result.get("inter_agent_disagreement") or {}).get("critic_rules_fired")
        or []
    )
    if err >= 2:
        ideas.append(
            "monotonic_tendency / criteria consensus: large |ŷ−y| with high R_final "
            "should escalate to abstain before accept"
        )
    if result.get("ood_status") in {"Yes", True, "yes"}:
        ideas.append("OOD∧weak-physics hard gate (already present; verify escalate)")
    set_ = result.get("prediction_set") or []
    if len(set_) >= 3:
        ideas.append("wide conformal edge-of-set check (predicted grade at edge of |set|≥3)")
    if "ml_vs_criteria" in rules:
        ideas.append("ml_vs_criteria already fired — escalate_abstain severity from precision")
    if not ideas:
        ideas.append(
            "Prefer abstain when fold max-prob severe error (|ŷ−y|≥2) and Critic "
            "non-endorse; add graded disagreement / monotonic series context"
        )
    return ideas


def _record_failure(
    ex: Path,
    result: dict,
    *,
    err: int,
    fold_max_prob: float,
) -> dict:
    """RC7: accepted-wrong failure only (caller must ensure not abstained)."""
    if result.get("abstained") or result.get("recommended_action") == "abstain":
        raise ValueError("RC7: failure_case must be accepted-wrong, not abstain")
    r_final = float(result.get("final_system_confidence") or 0.0)
    result = dict(result)
    agents = _agent_snapshot(result)
    catch = _would_have_caught(result, err)
    result["failure_analysis"] = {
        "kind": "accepted_wrong",
        "true_grade": result["true_grade"],
        "true_intensity": result["true_intensity"],
        "predicted_grade": result.get("predicted_grade"),
        "predicted_intensity": result.get("predicted_intensity"),
        "fold_pred_grade": result.get("fold_pred_grade"),
        "abs_error": err,
        "r_final": r_final,
        "fold_max_prob": float(fold_max_prob),
        "confident": True,
        "system_abstained": False,
        "inputs": result.get("inputs"),
        "agent_outputs_flags": agents,
        "checks_fired_or_skipped": {
            "critic_rules_fired": agents.get("critic_rules_fired"),
            "checks_skipped": agents.get("checks_skipped"),
            "route_trace": agents.get("route_trace"),
        },
        "why_wrong": (
            f"Held-out accepted prediction with |ŷ−y|={err} "
            f"({result.get('predicted_intensity')} vs true {result['true_intensity']}) "
            f"at R_final={r_final:.3f}, fold_max_prob={fold_max_prob:.3f}. "
            f"Action={result.get('recommended_action')} (not abstain)."
        ),
        "which_check_would_catch": catch,
        "diagnosis": (
            f"Accepted-wrong severe misclassification (|ŷ−y|={err}); "
            f"point {result.get('predicted_intensity')} vs true {result['true_intensity']}. "
            f"FSM action={result.get('recommended_action')}. "
            f"Inter-agent: {result.get('inter_agent_disagreement', {})}."
        ),
        "protocol": "held-out outer test fold; rules FSM; accepted-wrong only (RC7)",
    }
    _write(ex / "failure_case.json", result)
    _write(ex / f"heldout_failure_case_{result['case_no']}.json", result)
    _write_failure_analysis_md(ex, result)
    print(
        "heldout_failure_accepted_wrong",
        result["case_no"],
        "err",
        err,
        "action",
        result.get("recommended_action"),
        "r_final",
        r_final,
    )
    return result


def _is_accepted_wrong(result: dict) -> bool:
    return (
        not result.get("abstained")
        and result.get("recommended_action") not in {"abstain", None}
        and result.get("predicted_grade") is not None
    )


def _try_record_accepted_failure(
    ex: Path,
    result: dict,
    *,
    err: int,
    fold_max_prob: float,
    abstain_no: int | None,
    min_err: int,
) -> dict | None:
    """Record accepted-wrong failure iff distinct from abstain and |ŷ−y|≥min_err."""
    if not _is_accepted_wrong(result):
        return None
    err_fsm = abs(int(result["predicted_grade"]) - int(result["true_grade"]))
    if err_fsm < min_err:
        return None
    no = int(result["case_no"])
    if abstain_no is not None and no == int(abstain_no):
        return None
    r_final = float(result.get("final_system_confidence") or 0.0)
    high_conf = r_final >= 0.55 or float(fold_max_prob) >= 0.55
    if not high_conf and min_err >= 2:
        return None
    return _record_failure(ex, result, err=err_fsm, fold_max_prob=float(fold_max_prob))


def generate_heldout_c5_examples(cfg: dict, n_extra: int = 3) -> None:
    """RC7: separate held-out abstain vs accepted-wrong failure (prefer |ŷ−y|≥2)."""
    ex = resolve_path(cfg, "examples")
    ex.mkdir(parents=True, exist_ok=True)

    abstain_hit: dict | None = None
    failure_hit: dict | None = None
    scanned: list[dict] = []
    ctx0: dict | None = None
    n_outer = int(cfg.get("cv", {}).get("n_outer", 5))
    # Cache fold assessments so fallback |ŷ−y|≥1 does not re-run the graph.
    cache: dict[tuple[int, int], dict] = {}

    preferred_ctx: dict | None = None
    preferred_te_i: int | None = None
    preferred_fold_i: int | None = None

    for fold_i in range(n_outer):
        ctx = _fit_fold_context(cfg, fold_index=fold_i)
        if ctx is None:
            print("skip held-out C5: m1_results.json missing")
            return
        if ctx0 is None:
            ctx0 = ctx
        df = ctx["df"]
        te = ctx["te"]
        te_nos = df.iloc[te]["no"].astype(int).tolist()
        if PREFERRED_FAILURE_CASE_NO in te_nos:
            preferred_ctx = ctx
            preferred_te_i = te_nos.index(PREFERRED_FAILURE_CASE_NO)
            preferred_fold_i = fold_i

        y_te = ctx["y"][te]
        proba_te = _predict_proba_safe(ctx["pipe"], ctx["X"].iloc[te])
        pred_te = proba_te.argmax(axis=1)
        conf_te = proba_te.max(axis=1)

        severe_idx = [
            i
            for i in range(len(te))
            if abs(int(pred_te[i]) - int(y_te[i])) >= 2 and float(conf_te[i]) >= 0.55
        ]
        severe_idx.sort(key=lambda i: -float(conf_te[i]))
        order = severe_idx + [i for i in range(len(te)) if i not in set(severe_idx)]

        for i in order:
            if abstain_hit is not None and failure_hit is not None:
                break
            no = int(te_nos[i])
            result = _assess_heldout(ctx, no)
            result["fold_max_prob"] = float(conf_te[i])
            result["fold_pred_grade"] = int(pred_te[i])
            cache[(fold_i, no)] = result
            scanned.append(
                {
                    "fold": fold_i,
                    "no": no,
                    "action": result.get("recommended_action"),
                    "abstained": bool(result.get("abstained")),
                    "fold_err": abs(int(pred_te[i]) - int(y_te[i])),
                }
            )

            failure_no = (failure_hit or {}).get("case_no")
            # (i) correct abstain — must not reuse the failure case_no
            if (
                abstain_hit is None
                and result.get("abstained")
                and (failure_no is None or int(no) != int(failure_no))
            ):
                abstain_hit = result
                _write(ex / "abstain_case.json", result)
                _write(ex / f"heldout_abstain_case_{no}.json", result)
                print("heldout_abstain", no, "fold", fold_i)

            abstain_no = (abstain_hit or {}).get("case_no")
            if failure_hit is None:
                failure_hit = _try_record_accepted_failure(
                    ex,
                    result,
                    err=0,
                    fold_max_prob=float(conf_te[i]),
                    abstain_no=abstain_no,
                    min_err=2,
                )

            if len(scanned) >= 100:
                break
        if abstain_hit is not None and failure_hit is not None:
            break

    # Prefer case 42 if still looking for failure and it qualifies (|ŷ−y|≥2)
    if failure_hit is None and preferred_ctx is not None and preferred_te_i is not None:
        ctx = preferred_ctx
        te = ctx["te"]
        conf_te = _predict_proba_safe(ctx["pipe"], ctx["X"].iloc[te]).max(axis=1)
        i = preferred_te_i
        key = (int(preferred_fold_i or 0), PREFERRED_FAILURE_CASE_NO)
        result = cache.get(key) or _assess_heldout(ctx, PREFERRED_FAILURE_CASE_NO)
        result["fold_max_prob"] = float(conf_te[i])
        result["fold_pred_grade"] = int(
            _predict_proba_safe(ctx["pipe"], ctx["X"].iloc[te]).argmax(axis=1)[i]
        )
        cache[key] = result
        failure_hit = _try_record_accepted_failure(
            ex,
            result,
            err=0,
            fold_max_prob=float(conf_te[i]),
            abstain_no=(abstain_hit or {}).get("case_no"),
            min_err=2,
        )

    # Fallback: accepted |ŷ−y|≥1 if no severe accepted-wrong found (RC7)
    if failure_hit is None:
        for (fold_i, no), result in cache.items():
            hit = _try_record_accepted_failure(
                ex,
                result,
                err=0,
                fold_max_prob=float(result.get("fold_max_prob") or 0.0),
                abstain_no=(abstain_hit or {}).get("case_no"),
                min_err=1,
            )
            if hit is not None:
                failure_hit = hit
                print("heldout_failure_fallback_abs_err>=1", no, "fold", fold_i)
                break

    ctx = ctx0
    if ctx is None:
        return
    df = ctx["df"]
    te = ctx["te"]
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

    # Routing traces for F3 (missingness / OOD / Critic recheck)
    _write_routing_traces(ex, ctx, abstain_hit=abstain_hit)

    abstain_no = (abstain_hit or {}).get("case_no")
    failure_no = (failure_hit or {}).get("case_no")
    if (
        abstain_hit is not None
        and failure_hit is not None
        and abstain_no is not None
        and failure_no is not None
        and int(abstain_no) == int(failure_no)
    ):
        raise AssertionError(
            f"RC7: abstain_case and failure_case must have distinct case_no; "
            f"both are {abstain_no}"
        )

    keep_nos: set[int] = set(chosen[:n_extra])
    if abstain_no is not None:
        keep_nos.add(int(abstain_no))
    if failure_no is not None:
        keep_nos.add(int(failure_no))
    _prune_stale_heldout(ex, keep_nos=keep_nos)

    summary = {
        "fold_model": ctx["best"],
        "n_test_fold0": int(len(te)),
        "abstain_found": abstain_hit is not None,
        "abstain_no": abstain_no,
        "failure_found": failure_hit is not None,
        "failure_no": failure_no,
        "failure_kind": "accepted_wrong",
        "failure_abs_error": (failure_hit or {})
        .get("failure_analysis", {})
        .get("abs_error"),
        "cases_distinct": (
            abstain_no is None
            or failure_no is None
            or int(abstain_no) != int(failure_no)
        ),
        "preferred_failure_case_no": PREFERRED_FAILURE_CASE_NO,
        "scanned": scanned[:40],
        "note": (
            "RC7: abstain_case = held-out abstain; failure_case = held-out "
            "accepted-wrong preferring |ŷ−y|≥2 (fallback ≥1); "
            "failure_no ≠ abstain_no; case 42 class when available"
        ),
    }
    assert summary["cases_distinct"], "RC7: abstain_no and failure_no must differ"
    _write(ex / "c5_selection_summary.json", summary)

    if abstain_hit is None:
        print("WARN: no held-out abstain found in scanned fold cases")
    if failure_hit is None:
        print("WARN: no held-out accepted-wrong found; writing probe note")
        _write(
            ex / "failure_case.json",
            {
                "note": "No held-out accepted-wrong (|ŷ−y|≥2 then ≥1) across scanned "
                "folds; see c5_selection_summary.json",
                "scanned": scanned[:40],
                "delivered_planner": "rules_fsm",
                "failure_analysis": {"kind": "none_found"},
            },
        )


def _write_routing_traces(
    ex: Path, ctx: dict, *, abstain_hit: dict | None
) -> None:
    """Fresh F3 traces: missingness, OOD, executed Critic recheck."""
    df = ctx["df"]
    te = ctx["te"]
    fit_idx = ctx["fit_idx"]
    traces = {"missingness": None, "ood": None, "critic": None}
    for j in te:
        if all(traces.values()):
            break
        case = df.iloc[int(j)].to_dict()
        ref = build_fold_reference(df, fit_idx, case)
        out = run_graph(case, ref, ctx["fold_cfg"], use_llm=False, allow_critic=True)
        out["case_no"] = int(case["no"])
        out["true_grade"] = int(case["rockburst_grade"])
        out["true_intensity"] = intensity_name(int(case["rockburst_grade"]))
        out["fold_index"] = int(ctx["fold_index"])
        out["fold_model"] = ctx["best"]
        out["example_protocol"] = _rel_protocol("routing_trace", fold=ctx["fold_index"])
        out["delivered_planner"] = "rules_fsm"
        trace = " ".join(out.get("route_trace") or [])
        d_miss = case.get("d") is None or (
            isinstance(case.get("d"), float) and not np.isfinite(float(case.get("d")))
        )
        if traces["missingness"] is None and d_miss and "extra_validation" in trace:
            traces["missingness"] = out
        if traces["ood"] is None and out.get("ood_status") in {"Yes", True, "yes"}:
            traces["ood"] = out
        if (
            traces["critic"] is None
            and "recheck" in trace
            and out.get("critic_endorse") is False
        ):
            traces["critic"] = out
    if traces["critic"] is None and abstain_hit is not None:
        traces["critic"] = abstain_hit
    for key, blob in traces.items():
        if blob is None:
            continue
        _write(ex / f"trace_{key}.json", blob)
        print("trace", key, "case", blob.get("case_no"))


def _write_failure_analysis_md(ex: Path, result: dict) -> None:
    fa = result.get("failure_analysis") or {}
    agents = fa.get("agent_outputs_flags") or {}
    catch = fa.get("which_check_would_catch") or []
    checks = fa.get("checks_fired_or_skipped") or {}
    skips = (
        agents.get("checks_skipped")
        or checks.get("checks_skipped")
        or result.get("physics_skipped")
        or result.get("skipped_checks")
        or []
    )
    lines = [
        "# Held-out failure analysis (RC7 — accepted wrong)",
        "",
        "Standalone write-up regenerable via `scripts/generate_examples.py` "
        "(rules FSM; no LLM). Separate from held-out **abstain** examples "
        "(failure `case_no` must differ from abstain `case_no`).",
        "",
        f"- **Case no:** {result.get('case_no')}",
        f"- **Fold / model:** {result.get('fold_index')} / {result.get('fold_model')}",
        f"- **True intensity:** {fa.get('true_intensity')} (grade {fa.get('true_grade')})",
        f"- **Predicted intensity:** {fa.get('predicted_intensity')} "
        f"(grade {fa.get('predicted_grade')})",
        f"- **|ŷ−y|:** {fa.get('abs_error')} (prefer ≥2; fallback ≥1)",
        f"- **R_final:** {fa.get('r_final')}",
        f"- **Fold max-prob:** {fa.get('fold_max_prob')}",
        f"- **Recommended action:** {result.get('recommended_action')}",
        f"- **Abstained:** {fa.get('system_abstained')} (must be false for this file)",
        f"- **Termination reason:** {result.get('termination_reason')}",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(fa.get("inputs") or result.get("inputs"), indent=2, default=str),
        "```",
        "",
        "## Agent outputs / flags",
        "",
        "```json",
        json.dumps(agents, indent=2, default=str)[:4000],
        "```",
        "",
        "## Checks skipped",
        "",
    ]
    if skips:
        for s in skips if isinstance(skips, list) else [skips]:
            lines.append(f"- {s}")
    else:
        lines.append("- (none recorded)")
    lines.extend(
        [
            "",
            "## Checks fired (Critic / route)",
            "",
            "```json",
            json.dumps(checks, indent=2, default=str)[:2000],
            "```",
            "",
            "## Why wrong",
            "",
            str(fa.get("why_wrong") or fa.get("diagnosis") or ""),
            "",
            "## Which check would have caught it",
            "",
        ]
    )
    for c in catch:
        lines.append(f"- {c}")
    lines.extend(
        [
            "",
            "## Decision text",
            "",
            str(result.get("decision") or ""),
            "",
            f"Protocol: {fa.get('protocol')}",
            "",
        ]
    )
    (ex / "failure_analysis.md").write_text("\n".join(lines))


def archive_stale_llm_comparison(cfg: dict) -> None:
    """Label/archive stale Gemini planner comparison; do not re-run."""
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
            "All headline metrics and delivered examples use the deterministic rules FSM."
        ),
        "archived_copy": str(dest.name),
        "original_bytes": src.stat().st_size,
    }
    try:
        old = json.loads(src.read_text())
        if old.get("status") == "STALE":
            print("llm_planner_comparison already STALE pointer")
            return
        meta["historical_summary"] = {
            k: old.get(k)
            for k in ("n_cases", "agreement_rate", "fallback_rate", "agree_rate", "summary")
            if k in old
        }
    except Exception:  # noqa: BLE001
        pass
    src.write_text(json.dumps(meta, indent=2))
    print("archived llm_planner_comparison → STALE + pointer JSON")


def example_schema_ok(blob: dict) -> list[str]:
    """Return missing REQUIRED_EXAMPLE_FIELDS (empty ⇒ ok)."""
    return [f for f in REQUIRED_EXAMPLE_FIELDS if f not in blob]


def main() -> None:
    cfg = load_config()
    # RC8 archive first so regen writes into a clean examples tree.
    # Do not use run_name="__main__": archive_stale_artefacts raises SystemExit
    # from its CLI guard and would abort the rest of this script.
    import runpy

    archive_ns = runpy.run_path(
        str(Path(__file__).resolve().parent / "archive_stale_artefacts.py"),
        run_name="archive_stale_artefacts",
    )
    archive_ns["main"]()

    archive_stale_llm_comparison(cfg)
    generate_forced_abstain_example(cfg)
    generate_production_examples(cfg)
    generate_corrupt_ood_example(cfg)
    generate_heldout_c5_examples(cfg, n_extra=3)


if __name__ == "__main__":
    main()
