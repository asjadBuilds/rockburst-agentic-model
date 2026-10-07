#!/usr/bin/env python3
"""Compare rules vs Gemini planner on ≥30 cases (resumable, cached)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rockburst.config import ROOT, load_config, resolve_path
from rockburst.graph.runner import PLANNER_STATS, run_graph


def pick_cases(df: pd.DataFrame, n: int = 32) -> list[int]:
    chosen: list[int] = []
    ood = df.loc[(df["scf"] > 2.0), "no"].astype(int).tolist()
    chosen.extend(ood[:8])
    miss = df.loc[df["d"].isna(), "no"].astype(int).tolist()
    for x in miss:
        if x not in chosen:
            chosen.append(x)
        if len(chosen) >= 16:
            break
    for x in (1, 20, 21, 146, 244, 6, 315):
        if x in set(df["no"].astype(int)) and x not in chosen:
            chosen.append(x)
    for g in range(4):
        for x in df.loc[df["rockburst_grade"] == g, "no"].astype(int).tolist():
            if x not in chosen:
                chosen.append(x)
            if len(chosen) >= n:
                return chosen[:n]
    return chosen[:n]


def validate_gemini_model(cfg: dict) -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    from google import genai

    client = genai.Client(api_key=key)
    configured = cfg.get("gemini", {}).get("planner_model", "gemini-3.8-flash")
    available = []
    for m in client.models.list():
        mid = getattr(m, "name", None) or str(m)
        available.append(mid)
    if configured in {a.split("/")[-1] for a in available} or any(
        configured in a for a in available
    ):
        return configured
    for cand in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"):
        if any(cand in a for a in available):
            return cand
    raise RuntimeError(f"No suitable Gemini model; configured={configured}")


def acts(tr):
    return [t.split(":", 1)[-1] for t in (tr or [])]


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    model_id = validate_gemini_model(cfg)
    print(f"Configured model OK: {model_id}", flush=True)
    if model_id != cfg.get("gemini", {}).get("planner_model"):
        import yaml

        cfg.setdefault("gemini", {})["planner_model"] = model_id
        cfg_path = ROOT / "configs" / "default.yaml"
        raw = yaml.safe_load(cfg_path.read_text())
        raw.setdefault("gemini", {})["planner_model"] = model_id
        raw["gemini"]["lite_model"] = model_id
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        print(f"Updated config to {model_id}", flush=True)

    nos = pick_cases(df, 32)
    out = resolve_path(cfg, "baselines") / "llm_planner_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = resolve_path(cfg, "traces")
    traces.mkdir(parents=True, exist_ok=True)

    rows = []
    if out.exists():
        prev = json.loads(out.read_text())
        rows = prev.get("cases") or []
        done = {r["no"] for r in rows}
        print(f"Resuming with {len(done)} done", flush=True)
    else:
        done = set()

    for k in PLANNER_STATS:
        PLANNER_STATS[k] = 0

    for i, no in enumerate(nos):
        if no in done:
            continue
        print(f"[{i+1}/{len(nos)}] case {no} …", flush=True)
        case = df.loc[df["no"] == no].iloc[0].to_dict()
        rules = run_graph(case, df, cfg, use_llm=False)
        try:
            llm = run_graph(case, df, cfg, use_llm=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  LLM failed: {exc}", flush=True)
            llm = {
                "route_trace": ["fallback:error"],
                "recommended_action": None,
                "error": str(exc),
            }
        action_agree = acts(rules.get("route_trace")) == acts(llm.get("route_trace"))
        row = {
            "no": no,
            "rules_trace": rules.get("route_trace"),
            "llm_trace": llm.get("route_trace"),
            "rules_decision": rules.get("recommended_action"),
            "llm_decision": llm.get("recommended_action"),
            "action_seq_agree": action_agree,
            "disagreement": not action_agree,
        }
        rows.append(row)
        (traces / f"case_{no}_llm.json").write_text(json.dumps(llm, indent=2))
        report = {
            "model_id": model_id,
            "n_cases": len(rows),
            "action_agreement_rate": sum(1 for r in rows if r["action_seq_agree"])
            / max(len(rows), 1),
            "planner_stats": dict(PLANNER_STATS),
            "fallback_rate": PLANNER_STATS["fallbacks"]
            / max(PLANNER_STATS["llm_steps"] + PLANNER_STATS["fallbacks"], 1),
            "disagreement_cases": [r for r in rows if r["disagreement"]],
            "cases": rows,
            "partial": len(rows) < len(nos),
        }
        out.write_text(json.dumps(report, indent=2))
        print(
            f"  agree={action_agree} stats={PLANNER_STATS} saved {len(rows)}/{len(nos)}",
            flush=True,
        )

    report = json.loads(out.read_text())
    report["partial"] = False
    report["n_requested"] = len(nos)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "model_id", "n_cases", "action_agreement_rate", "planner_stats", "fallback_rate"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
