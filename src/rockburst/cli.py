"""CLI entrypoints for extract / train / evaluate / assess."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rockburst.config import ROOT, load_config, resolve_path


def cmd_extract(args: argparse.Namespace) -> int:
    from rockburst.data import extract_table_from_pdf, process_raw, write_raw_csv

    cfg = load_config(args.config)
    pdf = Path(args.pdf) if args.pdf else ROOT / cfg["paths"]["pdf"]
    df, parse_audit = extract_table_from_pdf(pdf)
    write_raw_csv(df, resolve_path(cfg, "raw_csv"), parse_audit, resolve_path(cfg, "audit_json"))
    processed, data_audit = process_raw(df, cfg)
    out = resolve_path(cfg, "processed_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out, index=False)
    (out.parent / "data_audit.json").write_text(json.dumps(data_audit, indent=2))
    print(f"Wrote {resolve_path(cfg, 'raw_csv')} ({len(df)} rows)")
    print(f"Wrote {out}; D missing={data_audit['d_missing']}; dup_groups={data_audit['n_dup_groups']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    import pandas as pd

    cfg = load_config(args.config)
    path = resolve_path(cfg, "processed_csv")
    if not path.exists():
        path = resolve_path(cfg, "raw_csv")
    df = pd.read_csv(path)
    row = df.loc[df["no"] == args.no].iloc[0]
    print(row.to_json(indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from rockburst.ml.train import train_baselines

    cfg = load_config(args.config)
    summary = train_baselines(cfg, quick=args.quick)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from rockburst.eval.ablation import run_ablations

    cfg = load_config(args.config)
    summary = run_ablations(cfg, quick=args.quick)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_assess(args: argparse.Namespace) -> int:
    from rockburst.graph.runner import assess_case

    cfg = load_config(args.config)
    result = assess_case(cfg, case_no=args.no, use_llm=args.llm)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rockburst", description="Rockburst agentic assessment CLI")
    p.add_argument("--config", default=None, help="Path to YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract Table S1 PDF → CSV + audit")
    e.add_argument("--pdf", default=None)
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("show", help="Dump one case as JSON")
    s.add_argument("--no", type=int, default=1)
    s.set_defaults(func=cmd_show)

    t = sub.add_parser("train", help="Train leakage-safe baselines (M1)")
    t.add_argument("--quick", action="store_true", help="Smaller grid for smoke runs")
    t.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Run ablations M1–M7 + robustness")
    ev.add_argument("--quick", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("assess", help="Run agentic assessment for one case")
    a.add_argument("--no", type=int, required=True)
    a.add_argument("--llm", action="store_true", help="Use Gemini planner if API key set")
    a.set_defaults(func=cmd_assess)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
