# Agentic Rockburst Intensity Assessment

Physics-guided multi-agent system for rockburst grade (0–3) assessment with reliability fusion, conformal uncertainty, and abstention.

**Design rule:** the planner (rules or Gemini) only chooses the next action. All numbers come from deterministic Python tools. Fusion decides trust and abstention.

**Headline (nested-unbiased M1, final-hardening @ 5% near-dup):** QWK **0.689**, accuracy **0.613** (`rf__median`). Fusion reduces to calibrated max-probability (set A); mean \(R_{\mathrm{final}}\) 0.650 is still overconfident vs base accuracy (~0.57–0.59). Rules planner is the default (no API key). See [`reports/technical_report.md`](reports/technical_report.md) and [`reports/design_decisions_log.md`](reports/design_decisions_log.md).

## Quick start for the client

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
make pipeline                                          # extract → train → evaluate (hours; full nested CV)
python scripts/rockburst_cli.py assess --no 1          # one case; rules planner; no API key
```

Outputs: `artifacts/models/`, `artifacts/baselines/`, `artifacts/traces/`, `artifacts/examples/`, `reports/`.
Saved models are git-ignored; if missing, run `make train` (or use the delivery zip models pack).
Optional Gemini (`--llm`): copy `.env.example` → `.env` and set `GOOGLE_API_KEY` (free-tier daily quota is tight; two model IDs may appear in the stale comparison file).

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
cp .env.example .env   # set GOOGLE_API_KEY only if using --llm
```

Pinned environment: [`requirements.txt`](requirements.txt), [`environment.yml`](environment.yml).

## One-command pipeline (no `--quick`)

```bash
make pipeline
# equivalent: extract → train → evaluate
```

## CLI

```bash
python scripts/rockburst_cli.py extract
python scripts/rockburst_cli.py train          # nested CV + conformal + live fusion
python scripts/rockburst_cli.py evaluate       # repeated grouped OOF ablations M1–M7
python scripts/rockburst_cli.py assess --no 1  # rules planner (default)
python scripts/rockburst_cli.py assess --no 146 --llm
python scripts/dup_sensitivity.py
python scripts/refit_fusion_item2.py           # constrained fusion refit + missing-D drop
python scripts/item4_metrics.py                # Critic AUROC / risk-coverage / per-class
python scripts/item5_dup_label_noise.py        # near-dup + label-noise policies
python scripts/generate_examples.py
pytest -q
```

`--quick` exists for smoke tests only. **Do not use `--quick` for any number that appears in reports.**

## Planner modes
| Mode | Flag | Trace tags |
|------|------|------------|
| Rules (default) | _(none)_ | `rules:…` |
| Gemini Flash | `--llm` | `llm:…` / `fallback:…` / `rejected:…` |

Conditional routing: physics may be skipped after a cheap pre-check; Critic rechecks call distinct tools; unresolved objections downgrade the decision one level.

## Repo layout

```
configs/default.yaml
src/rockburst/
scripts/
data/raw|processed/
artifacts/{models,baselines,traces,examples,planner_cache}/
reports/{methodology,technical_report,audit_log,design_decisions_log}.md
tests/
```

## Documentation
- [`DELIVERABLES.md`](DELIVERABLES.md) — client brief §7/§8 → file map
- [`COVER_NOTE.md`](COVER_NOTE.md) — short client cover note
- [`reports/methodology.md`](reports/methodology.md) — architecture, routing, fusion, conformal
- [`reports/technical_report.md`](reports/technical_report.md) — nested headline, agent contributions, §10 table
- [`reports/audit_log.md`](reports/audit_log.md) — before/after by correction item
- [`reports/design_decisions_log.md`](reports/design_decisions_log.md) — locked rules before measurement
