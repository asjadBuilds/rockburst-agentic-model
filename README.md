# Agentic Rockburst Intensity Assessment

Physics-guided multi-agent system for rockburst grade (0–3) assessment with reliability fusion, conformal uncertainty, and abstention.

**Design rule:** the planner (rules or Gemini) only chooses the next action. All numbers come from deterministic Python tools. Fusion decides trust and abstention.

**Headline (nested-unbiased M1):** QWK **0.621**, accuracy **0.567**. Rules planner is the default (no API key). See [`reports/technical_report.md`](reports/technical_report.md) and [`reports/preregistration.md`](reports/preregistration.md).

## Setup

```bash
# One-time
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
reports/{methodology,technical_report,audit_log,preregistration}.md
tests/
```

## Documentation
- [`reports/methodology.md`](reports/methodology.md) — architecture, routing, fusion, conformal
- [`reports/technical_report.md`](reports/technical_report.md) — nested headline, agent contributions, §10 table
- [`reports/audit_log.md`](reports/audit_log.md) — before/after by correction item
- [`reports/preregistration.md`](reports/preregistration.md) — locked rules before measurement
