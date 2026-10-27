# Agentic Rockburst Intensity Assessment

Physics-guided multi-agent system for rockburst grade (0–3) assessment with reliability fusion, conformal uncertainty, and abstention.

**Design rule:** the planner (rules or Gemini) only chooses the next action. All numbers come from deterministic Python tools. Fusion decides trust and abstention.

**Headline (full-mode nested-unbiased M1):** QWK **0.703**, accuracy **0.625** (`lightgbm__median`, best mean inner QWK). Repeated-OOF M1/M7 QWK is **0.670**; M7 abstains on **0.326**. The declared fusion rule selects set **C** and excludes inverted `r_data`; served R_final still trails max-prob on risk–coverage (0.642 vs 0.653). Rules planner is the default. See [`COVER_NOTE.md`](COVER_NOTE.md), [`reports/technical_report.md`](reports/technical_report.md), and [`reports/feedback_response.md`](reports/feedback_response.md).

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
python scripts/triplet_label_sensitivity.py   # RM12/RM13 (long)
python scripts/generate_examples.py           # RC7–RC8 (+ archive stale)
python scripts/generate_figures.py            # F1–F17 + headline_snapshot
jupyter nbconvert --to notebook --execute notebooks/end_to_end_demo.ipynb --inplace
pytest -q
```

`--quick` exists for smoke tests only. **Do not use `--quick` for any number that appears in reports.** Run metadata (mode, date, git hash, package versions) is written into baseline JSONs.

## Planner modes
| Mode | Flag | Trace tags |
|------|------|------------|
| Rules (default) | _(none)_ | `rules:…` |
| Gemini Flash | `--llm` | `llm:…` / `fallback:…` / `rejected:…` |

Conditional routing: physics may be skipped after a cheap pre-check (criteria retained); Critic→recheck cycles limited by `max_critic_recheck_cycles` (forced abstain on non-convergence); `max_loops` is a safeguard only.

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
