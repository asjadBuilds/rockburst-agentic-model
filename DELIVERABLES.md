# Deliverables map (client brief §7 / §8 + Round-3 remediation)

Paths relative to repo root. Regenerate figures/examples from shipped **full-mode** artefacts; do not hand-edit numbers in reports.

**Canonical maps:** this file + `COVER_NOTE.md` + `reports/feedback_response.md`.
`scripts/package_delivery.sh` writes `README_DELIVERY.txt` into the zip with the **same** start-here order (no contradiction with this file).

## §7 — Core deliverables

| Brief item | Path | Notes |
|------------|------|-------|
| Architecture diagram | `reports/methodology.md` (Mermaid) + **F1** `artifacts/figures/F01_system_architecture.png` | ≥300 dpi |
| Agent specs | `reports/methodology.md` | Schemas in `src/rockburst/agents/` |
| Routing / termination | methodology; **F2** | Critic→recheck cycles + `max_loops` safeguard → forced abstain |
| Fusion formulation | methodology; **F17** | Declared A–D rule selects C; inverted `r_data` excluded; RC negative stated |
| Leakage-safe validation | methodology; `m1_results.json` | Nested grouped CV; OOF fusion |
| Missing-data method | `technical_report.md`; `ablation_results.missing_data`; **F14** | Incl. iterative + physics recon |
| Uncertainty / abstention | methodology; **F12** | Production marginal split; alternatives compared; contiguous sets |
| Working code | `src/rockburst/`, `scripts/`, `Makefile` | Rules FSM offline |
| Notebook | `notebooks/end_to_end_demo.ipynb` | `jupyter nbconvert --execute … --inplace` |
| Saved models | `artifacts/models/*.joblib` | Git-ignored; from `train`; notebook pins sha256 |
| Baseline / ablation / robustness | `artifacts/baselines/*.json`; **F8–F14** | `headline_snapshot.json` (full only) |
| Example outputs | `artifacts/examples/heldout_*.json` (≥3) | Outer-fold held-out assessments with fold/model provenance |
| Technical report | `reports/technical_report.md` | Full-mode; history appendix |
| Limitations / failures | technical report; `failure_analysis.md` | RC7 accepted-wrong |

## §8 — Evidence extras

| Brief item | Path | Notes |
|------------|------|-------|
| Routing traces | `artifacts/examples/trace_*.json`; **F3** | Missingness / OOD / Critic |
| Abstention example | `heldout_abstain_case_*.json`, `forced_abstain_case.json` | Separate from failure |
| Failure analysis | `failure_case.json`, **`failure_analysis.md`** | Accepted-wrong \|ŷ−y\|≥2 |
| Corrupt OOD | `case_corrupt_ood.json` | Real perturbation recorded |
| Figures F1–F17 | `artifacts/figures/` + `captions.md` | `scripts/generate_figures.py` |
| Feedback reply | `reports/feedback_response.md` | **TC1–TC9, TM1–TM10, Tm1–Tm8, F1–F17** |
| Test transcript | `artifacts/test_transcript.txt` | Final full pytest result included in the package |
| Delivered planner | `reports/methodology.md`; `src/rockburst/graph/runner.py` | Headline = deterministic rules FSM |

## Intentionally excluded from delivery zip

| Item | Reason |
|------|--------|
| `.env` / API keys | Secrets (`.env.example` only) |
| `.cursor/`, `reports/claude_agent_context_pack.md` | Editor / internal context |
| `catboost_info/` | Training scratch |
| `artifacts/planner_cache/`, `artifacts/tuning_cache/`, `.venv/`, `logs/` | Ephemeral |
| `artifacts/archive/` (optional) | RC8 stale dumps; keep if auditing history |

## User regeneration

```bash
.venv/bin/python scripts/generate_examples.py
.venv/bin/python scripts/generate_figures.py
jupyter nbconvert --to notebook --execute notebooks/end_to_end_demo.ipynb --inplace
```
