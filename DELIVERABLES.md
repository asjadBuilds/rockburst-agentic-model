# Deliverables map (client brief §7 / §8)

Paths relative to repo root. Regenerate figures/examples from shipped models; do not hand-edit numbers in reports.

## §7 — Core deliverables

| Brief item | Path | Notes |
|------------|------|-------|
| Architecture diagram | `reports/methodology.md` (Mermaid) + **F1** `artifacts/figures/F01_system_architecture.png` | Rendered ≥300 dpi |
| Agent specs | `reports/methodology.md` (§ Full agent specifications) | Schemas in `src/rockburst/agents/` |
| Routing / termination | `reports/methodology.md`; **F2** | `max_loops` → forced abstain |
| Fusion formulation | `reports/methodology.md` (§ equations m4); **F17** | Set A; negative RC result stated |
| Leakage-safe validation | `reports/methodology.md`; `artifacts/baselines/m1_results.json` | Nested grouped CV |
| Missing-data method | `reports/technical_report.md`; `ablation_results.json` → `missing_data`; **F14** | Physical masks + recon |
| Uncertainty / abstention | methodology § conformal; **F12**; threshold artefacts | Mondrian + contiguous sets |
| Working code | `src/rockburst/`, `scripts/`, `Makefile` | Rules FSM offline |
| Notebook | `notebooks/end_to_end_demo.ipynb` | Execute to embed outputs |
| Saved models | `artifacts/models/*.joblib` | Git-ignored; from `train` |
| Baseline / ablation / robustness | `artifacts/baselines/*.json`; **F8–F14** | `headline_snapshot.json` |
| Example outputs | Held-out `artifacts/examples/heldout_*.json` (≥3) | Production cases labelled in-sample |
| Technical report | `reports/technical_report.md` | History appendix for superseded numbers |
| Limitations / failures | technical report; `failure_analysis.md` | C5 |

## §8 — Evidence extras

| Brief item | Path | Notes |
|------------|------|-------|
| Routing traces | `artifacts/examples/trace_*.json`; **F3** | Missingness / OOD / Critic |
| Abstention example | `heldout_abstain_case_77.json`, `forced_abstain_case.json` | Real FSM paths |
| Failure analysis | `failure_case.json`, **`failure_analysis.md`** | Held-out \|ŷ−y\|≥2 |
| Figures F1–F17 | `artifacts/figures/` + `captions.md` | `scripts/generate_figures.py` |
| Feedback reply | `reports/feedback_response.md` | C1–C8, M1–M16, m1–m11 |
| Delivered planner | methodology § Delivered system; STALE LLM JSON | Headline = rules FSM |

## Intentionally excluded from delivery zip

| Item | Reason |
|------|--------|
| `.env` / API keys | Secrets (`.env.example` only) |
| `.cursor/`, `reports/claude_agent_context_pack.md` | Editor / internal context (m7) |
| `catboost_info/` | Training scratch |
| `artifacts/planner_cache/`, `.venv/`, `logs/` | Ephemeral |
| Absolute machine paths in legacy JSON | Regenerate examples |
