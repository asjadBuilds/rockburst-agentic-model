# Deliverables map (client brief §7 / §8)

Paths checked with `ls` on branch `final-hardening`. Missing items are marked **not provided**.

## §7 — Core deliverables

| Brief item | Path | Notes |
|------------|------|-------|
| Architecture diagram | `reports/methodology.md` (§ Architecture, Mermaid) | Conditional physics skip, extra validation, Critic rechecks → fusion |
| Agent specs | `reports/methodology.md` (§ Agents and tools); `src/rockburst/agents/` | Tools in `agents/tools.py`, schemas in `agents/schemas.py` |
| Routing / termination design | `reports/methodology.md` (§ State transitions); `reports/design_decisions_log.md` | `max_loops=8` → forced abstain; Critic downgrade |
| Fusion formulation | `reports/methodology.md` (§ Reliability fusion); `reports/technical_report.md` | Final = calibrated max-prob (set A); flags only for other `R_*` |
| Leakage-safe validation | `reports/methodology.md` (§ Nested tuning, Leakage guard); `artifacts/baselines/m1_results.json` | Nested grouped CV; StratifiedGroupKFold |
| Missing-data method | `reports/technical_report.md` (§4.2); `artifacts/baselines/ablation_results.json` (`missing_data`); `artifacts/baselines/fusion_missing_d_drop.json` | |
| Uncertainty / abstention method | `reports/methodology.md` (§ Conformal prediction); `artifacts/baselines/threshold_sensitivity.json` | Split conformal; fusion bands + hard gates |
| Working code | repo root (`src/rockburst/`, `scripts/`, `Makefile`) | Offline with rules planner |
| Notebook | `notebooks/end_to_end_demo.ipynb` | |
| Saved models | `artifacts/models/best_baseline.joblib`, `ensemble.joblib`, `fusion.joblib` | Git-ignored; regenerate with `make train` or take from delivery zip |
| Baseline / ablation / robustness results | `artifacts/baselines/m1_results.json`, `ablation_results.json`, `dup_sensitivity.json`, `item5_dup_label_noise.json`, `threshold_sensitivity.json` | |
| Three example outputs | `artifacts/examples/case_1.json`, `case_15.json`, `case_20.json` | Further held-out cases under `artifacts/examples/heldout_case_*.json` |
| Technical report | `reports/technical_report.md` | |
| Limitations / failure cases | `reports/technical_report.md` (§ Limitations); `artifacts/examples/failure_case.json`, `trace_failure.json` | |

## §8 — Evidence extras

| Brief item | Path | Notes |
|------------|------|-------|
| Traces for dynamic routing | `artifacts/traces/case_1.json`, `case_15.json`, `case_146.json`; `artifacts/examples/trace_*.json` | LLM traces `artifacts/traces/case_*_llm.json` optional |
| Abstention example | `artifacts/examples/abstain_case.json`, `forced_abstain_case.json`, `trace_abstain.json` | |
| Failure analysis | `artifacts/examples/failure_case.json`, `trace_failure.json`; narrative in `reports/technical_report.md` / `reports/audit_log.md` | No separate standalone failure-analysis report — **not provided** as its own file; covered by examples + report sections |

## Intentionally excluded from delivery zip

| Item | Reason |
|------|--------|
| `.env` / live API keys | Secrets; use `.env.example` |
| `artifacts/planner_cache/` | May contain cached LLM payloads |
| `.venv/`, `logs/`, `__pycache__/` | Local/ephemeral |
