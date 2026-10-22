# Cover note

Physics-guided multi-agent rockburst intensity assessor with leakage-safe nested grouped validation, Mondrian conformal prediction sets, learned reliability fusion, abstention, and seeded reproducible tooling. **Delivered planner = deterministic rules FSM** (no API key). Gemini remains optional for exploratory routing only.

## Headline numbers (from `artifacts/baselines/headline_snapshot.json`)

| Quantity | Value |
|----------|-------|
| Production model | `rf__iterative` (best mean inner QWK) |
| Nested-unbiased M1 QWK / accuracy | **0.697** / **0.631** |
| Production pooled OOF QWK | **0.682** |
| Ablation M1/M7 QWK | **0.632** [0.579, 0.684] |
| M7 abstention | **0.517** (Critic escalate active) |
| Conformal | **mondrian_split**, coverage ≈ **0.933** |

Honest findings: M5/M6/M7 **identical on grade QWK**; agents change routes and abstention, not argmax. Under fusion set A, R_final does **not** beat max-prob on risk–coverage (negative result retained). Duplicate-label directionality remains a hypothesis — **which labels are authoritative?**

Figures F1–F17: `artifacts/figures/`. Point-by-point feedback reply: `reports/feedback_response.md`.

With more time: (1) confirm authoritative labels, (2) more data, (3) test agents on a second dataset.
