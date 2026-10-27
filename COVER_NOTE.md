# Cover note

Physics-guided multi-agent rockburst intensity assessor with leakage-safe nested grouped validation, marginal-split conformal prediction sets, learned reliability fusion, abstention, and seeded reproducible tooling. **Delivered planner = deterministic rules FSM** (no API key). Gemini remains optional for exploratory routing only.

## Headline numbers (from `artifacts/baselines/headline_snapshot.json` after full `train`/`evaluate` + `generate_figures.py`)

| Quantity | Value |
|----------|-------|
| Production model | `lightgbm__median` (best mean inner QWK) |
| Nested-unbiased M1 QWK / accuracy | **~0.703** / **0.625** |
| Production pooled OOF QWK | **~0.697** |
| Ablation M1/M7 QWK (full 5×5, `quick=false`) | **0.670** |
| M7 abstention | **0.326** (1,720 repeated OOF predictions) |
| Conformal | **marginal_split**, coverage ≈ **0.919** |
| RC: R_final vs max-prob area | **0.642 vs 0.653** (R_final does **not** beat; Δ≈−0.012) |
| Fusion feature set | **C**, with inverted `r_data` excluded |

Protected floors retained: nested M1 QWK **0.697**, production pooled **0.682**. Quick ablation **0.632** is invalid and is **not** cited.

Honest findings: M5/M6/M7 **identical on grade QWK**; agents change routes and abstention, not argmax. M7 accepted accuracy is 0.638 but is significantly below M1 thresholded to the same 67.4% coverage (Δ accuracy −0.032, 95% CI [−0.054, −0.010]). After dropping rows 315–344, the missing-D grade association largely disappears.

Figures F1–F17: `artifacts/figures/`. Point-by-point Round-3 reply: `reports/feedback_response.md`.

With more time: (1) confirm authoritative labels, (2) more data, (3) test agents on a second dataset.
