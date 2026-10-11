# Technical Report — Rockburst Agentic AI

Numbers below are from the **final-hardening** run on branch `final-hardening` (tol 5%, cross-fit fusion, locked feature set **A**). Artefact paths under `artifacts/baselines/`.

## Lead finding

### (a) Label-noise finding (hypothesis)

**Hypothesis only:** later duplicated copies appear graded one level higher (18 of 19 conflicting pairs +1 at the 1% conflict-pair analysis in `audit_log` / Item 5); **not** established as deliberate relabelling. At primary **5%** tol the same direction appears (37 of 43 pairs +1 in `item5_dup_label_noise.json`). Please confirm which label source is authoritative.

| Policy @ 5% | n | M1 QWK | M1 accuracy |
|-------------|---|--------|-------------|
| Keep all | 344 | 0.691 | 0.610 |
| Keep one (lowest `no`) | 306 | 0.724 | 0.663 |
| Drop conflicting groups | 283 | 0.736 | 0.661 |

Primary tol **5%** chosen after seeing 1% left known near-copies ungrouped (logged in `design_decisions_log.md`).

### (b) M1 results by protocol

Headline is **nested-unbiased** M1. Other protocols are listed for transparency; they are not interchangeable.

| Protocol | QWK | Accuracy | Production / note |
|----------|-----|----------|-------------------|
| Nested-unbiased (headline) | **0.689** | **0.613** | `rf__median`; `m1_results.json` → `nested_unbiased.pooled_oof` |
| Selection-biased | 0.698 | 0.628 | Same file → `selection_biased` |
| Ablation protocol (fixed mode, 5×5 grouped OOF) | 0.682 | 0.589 | `ablation_results.json` M1 |
| Tolerance runs `catboost__knn` @ 0.5/1/2/5% | 0.673 / 0.669 / 0.669 / 0.676 | 0.574 / 0.556 / 0.578 / 0.572 | `item5_dup_label_noise.json` |
| Near-dup QWK CI @ 5% | **0.671 [0.618, 0.717]** | — | group-resample CI in item5 |

### (c) What each agent contributes (including zeros)

| Component | Measured effect |
|-----------|-----------------|
| Physics skip | **990** skips / 1720 evals; skip iff full `check_case` score 1.0 + no issues |
| Extra validation | **681** invocations |
| Critic | Flag fires on ~**58%** of cases (`item4_metrics.json` flag_rate); acts as a caution label. AUROC **0.598 [0.564, 0.631]** (weak but above chance) |
| Routing | The only identified exclusive routing effect is the Critic. Routing does **not** change grade metrics or abstention (**M5 = M6 = M7**: QWK 0.682, abs **0.053**) |
| Fusion | Reduces to calibrated max-probability (feature set **A**). Sets C/D gave nominally better calibration (ECE 0.104 vs 0.134, Brier 0.207 vs 0.228) and area (0.687 vs 0.671) but not significantly (paired CI of A vs best [−0.033, +0.002]), so the simpler set was chosen by a rule fixed in advance. `r_data`, `r_physics`, and `r_stability` are reported flags and carry no significant extra signal on this dataset |
| Risk–coverage (final protocol) | Set A **0.671**; C/D **0.687**; B **0.667** (`fusion_feature_set_compare.txt`). Max-prob / inv-entropy areas from the item4 final-run probe: **0.682** / **0.685** |
| Abstention | M5=M6=M7 abstention **0.053**. Attribution to fusion thresholds vs hard gates: **not measured**. Hard-gate constants are hand-set; sensitivity in `threshold_sensitivity.json` |

### (d) Calibration and honest statements

- Mean \(R_{\mathrm{final}}\) **0.650** exceeds base accuracy (~0.57–0.59), so the system is still overconfident by about 7–8 points. Do **not** describe \(R_{\mathrm{final}}\) as matching base accuracy.
- Critic discrimination is weak but above chance: AUROC **0.598 [0.564, 0.631]**; caution label on ~58% of cases.
- **M5 = M6 = M7** on grade metrics and abstention; agents change routes, checks, and caution labels, not argmax grades.

---

## Dataset
- Source: Table S1, **n = 344**.
- Features: \(\sigma_\theta,\sigma_c,\sigma_t,\mathrm{SCF},B_1,B_2,W_{et},D\); target 0–3.
- \(D\) missing in **131** rows; `dup_group_id` at **5%** → **291** groups.

## Training protocol
- Seed `42`. Nested grouped CV: `n_outer=5`, `n_inner=3`, Optuna trials per config.
- Conformal headline: **`marginal_split`**; nested coverage ≈ **0.939**, mean set size ≈ **2.48**.
- Fusion: locked set **A** (`r_model` only); cross-fit stacker + calibrator + thresholds on other folds’ OOF; metrics from held-out scores (`fusion_protocol` in `m1_results.json`). Mean \(R_{\mathrm{final}}\) **0.650**; ECE **0.134**; Brier **0.228**. Serving `fusion.joblib` fitted on all OOF with the same mask.

## Ablation table (grouped OOF, 5×5)

| Model | QWK | Accuracy | Abstention |
|-------|-----|----------|------------|
| M1–M3 | 0.682 | 0.589 | 0 |
| M4 | 0.682 | 0.589 | **0.440** |
| M5–M7 | 0.682 | 0.589 | **0.053** |

## §4.2 Missing-data table (held-out fold models)

From `ablation_results.json` → `missing_data` (accuracy on n=40 probes unless noted):

| Imputer | complete | miss \(\sigma_\theta\) | miss \(\sigma_c\) | miss \(\sigma_t\) | miss Wet | miss D | multi | infer-only miss D | train+infer miss D |
|---------|----------|------------------------|-------------------|-------------------|----------|--------|-------|-------------------|--------------------|
| median | 0.40 | 0.375 | 0.425 | 0.40 | 0.30 | 0.35 | 0.225 | 0.362 | 0.391 |
| knn | 0.40 | 0.425 | 0.425 | 0.40 | 0.375 | 0.45 | 0.35 | 0.435 | 0.391 |
| iterative | 0.40 | 0.425 | 0.425 | 0.425 | 0.325 | 0.35 | 0.375 | 0.362 | (see JSON) |

## §4.3 Multiclass vs ordinal (pooled OOF by family)

Additional justified tabular method: **ExtraTrees** (alongside RF / boosting). Ordinal logistic is competitive but below RF headline.

| Model | QWK | Accuracy |
|-------|-----|----------|
| `rf__median` (production) | 0.693 | 0.631 |
| `extratrees__median` | 0.642 | 0.576 |
| `catboost__median` | 0.657 | 0.555 |
| `ordinal_logistic__median` | 0.623 | 0.535 |
| `ordinal_logistic__knn` | 0.620 | 0.529 |
| `logistic__median` | 0.589 | 0.491 |

## Brief §10 constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| No prompt-only agents without tools | **met** | `agents/tools.py`, physics, ML, fusion |
| Do not claim more agents ⇒ better performance | **met** | M5≡M6≡M7 on grades; Critic AUROC weak |
| No test data in preprocessing / selection | **met** | Nested inner selection; fold impute/scale; fusion cross-fit |
| Fusion formula / reliability weights | **partly** | Locked set A by pre-registered held-out rule; fusion reduces to calibrated max-prob; `r_data`/`r_physics`/`r_stability` are reported flags only |
| Do not hide when full system ≤ baseline | **met** | M7 ≰ M1 on QWK; agents do not improve grades |
| Not accuracy alone | **met** | QWK (+ CI), severe-error, coverage, ECE, risk–coverage, Critic AUROC CI |
| Abstain when inadequate confidence | **met** | Gates + fusion bands; M5–M7 abs ≈ 0.053 |

## LLM planner
`llm_planner_comparison.json` is from an earlier routing version (**stale**; two model IDs because of free-tier quota: `gemini-3.8-flash` → `gemini-3.5-flash`). The **rules planner** is the default and documented path; the project runs without an API key.

## Limitations
- \(n=344\); conflicting duplicate labels (see lead finding — which source is authoritative?).
- Grouping tolerance and conformal method chosen after seeing results (logged in `design_decisions_log.md`).
- Fusion feature set chosen on held-out evidence (set A).
- No CIs for some per-rule Critic metrics if absent from `item4_metrics.json`.
- Clean-room reproducibility checked on the final pipeline once (second run diffed, no differences).
- System still overconfident: mean \(R_{\mathrm{final}}\) 0.650 vs base accuracy ~0.57–0.59.
- Critic AUROC **0.598 [0.564, 0.631]** — weak error detection; caution label only.
- M7 does not improve grades vs M1.

## Artefacts
| Path | Content |
|------|---------|
| `artifacts/baselines/m1_results.json` | Nested/biased M1, cross-fit fusion |
| `artifacts/baselines/ablation_results.json` | M1–M7 + missing/robustness/critic |
| `artifacts/baselines/fusion_feature_set_compare.txt` | Final-protocol risk–coverage by feature set |
| `artifacts/baselines/item4_metrics.json` | Critic AUROC; max-prob / entropy areas |
| `artifacts/baselines/item5_dup_label_noise.json` | Tol sensitivity + label noise |
| `artifacts/baselines/threshold_sensitivity.json` | Hand-set constant sensitivity |
| `reports/design_decisions_log.md` | Design decisions (before/after seeing results) |
| `reports/audit_log.md` | Historical fix log |
| `artifacts/examples/` | Example case outputs and abstain/failure cases |
