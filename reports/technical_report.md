# Technical Report — Rockburst Agentic AI

Numbers below are from the **final-hardening step-6** full run on branch `final-hardening` (tol 5%, cross-fit fusion). Artefact paths under `artifacts/baselines/`.

## Lead finding

### (a) M1 results by protocol

| Protocol | QWK | Accuracy | Production / note |
|----------|-----|----------|-------------------|
| Nested-unbiased (headline) | **0.689** | **0.613** | `rf__median`; `m1_results.json` → `nested_unbiased.pooled_oof` |
| Selection-biased (not headline) | 0.698 | 0.628 | Same file → `selection_biased` |
| Ablation protocol (fixed mode, 5×5 grouped OOF) | 0.682 | 0.589 | `ablation_results.json` M1 |
| Fixed `catboost__knn` @ tol 0.5/1/2/5% (means, 5 repeats) | 0.673 / 0.669 / 0.669 / 0.676 | 0.574 / 0.556 / 0.578 / 0.572 | `item5_dup_label_noise.json`; CIs overlap |

### (b) Label-noise finding (hypothesis)

At primary **5%** near-dup tolerance: **43** conflicting pairs; direction later−earlier **+1: 37**, **−1: 5** (`item5_dup_label_noise.json`). Among pairs touching rows 315–344, mean Δ ≈ **+1.03**, frac +1 ≈ **97%**. **Hypothesis only:** later copies often appear one grade higher; **not** established as deliberate relabelling.

| Policy @ 5% | n | M1 QWK | M1 accuracy |
|-------------|---|--------|-------------|
| Keep all | 344 | 0.691 | 0.610 |
| Keep one (lowest `no`) | 306 | 0.724 | 0.663 |
| Drop conflicting groups | 283 | 0.736 | 0.661 |

Primary tol **5%** chosen after seeing 1% left known near-copies ungrouped (19↔342, 20↔328 @2%; 9↔337 @5%). Earlier headlines used 1%.

### (c) What each agent contributes (including zeros)

| Component | Measured effect |
|-----------|-----------------|
| Physics skip | **990** skips / 1720 evals; skip iff full `check_case` score 1.0 + no issues |
| Extra validation | **681** invocations |
| Critic + rechecks | Fire rate **0.52**; tools geo/skip/extra/recheck_p/d/m/critic = 730/990/681/21/146/717/1779 |
| Substantive M6≠M7 routes | Rate **0.52** (mostly `other_or_mixed` / Critic+recheck bundle; `critic_only` 11) |
| Grade / QWK / abstention M5 vs M6 vs M7 | **Identical** (QWK 0.682, abs **0.053**) |
| Fusion point estimate | Non-zero: `r_physics`, `r_uncertainty`, `n_missing`, `d_observed`; **zero:** `r_data`, `r_model`, `r_stability` |
| Critic flag AUROC | **0.598** (weak); large-set rule ≈ 0.628 |
| Risk–coverage `R_final` vs max-prob | Areas **0.515 vs 0.682**; paired Δ **−0.167** [−0.202, −0.125] — `R_final` does **not** beat max-prob |

### (d) Honest statements

- Fusion does **not** reduce to calibrated model confidence: learned `r_model` weight is **0**; confidence is driven mainly by `r_uncertainty` (+ physics / missingness features). Even so, `R_final` loses to max-prob on risk–coverage.
- Critic discrimination is **weak** (AUROC ≈ 0.60; CI not strongly above chance in spirit of the point estimate).
- **M5 = M6 = M7** on grade metrics and abstention; agents change routes/decisions more than argmax grades.

---

## Dataset
- Source: Table S1, **n = 344**.
- Features: \(\sigma_\theta,\sigma_c,\sigma_t,\mathrm{SCF},B_1,B_2,W_{et},D\); target 0–3.
- \(D\) missing in **131** rows; `dup_group_id` at **5%** → **291** groups.

## Training protocol
- Seed `42`. Nested grouped CV: `n_outer=5`, `n_inner=3`, Optuna trials per config.
- Conformal headline: **`marginal_split`**; nested coverage ≈ **0.939**.
- Fusion: cross-fit stacker + calibrator + thresholds on other folds’ OOF; metrics from held-out scores only (`fusion_protocol` in `m1_results.json`). Mean \(R_{\mathrm{final}}\) **0.641**; ECE **0.104**. Serving `fusion.joblib` fitted on all OOF.

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
| No hand-picked reliability weights without validation | **partly** | Learned stacker; `R_data`/`R_stability` (and `r_model`) zero in point estimate — brief \(R_{\mathrm{final}}=f(\ldots)\) only **partly** met |
| Do not hide when full system ≤ baseline | **met** | Paired risk–coverage Δ\<0; M7 ≰ M1 on QWK |
| Not accuracy alone | **met** | QWK, severe-error, coverage, ECE, risk–coverage |
| Abstain when inadequate confidence | **met** | Gates + fusion bands; M5–M7 abs ≈ 0.053 |

## Limitations
- \(n=344\); label-noise hypothesis at 5% (43 conflict pairs).
- M7 does not improve grades vs M1; routing changes more than predictions.
- `R_final` underperforms max-prob / inv-entropy on risk–coverage.
- Gemini: LLM planner agreement ≈ 0.91 with fallback ≈ 0.24 (`llm_planner_comparison.json`); treat as cache-assisted.

## Artefacts
| Path | Content |
|------|---------|
| `artifacts/baselines/m1_results.json` | Nested/biased M1, cross-fit fusion |
| `artifacts/baselines/ablation_results.json` | M1–M7 + missing/robustness/critic |
| `artifacts/baselines/item4_metrics.json` | Critic AUROC, risk–coverage + paired CI |
| `artifacts/baselines/item5_dup_label_noise.json` | Tol sensitivity + label noise |
| `artifacts/baselines/threshold_sensitivity.json` | Hand-set constant sensitivity |
| `reports/design_decisions_log.md` | Design decisions (before/after seeing results) |
| `reports/audit_log.md` | Historical fix log |
| `artifacts/examples/trace_*.json` | Held-out routing evidence |
