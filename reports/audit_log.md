# Audit log — harden / leakage correction

## Fix 1 — Fusion live agent scores (2026-10-06)

**What was wrong:** `ml/train.py` trained `ReliabilityFusion` on placeholder reliability features (`r_physics=0.8`, `r_stability=0.7` constants) while inference used live agent tools → train/serve mismatch.

**What changed:**
- Added `fusion/features.py::compute_reliability_features` as the single train/serve path.
- Added `quality/fold_ref.py::build_fold_reference` so IQR / near-dup / IsolationForest never see the case or its `dup_group_id`.
- Fusion OOF loop calls live agent tools with fold-fit pipelines; `R_model` from OOF probabilities.
- Stacker coefficients clipped to ≥0 for monotonicity; decision thresholds selected on held-out OOF scores.

**Tests:** `tests/test_leakage.py` — train/serve identity, monotonicity, fold reference exclusion.

## Fix 2 — Split Mondrian conformal

**What was wrong:** APS-style split-free heuristic; `class_thresholds` never fitted.

**What changed:** Mondrian (class-conditional) quantiles of `1 - p_y` on a group-aware calibration holdout inside each outer train fold. Coverage/mean set size reported on outer test folds.

**Full-run result (best model OOF):** coverage ≈ **0.834** vs nominal 0.90; mean set size ≈ **1.94**.

## Fix 3 — Honest OOF ablations

**What was wrong:** in-sample ablation; **M5 ≡ M1**.

**What changed:** repeated/grouped OOF folds; M5 monolithic fixed sequence; M6 coordinator without Critic; M7 full system; CIs + paired Δ.

### Ablation leakage bug found after first corrected evaluate (serious)

**Finding:** first `ablation_results.json` reported M6/M7 QWK = **1.0**. Root cause: `run_graph` called `run_predict` without the fold pipeline, so M6/M7 scored the **full-data production** `best_baseline.joblib` on the same cohort (in-sample).

**Fix:** `graph/runner.py` now honors `cfg["_fold_pipeline"]` / `cfg["_conformal_state"]`; `eval/ablation.py` injects the fold-trained pipe. Ablations re-run after this fix (`ablation_results.json` regenerated).

## Fix 4 — Nested Optuna

Equal `tuning.n_trials=8` per family; inner `cv.n_inner=3`; outer `cv.n_outer=5`. RBF `SVC(probability=True)` removed from the search (sklearn 1.9 deprecation + prohibitive nested-CV cost). Search spaces in `m1_results.json`. No LLM fine-tuning.

## Fix 5 — Leakage pytest suite

`tests/test_leakage.py` — group integrity, imputer train-only, target unused, fusion monotonicity/train-serve, conformal disjointness/coverage, max_loops forced abstain, threshold selection on val only.

---

## Phase 1 — Clean-room

- Artifacts backed up to `~/backups/rockburst-artifacts-20261006-022356`, then cleared.
- Fresh `.venv` via `uv`; pinned `requirements.txt` + `environment.yml`.
- `Makefile` target `pipeline` = extract → train → evaluate (no `--quick`).
- Full train (seed 42): best **`extratrees__iterative`**, OOF QWK **0.706**, accuracy **0.567**.
- Second seed-matched train for reproducibility: see below / run log.

## Phase 3 — Data handling

| Issue | Handling |
|-------|----------|
| B1 inconsistencies (20, 21, 244, + physics scan: 22 cases with issues) | **Flagged** by physics agent / `data_audit.json`; **not corrected or dropped** (retained as Critic tests) |
| OOD SCF 146–157 (12 rows with SCF>2) | **Flagged** (`scf_ood_threshold=2.0`); retained |
| Missing D (131 / 344) | **Retained**; missingness rate by grade ≈ 0:26%, 1:33%, 2:45%, 3:42%; `observed_*` indicators; impute inside train folds only |
| Near-dup groups / conflicting labels | 3 groups with duplicates; **1 conflicting-label group**: nos **6** (grade 2) vs **315** (grade 3), same `dup_group_id=5`. Policy: keep grouped (no cross-fold leak); **do not silent-resolve**; quality flags near-dups |
| PDF fused pairs (38/39, 81/82, 124/125, 210/211, 253/254, 296/297, 339/340) | All pairs `present: true` in `data/raw/parse_audit.json` |

## Phase 5 — Gemini planner

- `gemini-3.8-flash` **is a valid API model ID**.
- Free-tier daily quota (**20 req/day**) exhausted mid-run; switched to **`gemini-3.5-flash`** to finish ≥30 cases (documented in config).
- Temperature 0; Pydantic `PlannerAction`; exponential backoff; disk cache `artifacts/planner_cache/`; traces tag `llm` / `rules` / `fallback`.
- **32 cases** compared (file is **stale** vs final routing; two model IDs from free-tier quota). Rules planner is the default documented path; project runs without an API key. See `artifacts/baselines/llm_planner_comparison.json`.

---

## Before / after headline metrics

| Metric | Old (in-sample / proxy / `--quick` narrative) | New (OOF, live fusion, full run) |
|--------|-----------------------------------------------|----------------------------------|
| M1 best QWK | ~0.75 (quick / overstated) | **0.706** (`extratrees__iterative`) |
| M1 accuracy | not trustworthy | **0.567** OOF |
| Conformal coverage | unjustified APS | **0.834** (nominal 0.90) |
| Fusion features | placeholders 0.8 / 0.7 | live agent `R_*` on OOF |
| Ablation | in-sample; M5=M1 | grouped OOF; M5 distinct; fold pipes for M5–M7 |
| First post-fix ablation M6/M7 QWK | — | **invalid 1.0** (caught) → re-run after fold-pipe fix |

## Open risks / unverified

- Conformal coverage below nominal (0.83 vs 0.90) — expected on small n / model misspecification; not tuned on test.
- Missing-data “complete” accuracy in ablation sidecar can look inflated if it uses the production in-sample pipe; treat train-vs-inference block as the honest missingness check.
- Gemini free-tier limits make large `--llm` runs fragile; cache + model switch used.
- Second full train (same seed 42): **bit-identical** on best_model, best_qwk, accuracy, conformal coverage, fusion ECE / mean R_final (Δ = 0).

---

## Final checklist

| Item | Status | Evidence |
|------|--------|----------|
| Clean-room reproducibility | done | Backup `~/backups/rockburst-artifacts-20261006-022356`; fresh `uv` venv; pinned `requirements.txt` / `environment.yml`; `Makefile` `pipeline`; two full trains seed 42 → identical headline metrics |
| All leakage tests passing | done | `pytest -q` → **17 passed** (`tests/test_leakage.py` + physics/extract/routing) |
| Nested-CV tuning documented | done | `ml/tuning.py`, `m1_results.json` `search_spaces`/`n_trials`, technical report |
| Gemini planner validated and logged | done | `gemini-3.8-flash` valid; quota hit → `gemini-3.5-flash`; 32-case compare in `llm_planner_comparison.json` (agree 90.6%, fallback 24.4%); traces `llm`/`rules`/`fallback` |
| Full ablation with CIs | done | `ablation_results.json` after fold-pipe fix; M7−M1 QWK Δ = 0 (honest negative) |
| Missing-data and robustness | done | Same JSON: confidence ↓ with ε; missing D / multi / train-vs-inference |
| Routing traces for each condition | done | `artifacts/examples/` + `artifacts/traces/` (OOD 146, B1 244, forced abstain, failure OOF case, critic loops) |
| README, reports, pinned env | done | `README.md`, `methodology.md`, `technical_report.md`, `audit_log.md`, pinned deps |
| Limitations and negative results | done | Technical report + this log (M7 ≰ M1 accuracy; coverage < nominal; quota; caught M6/M7 QWK=1.0 bug) |

### Open risks
- Free-tier Gemini quotas; prefer cache + rules for CI.
- Ablation `n_repeats=1` (fold CIs only); increase repeats if compute allows.
- Serving assessments can look better than OOF because production pipe is refit on all data — report OOF numbers for claims.


---

## Pre-delivery round (2026-10-06) — Items 1–3 train pass

### Item 1 — Tolerance near-duplicate grouping

**What was wrong:** `dup_group_id` used all 8 features with greedy `allclose(rtol=1e-3, atol=1e-2)`, finding only 3 multi-row groups. Table S1 near-copies (e.g. 14↔316, 8↔322, 1↔327) were split across folds → OOF/calibration leakage.

**What changed:**
- Raw measured features only (`σθ, σc, σt, Wet, D` when present).
- Relative tolerance `near_dup_rel_tol=0.01` (1%); connected components (union-find).
- Quality near-dup flag uses the same rule; fold reference still drops whole group.
- Conflicting labels kept (not resolved).

**Sensitivity (`artifacts/baselines/dup_sensitivity.json`):**

| rel_tol | n_groups | multi | conflict groups | conflict pairs |
|--------:|---------:|------:|----------------:|---------------:|
| 0.5% | 325 | 19 | 15 | 15 |
| **1% (default)** | **319** | **24** | **18** | **19** |
| 2% | 314 | 26 | 21 | 25 |

Size dist @1%: `{1:295, 2:23, 3:1}`.

**Evidence:** known pairs share groups; `pytest` extract+leakage green; `data/processed` regenerated (`dup_groups=319`).

**Before → after (train headlines, after Item 1+2+3+6 in same pass):**

| Metric | Before (exact-ish groups) | After (1% CC groups + fixes) |
|--------|---------------------------|------------------------------|
| M1 selection-biased pooled QWK | 0.706 | 0.685 (`rf__iterative`) |
| M1 nested-unbiased pooled QWK | (not reported) | **0.621** (`catboost__knn` mode) |
| M1 accuracy (biased / nested) | 0.567 | 0.616 / 0.567 |
| Conformal coverage | 0.834 | **0.898** (marginal_split) |
| Conformal mean set size | 1.94 | 2.11 |

Drops in nested QWK are expected (near-dup leakage removed + honest selection). Not tuned to recover old numbers.

### Item 2 — Protocol reconciliation + selection bias

**What was wrong:** train reported pooled OOF QWK 0.706 while ablation mean-of-folds showed ~0.740; model×imputer chosen on the same outer OOF used as the headline.

**What changed:** every candidate emits `pooled_oof` and `mean_fold`; `selection_biased` vs `nested_unbiased` both reported; production `best_model` = mode of nested fold winners; full `candidate_table` in `m1_results.json`.

### Item 3 — Conformal under-coverage

**What was wrong:** `np.quantile(..., 1-α)` without finite-sample correction; Mondrian sparse-class fallback; coverage 0.834 vs 0.90.

**What changed:** quantile `⌈(n+1)(1-α)⌉/n`; compared Mondrian / marginal / CV+; kept **marginal_split** (coverage 0.898, closest to 0.90). Class calib counts example `[8,16,20,11]`.

### Item 6 — SVM restored

SVM via `CalibratedClassifierCV` (linear/RBF, reduced space, equal `n_trials`). Pooled OOF QWK ≈ 0.35–0.37 (weak but present in candidate table).



### Item 4 — Ablation risk control

**What was wrong:** QWK identical M1–M7; M5/M6/M7 abstention identical; risk-control claim not demonstrated.

**What changed / evidence (`ablation_results.json`, n_repeats=5):**
- Accepted vs abstained on M7: accepted acc 0.600 / QWK 0.685 / severe 0.060 vs abstained 0.359 / 0.316 / 0.141.
- Critic fire rate 42.4%; recheck 87/1720; **decision_changed=0**.
- M6 vs M5 route divergence **0%**.
- Finding: M5≡M6≡M7 on QWK/abstention; Critic does not change finals (thresholds not tuned on test).

### Item 5 — Missing / robustness fold-only

Rewrote experiments to use held-out fold models only; median/KNN/iterative + indicators; train-vs-inference missing-D. Production in-sample “complete” accuracy removed/relabeled.

### Item 6 — SVM

Restored under equal Optuna budget via CalibratedClassifierCV; pooled OOF QWK ~0.35–0.37 (weak but present).

### Item 7 — Statistical robustness

`n_repeats=5`; grouped bootstrap CIs over `dup_group_id` for ablation metrics.

### Item 8 — Examples / Gemini / secrets

- Examples: production files labeled; `heldout_case_{5,15,50}.json` from fold models.
- Gemini compare (cache): agreement 90.6%, fallback 24.4%, **no final-decision differences**; rules remain default.
- Secrets: `.env` gitignored and never tracked; no `AIza…` / live key values in `artifacts/planner_cache/`, `artifacts/traces/`, `reports/`, or git history text. `.env.example` has empty `GOOGLE_API_KEY=`. **No rotation required from repo evidence.**

### Item 9 — Brief §10

See `reports/technical_report.md` checklist table (all seven constraints **met** with evidence paths).

## Final before/after headline table

| Metric | Old (pre this round) | Corrected |
|--------|----------------------|-----------|
| M1 QWK (selection-biased pooled) | 0.706 | 0.685 |
| M1 QWK (nested-unbiased) | — | **0.621** |
| M1 accuracy (biased / nested) | 0.567 | 0.616 / 0.567 |
| Conformal coverage | 0.834 | **0.898** (marginal) |
| Conformal mean set size | 1.94 | 2.11 |
| ECE of \(R_{final}\) | ~0.200 | **0.141** |
| M4 abstention | 0.282 | **0.392** |
| M5/M6/M7 abstention | 0.044 | **0.053** |

## Clean-room verification (this round)

- Fresh `uv venv` at `/tmp/rockburst-cleanroom-venv`; `pip install -r requirements.txt` + editable install.
- `extract` → `train` → `evaluate` (no `--quick`).
- Clean-room train vs prior seed-42 train: **bit-identical** on `best_model`, `best_qwk`, selection-biased/nested QWK, fusion ECE / mean \(R_{\mathrm{final}}\), conformal coverage.
- Two consecutive seed-42 trains in project `.venv`: also bit-identical.
- `pytest -q`: **21 passed**.

## Open risks / could not verify
- Critic path never changes decisions under current thresholds (dataset/threshold structure).
- Gemini live quota may still fail without cache; comparison reused disk cache + residual fallbacks.
- `python3 -m venv` failed on this host (`ensurepip` missing); clean-room used `uv venv` as documented in README.

---

## Follow-up Item 1 — Conditional routing + real rechecks (2026-10-06)

**Pre-registered in** `reports/design_decisions_log.md` (physics skip, `R_data` cut 0.55, Critic downgrade, recheck tools) **before** this measure.

### What was wrong
- Geomechanics always ran (`missing or suspicious or True`).
- Critic rechecks re-invoked the same deterministic tools → no-ops; `decision_changed = 0`.
- Critic never affected `recommended_action`.

### What changed
- Cheap pre-check (`physics/precheck.py`); skip geomechanics when clean.
- Extra validation branch (alt-imputer + targeted perturbation) when D missing / imputed / `R_data < 0.55`.
- Rechecks: `recheck_physics` (tol band), `recheck_data` (alt-imputer + targeted pert), `recheck_model` (ordinal cross-check).
- Unresolved Critic → one-step decision downgrade (grade unchanged).
- LLM precondition guards → `rejected:` + rules fallback.
- Tests in `tests/test_routing.py` (10 routing/physics tests in suite).

### Evidence (full ablations, no `--quick`, seed 42, `n_outer=5`, `n_repeats=5`, n=1720)

| Metric | Before (prior audit) | After Item 1 |
|--------|----------------------|--------------|
| M6 route ≠ M5 | 0% | **61.0%** (1050/1720) |
| M7 route ≠ M5 | 0% | **100%** (1720/1720) |
| `geomechanics_skipped` | 0 | **1050** |
| `extra_validation` | 0 | **684** |
| `recheck_physics` / `data` / `model` | 0 | **25 / 181 / 630** |
| Critic fire rate | 42.4% | **49.0%** (843/1720) |
| `decision_changed` (M7 vs M6) | **0** | **450** (26.2%) |
| Recheck changed confidence / set | — | **0 / 0** |
| Recheck changed decision | — | **450** |
| M5/M6/M7 QWK | 0.667 identical | **0.667 identical** |
| M5/M6/M7 abstention | 0.053 identical | **0.053 identical** |
| Paired Δ QWK/severe (M7−M6, M6−M5, M7−M1) | 0 | **0** (CI [0,0]) |

**Decision-changed subset (n=450):** grade correct on **250** (55.6%); severe errors (`|Δ|≥2`) on **36** (8.0%). Grade is unchanged by the Critic; changes are trust-label downgrades only. Confidence and conformal sets did not move between M6 and M7 on this run (downgrade path does not recompute `R_final`).

## Follow-up Item 2 — Refit fusion (2026-10-06)

**Pre-registered in** `reports/design_decisions_log.md` before measure. Refit from saved OOF live `R` via `scripts/refit_fusion_item2.py` (constrained fit + isotonic + missing-D holdout).

### What was wrong
- `class_weight="balanced"` + post-hoc `coef ≥ 0` clip → only `r_model`, `r_uncertainty` nonzero (`[0,0,2.245,1.816,0]`).
- Heuristic fallback weights `[0.2,…,0.15]` when unfitted.
- No `R_final` recalibration; mean \(R_{\mathrm{final}}\) ≈ 0.713 vs base acc ≈ 0.567; ECE ≈ 0.141.

### What changed
- Bound-constrained non-negative logistic (L-BFGS-B); no class_weight; no clip-after-fit.
- Collinearity rule: `|corr(r_model,r_uncertainty)|=0.953`, VIF>5 → dropped **`r_uncertainty`** (lower univariate AUC); keep `r_model`.
- Isotonic calibration on inner holdout.
- Unfitted → abstain (heuristic removed).
- Zero-weight flags retained in report: `r_data`, `r_physics`, `r_stability` (do **not** enter `R_final` in the point estimate).

### Before / after (OOF live R, seed 42)

| Metric | Before | After |
|--------|--------|-------|
| Coefficients | `[0, 0, 2.245, 1.816, 0]` | `[0, 0, 4.433, 0*, 0]` (`*` dropped for collinearity) |
| Mean \(R_{\mathrm{final}}\) | 0.713 | **0.572** |
| Base accuracy | 0.567 | 0.567 |
| ECE of \(R_{\mathrm{final}}\) | 0.141 | **0.134** (set A cross-fit; supersedes earlier understated ECE) |
| Brier | — | 0.228 (set A) |
| Missing-D mean drop (held-out folds, n=213 obs-D) | — | **0.071** (76.5% cases drop) |

Bootstrap coef means (inner): `r_data`/`r_physics`/`r_stability` sometimes >0 across resamples, but point estimate keeps them at 0 — reported as zero-weight flags. Missing-D drop occurs mainly via `R_model` missingness discount (0.07 per missing feature), not via `R_data` weight.

**Limitation:** `R_data` / `R_physics` / `R_stability` still do not enter the point-estimate `R_final`. Documented, not force-weighted after seeing test results.

### Plain finding (Item 1)
M5 ≡ M6 ≡ M7 on accuracy / QWK / abstention / ECE still holds. Routes and recommended actions now diverge; agents still do not change the argmax grade or hard-abstain rate.

## Follow-up Item 3 — Nested thresholds + untuned constants (2026-10-06)

**Pre-registered** decision bands + a-priori `marginal_split` + domain hand-set constants in `reports/design_decisions_log.md` before measure.

### What was wrong
- `decide()` mapped both mid bands to `engineering_verification` and only returned `accept` at R ≥ ~0.90 (`accept_r_final` dead).
- Conformal method chosen by closeness to 0.90 on evaluation folds.
- Decision thresholds selected once globally / on train subsample, not clearly nested.
- Hand-set gates/penalties not listed with a priori justification + sensitivity.

### What changed
- Decision bands: abstain → engineering_verification → accept_with_caution → accept (`stacker.decide` + tests).
- Thresholds selected on **inner calib** of each outer fold in ablation.
- Conformal headline locked to **marginal_split** a priori; Mondrian/CV+ side-by-side retained.
- Sensitivity table in `artifacts/baselines/threshold_sensitivity.json` (do not pick best).

### Evidence
| Item | Value |
|------|-------|
| Decision mapping | abstain < eng < caution ≤ accept |
| Conformal (marginal / Mondrian / CV+) coverage | 0.898 / 0.927 / 0.904 |
| Nested-unbiased M1 QWK / acc (headline) | **0.621 / 0.567** |
| Ablation M1 QWK / acc | 0.667 / 0.587 |
| Ablation M1 QWK CI | [0.620, 0.712] |
| Gap explanation | Ablation scores one production mode on repeated OOF; nested pools per-fold inner winners — nested stays headline |

Sensitivity (single outer fold, fuse-only probe under new fusion): baseline abstention high (~0.62); listed ±deltas did not move QWK/abstention on that fold — constants remain domain-justified, not retuned.

## Follow-up Item 4 — Missing metrics (2026-10-06)

**Evidence:** `artifacts/baselines/item4_metrics.json` (5-fold grouped OOF, seed 42, no `--quick`).

| Metric | Value |
|--------|-------|
| Critic flag AUROC (error detection) | **0.592** |
| Error rate flagged / not flagged | 0.543 / 0.350 |
| Flag precision / recall | 0.543 / 0.476 |
| Large-set rule AUROC | 0.626 (best individual rule) |
| Physics / disagreement rule fire rate | 0 (on this first-pass probe) |
| Risk–coverage area (final protocol sets) | A **0.671**, C/D **0.687**, B **0.667** |
| Risk–coverage max-prob / inv-entropy (final-run probe) | **0.682** / **0.685** |
| Severe-error overall (M1) | 0.073 |
| Severe-error per class 0/1/2/3 | 0.060 / 0.092 / 0.041 / 0.110 |
| Conformal coverage per class 0/1/2/3 | 0.900 / 0.878 / 0.902 / 0.932 |
| M7 abstention (this pass, new fusion) | 0.122 |

Confusion matrices (counts + row-norm) for M1 and M7 stored in the JSON. Missing-data / robustness tables remain in `ablation_results.json` (held-out fold models); Item 2 missing-D drop mean **0.071**.

## Follow-up Item 5 — Near-dup sensitivity + label noise (2026-10-06)

**Pre-registered** NN-gap tolerance choice (keep 1%) before using headline movement for selection.

### Headlines by `near_dup_rel_tol`

| Tol | Groups | Conflict groups | M1 QWK | M1 acc | Conformal cov |
|-----|--------|-----------------|--------|--------|---------------|
| 0.5% | (see JSON) | 15 | 0.720 | 0.622 | 0.922 |
| **1%** | 319 | 18 | **0.654** | **0.578** | **0.901** |
| 2% | 314 | 21 | 0.691 | 0.596 | 0.866 |
| 5% | 291 | 23 | 0.675 | 0.602 | 0.922 |

M7 abstention at default 1%: **0.053** (full ablation) / **0.122** (Item 4 pass with refit fusion).

### Pair checks
- 19↔342: max rel ≈ 1.51% → first grouped at **2%**
- 20↔328: max rel ≈ 1.47% → first grouped at **2%**
- 9↔337: max rel ≈ 4.41% → first grouped at **5%**

### Label-noise direction (19 conflict pairs @ 1%)
- Δ (later − earlier): **+1: 18**, −1: 0, |Δ|>1: 1; mean Δ ≈ **1.05**
- Pairs touching rows 315–344: n=18, mean Δ ≈ 1.06, **94%** are +1 (state without over-interpreting)

### Policies (M1 OOF @ 1% groups)
| Policy | n | QWK | Accuracy |
|--------|---|-----|----------|
| Keep all | 344 | 0.673 | 0.581 |
| Keep one (lowest `no` per conflict group) | 325 | 0.698 | 0.655 |
| Drop conflicting groups | 307 | 0.684 | 0.622 |

Label conflicts imply a ceiling below perfect accuracy; ≥19 disagreeing pairs. Details: `artifacts/baselines/item5_dup_label_noise.json`.

## Follow-up Item 6 — Final rewrite + re-verification (2026-10-06)

### Docs
- Updated `README.md`, `methodology.md`, `technical_report.md` (label-noise lead; nested headline; measured agent contributions including zeros; §10 table with met/partly; limitations).
- `reports/design_decisions_log.md` holds locked rules for Items 1–5.

### Verification
| Check | Result |
|-------|--------|
| `pytest -q` (project `.venv`) | **28 passed** |
| Clean-room `uv venv` + install + `pytest -q` | **28 passed** |
| Fusion refit twice from same OOF `R` | **bit-identical** coef + mean \(R_{\mathrm{final}}\) |
| API key pattern `AIza…` in `reports/`, `src/`, `scripts/`, `configs/`, `artifacts/` | **no matches** |
| Local `.env` | present with `GOOGLE_API_KEY` (not committed; expected) |

### Unresolved / unverified
- Full clean-room `make pipeline` ×2 (extract→train→evaluate, no `--quick`) not re-run end-to-end in this round (nested Optuna + 5×5 ablation ≈ hours). Prior round documented bit-identical seed-42 trains; fusion stacker now verified identical on saved OOF `R`.
- Git history key scan not re-executed here (local `.env` remains the only live key location found).
- Ablation after nested-threshold code change not fully re-run; Item 1 ablation numbers are pre–Item-3 threshold nesting.
- Fusion still zero-weights `R_data`/`R_physics`/`R_stability` in the point estimate.

---

## Follow-up round — before/after summary table

| Metric | Before this round | After |
|--------|-------------------|-------|
| M5/M6/M7 route divergence (M6≠M5) | 0% | **61.0%** |
| M7 route ≠ M5 | 0% | **100%** |
| `decision_changed` | 0 | **450 / 1720** |
| M7 abstention (full ablation) | 0.053 | 0.053 (ablation); **0.122** (Item 4 + refit fusion) |
| ECE of \(R_{\mathrm{final}}\) | 0.141 | **0.134** (set A; final protocol) |
| Mean \(R_{\mathrm{final}}\) vs base acc | 0.713 vs 0.567 | **0.650** vs ~0.57–0.59 (still overconfident ~7–8 pts) |
| Fusion | multi-channel | **reduces to calibrated max-prob (set A)**; C/D nominally better but not significant |
| Conformal coverage per class | partial | see `m1_results.json` / item4 |
| Risk–coverage (final protocol) | — | A 0.671, C/D 0.687, B 0.667; max-prob 0.682; entropy 0.685 |
| Critic flag AUROC | — | **0.598 [0.564, 0.631]**; fires on ~58% (caution label) |

Supersession note: earlier Item 4 fold path gave R_final 0.515; superseded by the cross-fit protocol; cause not investigated.
