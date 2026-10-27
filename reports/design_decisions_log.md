# Design decisions log

Renamed from `preregistration.md` (final-hardening step 7). Entries are marked **before** or **after** seeing results. Do not read “locked a priori” unless the entry says so and git history supports it.

**Slice 8 (after remediation):** production conformal = Mondrian + contiguous sets (overrides earlier a-priori marginal note); production model = best mean inner QWK (not mode-of-winners); Critic may escalate to abstain (overrides earlier “floor at engineering_verification”); delivered planner = rules FSM. See `reports/feedback_response.md` and `reports/technical_report.md` history appendix for superseded numbers.

Changed after seeing results (both versions):

| Rule | Before | After (final-hardening) |
|------|--------|-------------------------|
| Physics skip | Cheap SCF/B1/B2 + Wet/D in range (duplicated logic) | Skip iff `check_case` returns score 1.0 and no issues |
| Near-dup primary tol | **0.01 (1%)** | **0.05 (5%)** |
| `r_model` | `max(p)*(1-0.07*n_miss)` | raw `max(p)`; missingness learned in stacker |
| Fusion metrics | Pooled OOF fit+score | Cross-fit held-out scores only |
| M6 | `allow_critic=False` also blocked `extra_validation` | M6 = coordinator + extra_validation; Critic off |

---

## Item 1 — Routing and Critic (written 2026-10-06, before Item 1 measure)

### 1a. Physics skip

**Original (before final-hardening):** cheap pre-check on SCF/B1/B2 + Wet/D ranges.

**After seeing skip≠full-tool mismatch (final-hardening step 1):** skip geomechanics only when full `check_case` returns `r_physics == 1.0` and `issues == []` (no duplicated penalty logic).

### 1a. Extra validation branch

Run an additional validation step (Item 1b tools: alt-imputer + targeted perturbation) when **any** of:

- D is missing
- any key feature was imputed / listed in `missing_features`
- `data_reliability` \< **0.55**

Cut **0.55** = below the “moderate” stability band used elsewhere in the codebase (stability labels use 0.5 / 0.75). Domain: low input reliability warrants extra checks before trust.

### 1a. Critic review triggers

Existing: model disagreement ≥ `routing.disagreement_threshold` (0.35), `|prediction_set| ≥ large_set_size` (3), OOD, physics score \< 0.6.

Plus: unresolved objections from the extra validation branch (large alt-imputer grade flip or large probability spread, or low targeted stability).

### 1b. Recheck tools (must differ from first pass)

| `request_recheck` | Recheck tool (not the first-pass agent) |
|-------------------|----------------------------------------|
| `physics` | Physics with **perturbed tolerance band** (report borderline relations) |
| `data` | **Alt-imputer re-prediction** + **targeted perturbation** on imputed/flagged features |
| `model` | **Independent cross-check model** (ordinal vs primary tree family) |

### 1c. Unresolved Critic objection → decision downgrade

After recheck, if Critic still `endorse=False`:

- Downgrade one level: `accept` → `accept_with_caution` → `engineering_verification` (floor).
- Grade stays as predicted.
- Reason recorded on the assessment output.

Not tuned on test results.

### Decision bands (locked for later Item 3; documented now)

Confidence ascending:

| Condition | Decision |
|-----------|----------|
| `R < abstain_t` | `abstain` |
| `abstain_t ≤ R < eng_t` | `engineering_verification` |
| `eng_t ≤ R < caution_t` | `accept_with_caution` |
| `R ≥ accept_t` | `accept` |

Require `abstain_t < eng_t < caution_t ≤ accept_t`. Thresholds selected on inner data only (Item 3).

---

## Item 2 — Fusion refit (written before Item 2 measure; amended in final-hardening)

1. Fit binary logistic for “base prediction correct” on OOF live reliabilities.
2. **No** `class_weight="balanced"`.
3. Reliability channels: non-negative L-BFGS-B; **after results:** `n_missing` / `d_observed` free-signed (missingness may reduce trust).
4. Collinearity rule for `r_model` vs `r_uncertainty` unchanged; zero-weight features reported.
5. Isotonic calibration on inner / non-test data only.
6. Unfitted stacker: abstain; **after seeing 100% abstain bug:** single-class OOF correctness uses constant-rate fallback so fold fusion remains scorable.
7. Missingness: hand 0.07 discount **removed** after diagnosing train/serve gap; learned via stacker features instead.

## Item 3 — Thresholds / conformal (written before Item 3 measure)

- Decision bands as in Item 1 table; select four cutoffs on non-test data.
- Headline conformal: **`marginal_split`** (simplest finite-sample guarantee — design choice, not “a priori” in the strong sense after iteration). Mondrian and CV+ side-by-side; not chosen by closeness to 0.90.
## Item 5 — Near-dup tolerance

- **Before results (original):** Choose from NN gap; locked at **0.01 (1%)**.
- **After seeing results (final-hardening):** Primary **0.05 (5%)** because 1% left known near-copies ungrouped (19–342, 20–328 @2%; 9–337 @5%). Earlier headlines used 1%. Sensitivity at 0.5/1/2/5% with fixed `catboost__knn`, ≥5 repeats.

## Item 6 — Hand-set decision-path constants (assumption + sensitivity; after seeing routing design)

Written **after** seeing routing/Critic design, **before** final-hardening full run. Not learned on test folds. Sensitivity via `scripts/threshold_sensitivity.py` (± fixed deltas; report all; do not pick best).

| Constant | Value | Status |
|----------|-------|--------|
| Hard abstain: \|set\|≥3 & stability < 0.55 | 0.55 | assumption |
| Hard abstain: disagreement ≥ 0.5 & R_final < 0.75 | 0.5 / 0.75 | assumption |
| Hard abstain: OOD & physics < 0.7 & r_model < 0.6 | 0.7 / 0.6 | assumption |
| Physics penalties wet/D/SCF-OOD | 0.1 / 0.05 / 0.15 | assumption |
| Routing disagreement / physics / set size | 0.35 / 0.6 / 3 | assumption |
| Extra validation R_data cut | 0.55 | assumption |

## Item 7 — Fusion cross-fit (final-hardening step 4)

Written **before** final full run: ECE / Brier / mean R_final / risk-coverage computed only from held-out fold scores after fitting stacker+calibrator+thresholds on other folds' OOF. Production `fusion.joblib` still fitted on all OOF for serving.

## Item 8 — Verification (final-hardening step 8, after seeing results)

- Second clean checkout (detached worktree at same commit): `make pipeline` + `item4_metrics`; `scripts/diff_headline_metrics.py` → **no differences** in headline keys.
- `pytest -q`: 32 passed.
- Secrets: `git log --all -S"AIza"` hits historical commit `fe8265f` only. Working-tree grep under `artifacts/`, `reports/`, `logs/` finds only documentation mentions in `audit_log.md` (no live key values). `.env` gitignored.

## Item 9 — Fusion feature-set selection (BEFORE comparison)

Written **before** running the A/B/C/D cross-fit comparison on saved OOF vectors.

**Candidates (masks on `FEATURE_NAMES`):**
- A: `r_model` only (max-prob)
- B: `r_model` + `n_missing`
- C: current 7 features minus `d_observed`
- D: current 7 features

**Protocol:** same grouped cross-fit as production (stacker + calibrator + thresholds fit on other folds’ OOF; score held-out only). Metrics: risk–coverage area, ECE, Brier, mean \(R_{\mathrm{final}}\). Also report paired grouped-bootstrap CI of each set’s area minus the best set’s area (`dup_group_id`).

**Selection rule (locked before results):** among sets whose risk–coverage area is statistically tied with the best (paired Δ CI vs best contains 0), choose the **simplest**; ties among tied sets go to the simpler set. Simplicity order: A ≺ B ≺ C ≺ D. If no other set ties the best, keep the best.

**Honesty note (to be filled after choice):** `n_missing` / `d_observed` were added after seeing earlier results; the final feature set is chosen on held-out evidence under this rule.

### Item 9 — result (after comparison)

| set | area | ECE | Brier | meanR | Δ vs best CI |
|-----|------|-----|-------|-------|--------------|
| A | 0.6714 | 0.1338 | 0.2276 | 0.6504 | [−0.033, +0.002] |
| B | 0.6671 | 0.1169 | 0.2228 | 0.6478 | [−0.037, −0.002] |
| C | 0.6865 | 0.1040 | 0.2073 | 0.6409 | best (tied with D) |
| D | 0.6865 | 0.1040 | 0.2073 | 0.6409 | tied with C |

**Chosen: A** (`r_model` / max-prob only) — simplest set whose paired Δ CI vs best contains 0. B excluded (CI entirely below 0).

**Honesty:** `n_missing` and `d_observed` were added after seeing earlier results; the feature set was chosen on held-out evidence under the pre-registered rule above. Locked in `CHOSEN_FUSION_SET` / `locked_feature_mask()` for production, ablation, and item4.

## Round-2 Phase 3 — RC3 / RM7 (written with code; artefacts after Phase 5 train)

1. **Calibrator n:** Platt / beta / smoothed isotonic on **all cross-fitted OOF raw scores** (~344), not the ~70 inner holdout. Production stacker still fits on all OOF; metrics remain held-out fold scores.
2. **R_final bound and interval:** `conf_eps` clips away from exact 0/1; `r_final_interval` reports grouped-bootstrap stacker-plus-calibrator quantiles, or NaN bounds when no interval ensemble is fitted—never a fabricated constant range.
3. **Decision utilities:** `DECISION_BAND_UTILITIES` = engineering-cost asymmetry (false accept −1, correct accept +1; abstain-on-error mildly rewarded). Alternative: `target_accept_rate` coverage mode for sensitivity. Bands forced non-empty (`min_gap=0.05`); no silent merge of caution≡accept.
4. **No 0.49 clamp:** abstain + high R_final keeps true R_final; `decision_override_reason` records the gate/Critic override.
5. **RM7:** `compare_fusion_sets` + `component_aurocs` on final leakage-safe R; keep set A only if it still wins/ties on RC area — do not force multi-channel fusion.

## Round-2 Phase 4 — RM8–RM11 + RC4/RC6 hooks (code only; no full train)

1. **RM8:** Set-size distribution per class/fold in `m1_results`; compare Mondrian / marginal / CV+ / ordinal nonconformity / class-conditional grade-3 α; emit coverage–size–abstention trade-off.
2. **RM10 decode:** Compare argmax vs median by **mean fold QWK** on the production model; adopt median only if it wins; else justify argmax. Nested selection remains **best mean inner QWK** (no silent M1 regression). Capacity-matched multiclass vs ordinal table for F8.
3. **RM11:** Broaden RF (`min_samples_leaf`, `max_features`); booster subsample/L2; SVM = StandardScaler + balanced weights + CalibratedClassifierCV. Candidate table carries fold QWK CIs for all families.
4. **RM9:** M2 = importance overlay (metrics ≡ M1); all-fold permutation + local attributions; Critic rule `implausible_explanation_direction`; F15 keys `importances_mean` / `ranked`.
5. **RC4/RC6 hooks:** `matched_coverage.comparisons` covers M3–M7 versus M1 independently at each system's own coverage, with paired grouped CIs; robustness emits flip/|Δp|/entropy/set size/R_final (+ CIs); missing-data includes abstention, accepted accuracy, confidence/accuracy change, sample size, and CI-gated degradation flags.

## Round-2 Phase 5 — USER long runs (RC1, RC6, RM12, RM13, Rm8 metadata)

**Agent ships code only; user runs long jobs** (constraint 5 / out-of-scope agent train·evaluate).

1. **Rm8:** `build_run_metadata` → every `m1_results.json`, `ablation_results.json`, `triplet_label_sensitivity.json`, fusion set-compare JSON (mode, date_utc, git_hash, package_versions, command).
2. **RM12:** `scripts/triplet_label_sensitivity.py` — production M1+M7 under `(σc, σt, Wet)` groups **and** side-by-side `dup_group_id` headlines → `artifacts/baselines/triplet_label_sensitivity.json`.
3. **RM13:** same script — three policies (`as_given`, `drop_rows_315_344`, `earliest_of_conflict`) with production M1+full M7; `missing_d_comparison` with/without 315–344 (report narrative in Phase 6).
4. **RC1/RC6:** full `train` + `evaluate` (no `--quick`); iterative imputer in missing-data; all fold test cases in robustness.
5. **RM7 artefact:** after train, `scripts/compare_fusion_sets_slice4.py` on refreshed `fusion_R_oof.npy`.

## Round-2 Phase 6 — Examples, figures, notebook, reports (RC7–RC10, RC4/RC6 report, RM14, §8)

**Agent: code/docs/tests only. User runs** `generate_examples.py`, `generate_figures.py`, `jupyter nbconvert --execute`.

1. **RC7:** `failure_case` = held-out **accepted-wrong** preferring `|ŷ−y|≥2` (case-42 class when available); `abstain_case` separate; `failure_analysis.md` lists inputs, agent flags, skips, which check would catch.
2. **RC8:** `scripts/archive_stale_artefacts.py` moves item4 / old fusion slices / `*_llm.json` traces → `artifacts/archive/`; `case_corrupt_ood` records real `perturbation`; examples carry `case_no`/`true_grade`/`fold_index`/`fold_model`.
3. **RC9 + §8:** `generate_figures.py` raises on missing keys; no CM fabrication; captions include n/folds/repeats/model; F2 post-RC5 cycle semantics; F9b matched-coverage; F10 honest R_final loss; F11 real reliability bins.
4. **RC10:** notebook raw CSV → preprocess → load models by sha256 → eval tables → ≥3 held-outs → embed F01/F08/F12; `README_DELIVERY.txt` aligned with `DELIVERABLES.md`.
5. **RM14 / reports:** `feedback_response.md` Round-2 evidence map; technical report + COVER_NOTE + README synced to full-mode numbers (quick 0.632 retired).
