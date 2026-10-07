# Claude agent context pack — rockburst project review

Share this file with an external judge. It consolidates **essential code + result summaries** (Priority 1–3) without secrets, `*.joblib`, `planner_cache/`, or the raw PDF.

**How to use:** Check Priority 1 first (deliverability). Then Priority 2 (supporting evidence). Priority 3 is spot-checks.

**Note on `data_audit.json`:** There is no `artifacts/baselines/data_audit.json`. The live audit is `data/processed/data_audit.json` (also mirrored in part inside `dup_sensitivity.json` → `default_audit`).

**Note on conformal:** Calibration / quantile logic lives in `src/rockburst/uncertainty/conformal.py`. `estimates.py` only does entropy, disagreement, stability, and calls `conformal_prediction_set`.

---

## Line counts (`wc -l`)

| Lines | Path |
|------:|------|
| 344 | `src/rockburst/graph/runner.py` |
| 767 | `src/rockburst/eval/ablation.py` |
| 34 | `src/rockburst/fusion/features.py` |
| 151 | `src/rockburst/fusion/stacker.py` |
| 88 | `src/rockburst/uncertainty/estimates.py` |
| 216 | `src/rockburst/uncertainty/conformal.py` *(needed for calib; not in original list)* |
| 638 | `src/rockburst/ml/train.py` |
| 257 | `src/rockburst/ml/tuning.py` |
| 166 | `src/rockburst/data/audit.py` *(dup_group_id)* |
| 110 | `src/rockburst/quality/near_dup.py` |
| 25 | `src/rockburst/quality/fold_ref.py` |
| 130 | `src/rockburst/quality/checks.py` |
| 75 | `configs/default.yaml` |
| ~12.7k | `artifacts/baselines/m1_results.json` |
| 1128 | `artifacts/baselines/ablation_results.json` |
| 186 | `artifacts/baselines/dup_sensitivity.json` |
| 297 | `src/rockburst/agents/tools.py` |
| 64 | `src/rockburst/agents/schemas.py` |
| 167 | `src/rockburst/physics/relations.py` |
| 252 | `tests/test_leakage.py` |
| 817 | `artifacts/baselines/llm_planner_comparison.json` |
| 88 | `data/processed/data_audit.json` |
| 88 | `reports/technical_report.md` |
| 88 | `reports/methodology.md` |
| 75 | `README.md` |
| 31 | `Makefile` |

---

# PRIORITY 1 — deliverability

## 1. Planner / routing — `src/rockburst/graph/runner.py` (344 lines)

### Actions
`data_quality | geomechanics | predict | uncertainty | critic | finalize`

### Rule planner (`rule_based_next_action`)
Fixed happy path unless Critic intervenes:

1. If `data_q is None` → `data_quality`
2. If `phys is None` → `geomechanics` *(note: condition is `(missing or suspicious or True)` — always true once data_q exists, so physics always runs)*
3. If `pred is None` → `predict`
4. If `unc is None` → `uncertainty`
5. Critic **only if** `allow_critic` and any of:
   - `model_disagreement >= routing.disagreement_threshold` (default 0.35)
   - `len(prediction_set) >= large_set_size` (default 3)
   - `data_q.ood_flag`
   - `physics_score < 0.6`
6. Else → `finalize`

### Critic recheck (max 1 per kind)
If Critic ran and `endorse is False`:
- `request_recheck == "physics"` → clear `phys`+`critic`, route `geomechanics` (once)
- `"data"` → clear `data_q`+`critic`, route `data_quality` (once)
- `"model"` → clear `pred`+`unc`+`critic`, route `predict` (once)
- else → `finalize`

**Important:** Recheck clears intermediate state and re-runs tools, but **finalize always calls `fuse_assessment`**, which does **not** change `predicted_grade` based on Critic. Grade always comes from `pred.predicted_grade`. Decision gates are fusion thresholds + a few hard abstain rules in `fuse_assessment`. That is why ablation `decision_changed` stays 0 unless grade or `recommended_action` differs vs M6.

### LLM planner
- Optional Gemini (`use_llm=True`); JSON `{"action": ...}` only.
- Prompt prefers DQ→phys→predict→unc; critic on OOD/low physics/disagreement.
- Cache under `artifacts/planner_cache/` keyed by SHA256(prompt).
- On failure → `fallback:` rules.
- Trace tags: `llm:`, `rules:`, `fallback:`.

### Forced abstain
If loop exceeds `max_loops` without finalize → fuse anyway, set `recommended_action="abstain"`, append `forced_finalize`.

### Fold injection for OOF eval
`cfg["_fold_pipeline"]`, `_fold_ensemble`, `_conformal_state`, `_fold_model_name` override production artefacts during ablation.

---

## 2. Ablations M1–M7 — `src/rockburst/eval/ablation.py` (767 lines)

### Definitions (from code + results)
| ID | What it is |
|----|------------|
| **M1** | Fold ML predict only (argmax grade) |
| **M2** | M1 metrics + permutation-importance sidecar |
| **M3** | If `physics_score < 0.5`, blend probs toward uniform; else = M1 |
| **M4** | Same grade as M1; abstain flag if `stability_score < 0.45` or `\|set\|≥3` |
| **M5** | **Real system:** `run_monolithic_fixed_sequence` — DQ→phys→predict→unc→`fuse_assessment`, no coordinator, no Critic |
| **M6** | **Real system:** `run_graph(..., use_llm=False, allow_critic=False)` |
| **M7** | **Real system:** `run_graph(..., allow_critic=True)` |

### Which model is used
```python
best_name = m1_meta["nested_unbiased"]["production_model"]  # or best_model
# → catboost__knn (nested mode of fold winners)
```
Not the selection-biased winner (`rf__iterative`). Same fold pipe + conformal + fold-trained fusion for M5/M6/M7.

### Bootstrap CIs
- Pool case-level preds across `n_outer=5 × n_repeats=5` (= 1720 rows = 5×344).
- Resample unique `dup_group_id` with replacement (400 boots; 50 if quick).
- Recompute accuracy / QWK / severe / abstention on selected cases.
- Paired Δ are **fold-level** (`fold_list_ci`), noted separately.

### Critic / coordinator stats logic
- `decision_changed`: M7 `recommended_action` ≠ M6 **or** grade ≠ M6 grade.
- Result: **`decision_changed = 0`** of 1720 (rechecks fire but do not change grade/decision).
- `route_diverged_from_m5`: compares M6 action list to fixed M5 sequence → **0** under rules planner (M6 always walks the same sequence when Critic disabled).

### Headline results (`ablation_results.json`)
- `m1_best`: `catboost__knn`
- M1–M7 **identical** QWK mean **0.667** [0.620, 0.712]; accuracy **0.587**
- Abstention: M1/M3=0; M4=**0.392**; M5=M6=M7=**0.053**
- Paired Δ QWK all **0.0**
- Critic fired **730/1720 (42.4%)**; recheck **87**; `decision_changed` **0**
- `agent_value_finding`: M5≡M6≡M7 on QWK and abstention
- Value of agents shows in **accepted vs abstained** (M7): accepted n≈1628 acc 0.600 / QWK 0.685; abstained n≈92 acc 0.359 / QWK 0.316

---

## 3. Fusion — `fusion/features.py` (34) + `fusion/stacker.py` (151)

### Live score vector
`compute_reliability_features` → `pack_r_vector(r_data, r_physics, r_model, r_uncertainty, r_stability)`.

### Training
`ReliabilityFusion`: `LogisticRegression(class_weight="balanced")` on OOF live R → `y_correct` (1 if base grade correct).

### Monotonicity clipping
After fit: `coef_ = np.maximum(coef_, 0)`.

### `r_final`
Fitted → `predict_proba[..., 1]`; else heuristic weights `[0.2,0.2,0.25,0.2,0.15]`.

### Threshold selection (`select_decision_thresholds`)
Grid search abstain/accept on validation R only; utility rewards correct accepts, penalizes wrong accepts, mild reward for abstaining on errors. Sets:
`abstain_r_final`, `accept_with_caution`, `accept_r_final`, `engineering_verification`.

### Production thresholds (from `m1_results.json`)
```
abstain_r_final ≈ 0.65
accept_with_caution ≈ 0.70
accept_r_final ≈ 0.75
engineering_verification ≈ 0.90
```
Fusion coef (nonneg): `[[0, 0, 2.245, 1.816, 0]]` → only `r_model` and `r_uncertainty` carry weight after clip.

---

## 4. Uncertainty / conformal

### `estimates.py` (88) — runtime scores
- Entropy, ensemble disagreement (`1 - majority_frac`)
- Perturbation stability → `r_stability`
- `r_uncertainty = 1 - (0.6·H/log4 + 0.4·disagreement)`
- Calls `conformal_prediction_set`

### `conformal.py` (216) — calibration (the part you asked about)
- Nonconformity: `1 - p_y`
- Quantile: `k = ceil((n+1)(1-α))`, take `sorted[k-1]` (`finite_sample_quantile`)
- Methods: `marginal_split`, `mondrian_split` (per-class if count≥5), `cv_plus_grouped`
- **Group-aware calib holdout:** `_split_fit_calib` in `train.py` — StratifiedGroupKFold on **outer-train** groups; first fold’s “test” slice = calib, rest = fit. Disjoint from outer test.
- Chosen method in artefacts: **`marginal_split`** (coverage ≈0.898, mean set ≈2.11; closest to nominal 0.90)

---

## 5. Nested training — `ml/train.py` (638) + `ml/tuning.py` (257)

### Fold construction
- Outer: `StratifiedGroupKFold(n_outer=5, groups=dup_group_id)`
- Inner (Optuna): `StratifiedGroupKFold(n_inner=3)` inside each outer train
- Equal Optuna budget `n_trials=8` per model family (logistic, svm, rf, extratrees, xgb, lgbm, catboost, ordinal_logistic) × imputers (median/knn/iterative)

### Selection
1. **selection_biased:** argmax **pooled outer-OOF QWK** → `rf__iterative`, QWK **0.685**, acc **0.616**
2. **nested_unbiased:** per outer fold pick model×imputer by **inner QWK only**, then score that fold’s outer test
   - Fold winners: `extratrees__knn`, `catboost__knn`×3, `lightgbm__knn`
   - **Mode of nested winners = production:** `catboost__knn`
   - Nested pooled QWK **0.621**, acc **0.567**
3. Headline `best_model` / artefacts use **nested mode**, not biased winner

### SVM
Restored via `CalibratedClassifierCV(sigmoid, cv=2)`; reduced search space, equal trial budget.

---

## 6. `dup_group_id` + quality fold safety

### Build — `data/audit.py` → `assign_dup_groups`
- Features: raw measured only `sigma_theta, sigma_c, sigma_t, wet, d` (`near_dup.py` `RAW_DUP_COLS`)
- Edge if relative diff `< rel_tol` on every overlapping finite feature (min overlap 3)
- Groups = connected components; conflicting labels **kept**
- Default `near_dup_rel_tol=0.01` from config
- Called from `process_raw`

### `quality/fold_ref.py`
Train-fold reference **drops case `no` and entire `dup_group_id`**.

### `quality/checks.py`
IQR outliers, SCF OOD, near-dup search (same rel_tol), IsolationForest → `data_reliability` / `ood_flag`. Defensively drops self/group again.

---

## 7. `configs/default.yaml` (75 lines) — full

```yaml
seed: 42
paths:
  pdf: sustainability-2534712-supplementary.pdf
  raw_csv: data/raw/rockburst.csv
  processed_csv: data/processed/rockburst_processed.csv
  audit_json: data/raw/parse_audit.json
  artifacts: artifacts
  traces: artifacts/traces
  models: artifacts/models
  baselines: artifacts/baselines
  examples: artifacts/examples

features: [sigma_theta, sigma_c, sigma_t, scf, b1, b2, wet, d]
target: rockburst_grade
group_col: dup_group_id

cv: {n_outer: 5, n_inner: 3, n_jobs: 1, n_repeats: 5}
tuning: {n_trials: 8}

gemini:
  planner_model: gemini-3.5-flash   # was 3.8-flash; free-tier quota note in comments
  lite_model: gemini-3.5-flash
  temperature: 0.0
  max_retries: 5
  cache_dir: artifacts/planner_cache

physics:
  rel_tol: 0.05
  abs_tol: 0.05
  scf_ood_threshold: 2.0
  wet_range: [0.5, 40.0]
  d_range: [50.0, 2500.0]

quality:
  iqr_factor: 3.0
  near_dup_rel_tol: 0.01

uncertainty:
  mild_eps: 0.05
  strong_eps: 0.15
  n_perturb: 20
  conformal_alpha: 0.1

routing:
  max_loops: 8
  disagreement_threshold: 0.35
  abstain_r_final: 0.45          # overridden by learned fusion thresholds at serve time
  accept_r_final: 0.70
  large_set_size: 3

fusion:
  decisions: [accept, accept_with_caution, engineering_verification, abstain]
```

---

## 8–10. Results JSON summaries

### `m1_results.json` — top-level keys
`seed, n_outer, n_inner, n_trials, search_spaces, models, complete_case_n(213), d_missing_n(131), conformal, protocol_note, candidate_table(24), selection_biased, nested_unbiased, best_model(=catboost__knn), best_qwk(=0.621 nested), best_qwk_selection_biased(=0.685), best_qwk_nested_unbiased, best_params, model_path, conformal_state, ensemble_path, fusion_path, fusion_coef, decision_thresholds, fusion_r_final_ece(~0.141), fusion_mean_r_final(~0.713), fusion_oof_accuracy_of_base(~0.567), fusion_reliability_diagram`

**Top candidates by selection-biased pooled_oof.qwk:**  
rf__iterative 0.685 → rf__median 0.681 → extratrees__median 0.676 → catboost__iterative 0.674 → … → svm__* ~0.35

### `ablation_results.json` — top-level keys
`m1_best, n_outer, n_repeats, ci_method, definitions, ablations, paired_differences, risk_coverage, critic_statistics, coordinator_m6_vs_m5, trace_samples, n_cases, protocol, agent_value_finding, robustness, missing_data`  
(+ M2 note). See §2 for numeric headline.

### `dup_sensitivity.json`
| rel_tol | n_groups | multi | conflict groups | conflict pairs |
|--------:|---------:|------:|----------------:|---------------:|
| 0.005 | 325 | 19 | 15 | 15 |
| **0.01** (default) | **319** | **24** | **18** | **19** |
| 0.02 | 314 | 26 | 21 | 25 |

Justification: Table S1 near-copies differ ~0.1–0.2% on σθ; 1% catches the block.

---

# PRIORITY 2 — supporting evidence

## 11. Critic / tools / schemas

### `agents/schemas.py`
Pydantic I/O: `DataQualityOutput`, `PhysicsOutput`, `PredictiveOutput`, `UncertaintyOutput`, `CriticOutput` (`request_recheck ∈ {none,physics,data,model}`, `endorse`), `AssessmentOutput`.

### Critic rules (`run_critic` in `tools.py`)
| Condition | Effect |
|-----------|--------|
| `physics_score < 0.6` (+ high `r_model>0.8`) | challenge; `request=physics`; `endorse=False` |
| `ood_flag` and `r_model>0.75` | contradiction; `request=data`; `endorse=False` |
| `disagreement ≥ threshold` | challenge; `request=model` (if none yet); `endorse=False` |
| `\|prediction_set\| ≥ large_set_size` | challenge; `endorse=False` (may leave `request=none`) |
| B1 inconsistency in physics issues | challenge; prefer `request=physics` |

### Extra abstain gates in `fuse_assessment` (after fusion.decide)
- `|set|≥3` and `stability_score < 0.55` → abstain
- `disagreement ≥ 0.5` and `r_final < 0.75` → abstain
- OOD + `physics_score < 0.7` + `r_model < 0.6` → abstain

**Grade is never rewritten by Critic.**

---

## 12. Physics — `physics/relations.py`
Exact (rel_tol/abs_tol = 0.05):
- `SCF = σθ/σc`, `B1 = σc/σt`, `B2 = (σc−σt)/(σc+σt)`
Empirical: Wet ∈ [0.5, 40], D ∈ [50, 2500]; SCF OOD if > 2.0  
Score: exact_ok/3 − penalties (wet OOR 0.1, D OOR 0.05, D missing 0.05, SCF OOD 0.15).

---

## 13. `tests/test_leakage.py` (252) — would fail if leak returns
| Test | Asserts |
|------|---------|
| `test_dup_group_never_spans_train_test` | StratifiedGroupKFold train/test group sets disjoint |
| `test_known_pairs_grouped_and_fold_safe` | Pairs (6,315), (14,316), (8,322), (1,327) share group; folds safe |
| `test_imputer_scaler_fit_on_train_only` | Prep fit on train only |
| `test_target_never_used_for_imputation` | Target not in X |
| `test_fold_reference_excludes_self_and_group` | Ref drops self + group |
| `test_fusion_monotonicity` | coef≥0; worse R ⇒ ≤ r_final |
| `test_train_serve_reliability_path_identical` | features.py ≡ pack_r_vector |
| `test_conformal_calib_disjoint_from_fit` | fit/calib/test pairwise disjoint |
| `test_finite_sample_quantile_formula` | n=10, α=0.1 → 10.0 |
| `test_conformal_coverage_near_nominal` | mean cov in [0.80, 0.98] |
| `test_max_loops_forced_abstain` | patched infinite loop → abstain + forced_finalize |
| `test_thresholds_selected_on_validation_only` | API exists; abstain < accept |

These are real guardrails, not log-only claims — reintroducing group leakage or target-in-impute would fail the corresponding tests.

---

## 14. `llm_planner_comparison.json`
- `model_id`: gemini-3.5-flash (started on 3.8-flash; quota exhausted)
- `n_cases`: 32; `action_agreement_rate`: **0.906**; `fallback_rate`: **0.244**
- `planner_stats`: api_calls=19, cache_hits=22, fallbacks=10, llm_steps=31, rules_steps=40
- **3 disagreement cases** (146, 147, 57): different route traces, **same final decision `accept`**
- No final-decision differences reported

---

## 15. Data audit (`data/processed/data_audit.json`)
- n_rows=344, d_missing=131, n_dup_groups=319, multi=24, conflicting_label_groups=18
- known_problem_rows: bad_b1 `[20,21,244]`, ood_scf `146–157`
- near_dup_rel_tol=0.01; features = RAW_DUP_COLS

---

## 16. Reports (current)

### `technical_report.md` — key claims
- Nested-unbiased production `catboost__knn` QWK 0.621 vs biased `rf__iterative` 0.685
- Ablation M5≡M6≡M7; Critic fires but `decision_changed=0`
- Agents help via abstention risk control, not QWK lift
- §10 constraints checklist all marked **met**
- Limitations: n=344, 19 conflicting pairs, Gemini quotas, prod examples look better than OOF

### `methodology.md` — architecture
Coordinator routes tools; fusion decides trust/abstain; nested selection; group-aware conformal; leakage guard documented. Matches code above.

---

# PRIORITY 3 — spot checks

## 17. Traces (`artifacts/traces/`)

**Routing that diverges from fixed sequence (Critic recheck):**

`case_146.json` (OOD SCF block) — rules + data recheck:
```
rules:data_quality → geomechanics → predict → uncertainty → critic
  → data_quality → critic → finalize
recommended_action: accept  (grade 3, ood Yes, r_final≈0.925)
```

`case_244.json` (bad B1) — physics recheck:
```
… → critic → geomechanics → critic → finalize
recommended_action: accept  (physics_score≈0.33, still accept via high r_model)
```

`case_146_llm.json` — LLM routing differs:
```
llm:data_quality → llm:critic → fallback:geomechanics → llm:predict
  → fallback:uncertainty → llm:finalize
decision still accept
```

**Forced abstain** (not in traces folder as live assess; in examples):
`artifacts/examples/forced_abstain_case.json` — `route_trace` ends with `forced_finalize`, `abstained: true`.

Among ~40 traces: many Critic routes exist; **none** set `decision_changed` in ablation sense (grade/action vs no-Critic path unchanged in eval).

---

## 18. Examples

### Held-out — `artifacts/examples/heldout_case_15.json`
```json
{
  "predicted_grade": 0,
  "true_grade": 0,
  "recommended_action": "engineering_verification",
  "final_system_confidence": 0.6687773809518929,
  "route_trace": ["rules:data_quality","rules:geomechanics","rules:predict","rules:uncertainty","rules:finalize"],
  "example_protocol": "held-out fold model: case was in outer test fold; pipeline fit only on fold train/fit indices",
  "fold_model": "catboost__knn"
}
```

### Production — `artifacts/examples/case_1.json`
```json
{
  "predicted_grade": 3,
  "prediction_set": [1,2,3],
  "recommended_action": "engineering_verification",
  "final_system_confidence": 0.6979392027111886,
  "route_trace": ["…","rules:critic","rules:finalize"],
  "example_protocol": "production model, case seen in training (full-data refit of best_baseline.joblib)"
}
```

### Forced abstain — `forced_abstain_case.json`
`recommended_action: abstain`, `decision: "Forced stop at loop limit…"`, trace includes `forced_finalize`.

---

## 19. README + Makefile

**Pipeline:** `make pipeline` → extract → train → evaluate (no `--quick`).

**CLI:** `extract | train | evaluate | assess [--no N] [--llm]`

**Design rule:** planner only chooses next action; all numbers from deterministic Python; fusion decides trust/abstain.

**Makefile targets:** `setup, extract, train, evaluate, pipeline, test, assess-examples`.

---

# Judge cheat-sheet (answers to the stated questions)

| Question | Answer from this pack |
|----------|------------------------|
| Can coordinator route differently from fixed sequence? | **Yes in code** (Critic rechecks; LLM can reorder). **Under M6 rules ablation:** route divergence from M5 = **0%**. Live traces show Critic loops. |
| How does Critic recheck work? | Clears one of phys/data/model (+critic), max 1 each; re-runs tool; then finalize again. |
| Why `decision_changed` always 0? | Metric compares M7 vs M6 **grade/`recommended_action`**. Critic does not alter grade; fusion decisions usually identical after recheck. |
| Are M5/M6 real systems? | **Yes.** M5 = monolithic fuse path; M6 = rules graph without Critic; M7 = with Critic. |
| Biased vs nested model? | Ablations + production use **nested mode `catboost__knn`**. Biased winner `rf__iterative` reported separately. |
| Bootstrap CIs? | Grouped bootstrap over `dup_group_id` on pooled case-level preds. |
| Conformal split / quantile / groups? | Fit/calib via group SGKF inside outer train; quantile `⌈(n+1)(1-α)⌉/n`; method = marginal_split. |
| Leakage tests real? | Yes — would fail on group leakage, train-only impute, fold-ref, calib disjointness, forced abstain. |

---

*Generated for external review. No API keys, joblib weights, or planner_cache contents included.*
