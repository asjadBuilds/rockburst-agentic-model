# Point-by-point response — Round-3 remediation

This response maps the **Round-3 identifiers directly**. Numerical claims use full-mode artefacts only (`quick=false`, five outer folds × five repeats where applicable). The delivered coordinator is the deterministic rules FSM; optional Gemini routing is not used for reported metrics.

Status vocabulary:

- **Fixed and verified** — implementation, focused test, and regenerated evidence agree.
- **Fixed; delivery closure pending** — substantive correction is complete, but clean-checkpoint provenance or final packaging is completed in the closure phases.
- **Documented limitation** — an unfavorable result is retained rather than presented as an improvement.

## Critical findings

| ID | Status | Evidence |
|----|--------|----------|
| **TC1** Critic loop could not converge | **Fixed and verified** | Structured objections and resolution evidence in `agents/schemas.py`, `agents/tools.py`, `agents/rechecks.py`, and `graph/runner.py`. Rechecks cannot repeat on unchanged evidence; unresolved/no-new-evidence paths finalize conservatively. Tests: `test_phase2_rm1_rm6.py`. Full statistics: `ablation_results.json → critic_statistics`. |
| **TC2** Production fusion discarded agent evidence | **Fixed and verified** | The declared A–D rule selects production set **C**; inverted `r_data` is excluded. `fusion/set_compare.py`, `fusion/stacker.py`, `fusion_set_comparison_slice4.json`, F17. |
| **TC3** Calibrator/stacker mismatch and unusable bands | **Fixed and verified** | One cross-fitted production stacker/calibrator path; beta calibration on served-score protocol; learned nonempty bands; grouped-bootstrap intervals. `fusion/fold_oof.py`, `fusion/stacker.py`, `test_phase3_rc3_rm7.py`, F11/F17. |
| **TC4** Invalid matched coverage | **Fixed and verified** | Every M3–M7 system is compared at its own coverage against the same-size highest-confidence M1 subset on all 1,720 predictions, with paired grouped-bootstrap CIs. M7 Δ accuracy −0.032, CI [−0.054, −0.010]. `eval/metrics.py`, F9b/F10. |
| **TC5** CatBoost failures hidden as zero results | **Fixed and verified** | CatBoost uses compatible bootstrap/subsample settings; required candidate/fold failures raise immediately. Smoke and fail-fast tests: `test_phase4_modelling.py`. |
| **TC6** Failure example and analysis inadequate | **Fixed and verified** | Accepted-wrong case **88**, true strong/predicted light, absolute error **2**, distinct abstention case **129**. `failure_case.json`, `heldout_failure_case_88.json`, `failure_analysis.md`, `c5_selection_summary.json`. |
| **TC7** Stale/duplicate artefacts and absolute paths | **Fixed; delivery closure pending** | Stale outputs are archived and packaging excludes archive/development content. Repository-relative provenance and whole-delivery absolute-path scanning are finalized in closure Phases 3–5; this item is not closed until the clean package passes. |
| **TC8** Explanation-cache and imputer defects | **Fixed and verified** | Per-case explanation state; production-family/fold-trained validation; complete cases cannot emit imputation objections. `ml/explain.py`, `agents/rechecks.py`, `graph/runner.py`, `test_phase1_rc2_rc5.py`. |
| **TC9** Figures did not meet specifications | **Partly fixed; content closure pending** | F1–F17 exist as PNG/PDF and are now embedded in `technical_report.md`, but source inspection found remaining specification gaps in F4, F6, F7, F10, F11, and F16. In particular F11 still renders a base-curve placeholder. These figures must not be called accepted until their plotting code and regenerated outputs pass the original figure-by-figure requirements. |

## Major findings

| ID | Status | Evidence |
|----|--------|----------|
| **TM1** Duplicated/unjustified Critic rules | **Fixed and verified** | Duplicate rules consolidated; grouped-bootstrap detection statistics drive severity; ineffective severe rules are disabled. `run_critic`, `critic_statistics.rule_detection_rates`, F16. |
| **TM2** B1 voter and duplicate stress-ratio vote | **Fixed and verified** | B1 is removed from consensus when unsupported; one SCF-based stress-ratio vote remains; criterion agreement is reported honestly. `physics/criteria.py`, `criterion_label_agreement`. |
| **TM3** Incorrect/weak physical checks | **Fixed and verified** | Kirsch/overburden depth range, SCF≥1 routing, and every-case σθ/SCF and Wet monotonic checks. `physics/relations.py`, `quality/checks.py`, routing tests. |
| **TM4** Conformal choice and grade-3 variant | **Fixed and verified** | Marginal, Mondrian, and distinct grade-3 variants are compared on coverage, class coverage, size, and abstention. Production is **marginal_split**. `uncertainty/conformal.py`, `m1_results.json`, F12. |
| **TM5** Unsupported robustness/missing conclusions | **Fixed and verified** | Full 1,720 assessments, CIs, CI-gated flags, M7 abstention/accepted accuracy/confidence/accuracy changes, and actual scenario sample sizes. `eval/ablation.py`, F13/F14. |
| **TM6** Base overconfidence and missing-D shortcut | **Fixed and verified** | Post-hoc base calibration and observed-indicator/drop-row sensitivity are reported. `triplet_label_sensitivity.json`, technical report. |
| **TM7** RM12/RM13 results not reported | **Fixed and verified** | Triplet grouping and three label policies are in `triplet_label_sensitivity.json` and the technical report. |
| **TM8** Status/report claims overstated | **Fixed; delivery closure pending** | This response uses Round-3 IDs and distinguishes verified fixes, limitations, and remaining packaging/provenance closure. Final status is promoted only after clean-run/package gates pass. |
| **TM9** Notebook omitted preprocessing/evaluation | **Fixed and verified** | The executed notebook processes raw data and performs real held-out outer-fold M1/M7 assessment. `notebooks/end_to_end_demo.ipynb`. Portable path cleanup remains a closure Phase-3 gate. |
| **TM10** Tests missed defects/no transcript | **Fixed; delivery closure pending** | Tests cover objection resolution/non-repetition, train/serve equality, CatBoost smoke/fail-fast, complete-case imputation behavior, and artefact regeneration. A client-facing pytest transcript is added and packaged in closure Phase 5. |

## Minor findings

| ID | Status | Evidence |
|----|--------|----------|
| **Tm1** Stale coordinator description/default | **Fixed and verified** | `graph/runner.py` identifies the deterministic rules FSM and the optional fallback agrees with `configs/default.yaml` (`gemini-3.5-flash`). |
| **Tm2** Ungraded disagreement | **Fixed and verified** | Output includes Low/Moderate/High level and named parties. `inter_agent_disagreement`, routing/output tests. |
| **Tm3** Missing hazard statement | **Fixed and verified** | Decision text derives hazard guidance from grade, conformal set, criteria, and disagreement. Example JSON and failure analysis show the rendered statement. |
| **Tm4** Development files/missing Make targets | **Fixed; delivery closure pending** | Make targets cover preprocessing, training, evaluation, sensitivities, examples, figures, notebook, tests, archive, and package. Final removal/exclusion of every named development file remains a package-audit gate. |
| **Tm5** Untraceable dirty runs | **Fixed; delivery closure pending** | Metadata code records dirty state and configuration/input hashes. A single clean committed full run is required in closure Phase 4. |
| **Tm6** Figure formatting | **Partly fixed; content closure pending** | Most formatting was corrected, but F11 still contains placeholder text and remaining figure-specific corrections must be inspected after regeneration. |
| **Tm7** Missing methodology constants | **Fixed and verified** | Methodology documents fusion/calibration, thresholds, hard gates, and utility rationale; design decisions retain exact constants. |
| **Tm8** Remaining labels/comments inconsistent | **Fixed and verified** | Production near-duplicate tolerance is consistently 5%; pooled risk–coverage is used; recheck counters are separately named; archived files are not evidence. |

## Figure verification

| Figure | Status | Corrected content |
|--------|--------|------------------|
| **F1** | Regenerated | Full data flow, agent tools, shared state, Critic/recheck loop, fusion/output. |
| **F2** | Regenerated | Real state order, conditions, rechecks, and termination paths. |
| **F3** | Regenerated | Fresh missingness, OOD, and actual Critic-recheck traces. |
| **F4** | **Open** | Current implementation shows grade/missing-D/SCF summaries, not all eight features by grade. |
| **F5** | Regenerated | Readable SCF/B1/B2 consistency with flagged cases. |
| **F6** | **Open** | Current right panel plots earlier case number versus grade shift, not earlier versus later case number colored by shift. |
| **F7** | **Open** | Current output has M1 counts and normalized M1 only; M7 all/accepted matrices remain absent. |
| **F8** | Regenerated | Successful candidates, CIs, production and nested results. |
| **F9** | Regenerated | QWK, severe error, and abstention evidence with CIs. |
| **F9b** | Regenerated | Pooled areas and corrected same-coverage comparisons. |
| **F10** | **Open** | Pooled M1/M4/M7 curves are present, but entropy and bootstrap bands requested by the review remain absent. |
| **F11** | **Open** | Served-score bins are real, but the base panel still contains “Base bin curve not shipped” placeholder text. |
| **F12** | Regenerated | Method comparison, per-class coverage, and set sizes. |
| **F13** | Regenerated | Robustness changes against baseline without internal notes. |
| **F14** | Regenerated | Accuracy, confidence, abstention, CIs, and train/inference panel. |
| **F15** | Regenerated | Model dependence and per-case explanation evidence. |
| **F16** | **Open** | Current plotting path labels a generic detection-rate panel and does not reliably render the requested AUROC plus flagged/unflagged error comparison. |
| **F17** | Regenerated | Production set-C weights/CIs and confidence evidence. |

All figures are embedded in `reports/technical_report.md`; source PNG/PDF files and generated captions are under `artifacts/figures/`. Embedding is complete, but the figures marked **Open** above still require content corrections and regeneration.

## Honest limitations retained

- Agents do not improve grade QWK over M1; their measured contribution is routing and selective decision support.
- Served R_final does not outperform M1 max-probability on risk–coverage.
- M7 accepted accuracy is below M1 thresholded to the same coverage in the consolidated evaluation.
- Several Critic rules have only modest discrimination.
- Clean-checkpoint provenance, portable paths, the packaged pytest transcript, and final ZIP audit remain explicit closure gates rather than being prematurely labelled complete.
