# Figure captions (F1–F17)



**F01_system_architecture.** Visible arrows: Case→Coordinator→agent tools→Critic→Fusion→engineering output; shared CaseState. Rules FSM; LLM planner optional, not for headlines.

**F02_state_transitions.** Post-RC5 routing: Critic→recheck cycles (limit=3), max_loops=24 safeguard, termination/abstention conditions.

**F03_routing_traces.** Fresh traces: missingness, OOD, executed Critic recheck (cases [13, 67, 6]).

**F04_dataset_overview.** Grade counts, missing-D by grade and rows 315–344, SCF densities (n=344; densities clipped ≥0).

**F05_physics_consistency.** Listed vs recomputed SCF/B1/B2; tolerance ±2% rel / 0.006 abs in all panels; B1 mismatches annotated by case no (n=344).

**F06_near_dup_conflicts.** Conflicting near-duplicate pairs at rel_tol=0.05; 315–344 block and +1 shift pattern highlighted (n_pairs=37).

**F07_confusion_matrices.** Real M1 confusion counts + row-norm from nested_unbiased; no fabrication. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full. M7 grade QWK≡M1; abstention changes accepted subset (see matched_coverage).

**F08_model_comparison.** All candidates with fold QWK CIs; multiclass vs ordinal highlighted; nested-unbiased production=lightgbm__median; n_outer=5 (RM10/RM11).

**F09_ablation_m1_m7.** Ablation QWK, severe-error, abstention ± grouped-bootstrap CI. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full.

**F09b_risk_coverage_areas.** Pooled RC areas plus per-system same-coverage accuracy/severe deltas against M1 on N=1720 predictions (n=344, folds=5, repeats=5, model=lightgbm__median, mode=full).

**F10_risk_coverage.** Common-axis RC curves for ['M1', 'M4', 'M7'] with base-accuracy line. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full. M7/R_final loses to M1 max-prob on RC area (0.642 vs 0.653).

**F11_reliability.** R_final reliability bins + counts from fusion set C (ECE=0.06326617499476576, CI=[0.04956245352153262, 0.10658690718278299]); M1 ECE=0.23158585199435877. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full.

**F12_conformal.** Per-class coverage vs nominal 1−α, mean set size, fold variability; method=marginal_split, alpha=0.1.

**F13_robustness.** Full-n robustness flip/|Δp|/entropy/set-size/R_final mild vs strong. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full. Honest title: n_cases=1720.

**F14_missing_data.** All imputers incl. iterative + physics recon + train/inference protocol. n=344, folds=5, repeats=5, model=lightgbm__median, mode=full.

**F15_explainability.** All-fold permutation importance (n_folds=5) + SCF/Wet/σθ dependence; local explanations n=15 in artefact.

**F16_critic.** Per-rule fires / detection rates from critic_statistics (flagged vs unflagged error + CIs when present). n=344, folds=5, repeats=5, model=lightgbm__median, mode=full.

**F17_fusion.** Fusion weights + bootstrap CIs; component AUROC. chosen=C production_selected=C.
