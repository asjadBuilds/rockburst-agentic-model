# Figure captions (F1–F17)



**F01_system_architecture.** Rules FSM coordinator routes deterministic agents/tools; Critic may request rechecks; ReliabilityFusion emits R_final and engineering decision. LLM planner optional and not used for headline metrics.

**F02_state_transitions.** Routing conditions, Critic recheck loops, max_loops forced abstain, and fusion termination.

**F03_routing_traces.** Three routing traces: missingness, OOD, and Critic-triggered path (rules FSM).

**F04_dataset_overview.** Grade counts, missing-D patterns (by grade and rows 315–344), SCF densities by grade.

**F05_physics_consistency.** Listed vs recomputed SCF/B1/B2 with ±2% relative / 0.006 abs tolerance; flagged mismatches in red.

**F06_near_dup_conflicts.** Conflicting near-duplicate pairs at rel_tol=0.05; highlight involves rows 315–344 (M14).

**F07_confusion_matrices.** M1 and M7 confusion structure from ablation class-recall (row-normalised proxy when full CM absent).

**F08_model_comparison.** Pooled OOF QWK, ordinal MAE, severe-error by model×imputer; ordinal/Frank–Hall highlighted (M8/M9).

**F09_ablation_m1_m7.** Ablation QWK, severe-error, abstention with grouped-bootstrap CIs from ablation_results.json.

**F09b_risk_coverage_areas.** Risk–coverage curve areas by ablation ID when present in artefacts.

**F10_risk_coverage.** Risk–coverage curves for M1/M4/M7 when curve points shipped; else area summary from ablation_results.json.

**F11_reliability.** Reliability schematic with ECE from artefacts (M1 ECE=0.047605861893215945, R_final ECE=None).

**F12_conformal.** Per-class coverage vs nominal 1−α and mean set size for production method=mondrian_split.

**F13_robustness.** Flip rate, |Δp|, entropy Δ, R_final Δ for mild/strong ε with CIs (ablation robustness).

**F14_missing_data.** Missing-data QWK by physical mask / imputer and M7 path including physics reconstruction.

**F15_explainability.** Held-out permutation importance from m2_importance.json and SCF/Wet vs grade scatter (SHAP optional).

**F16_critic.** Critic rule fires and fire-rate/AUROC from ablation critic_statistics / agent_value_finding.

**F17_fusion.** Learned fusion coefficients from fusion.joblib; set-comparison / CI artefact alongside.
