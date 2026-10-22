# Point-by-point response to client Feedback

For each ID: **fixed** / **changed** / **disputed**, with evidence paths. Delivered system = **rules FSM**; numbers from shipped artefacts (`headline_snapshot.json` + baselines JSON).

## Critical (C1–C8)

| ID | Status | Evidence / files |
|----|--------|------------------|
| **C1** Critic recheck never runs | **fixed** | Keep Critic until recheck; `graph/runner.py`, `tests/test_routing.py`; traces show `recheck_*` |
| **C2** Loop limit → normal decision | **fixed** | `max_loops` → forced abstain; `forced_abstain_case.json`; routing tests |
| **C3** Fusion / confidence degenerate | **changed** | Cross-fit recalibration; learned thresholds only by default; `fusion/stacker.py`, `agents/tools.py`, Slice 2 scripts |
| **C4** R_final ranks worse than max-prob | **changed** (negative retained) | Documented; RC max-prob 0.668 vs R_final 0.535 under set A — `ablation_results.json` |
| **C5** Abstain/failure examples | **fixed** | Held-out abstain #77; `failure_analysis.md`; `scripts/generate_examples.py` |
| **C6** Robustness leakage / not all folds | **fixed** | Fold-local ensembles; all-fold robustness protocol; `eval/ablation.py` |
| **C7** Missing-data not production/physical | **fixed** | Physical masks + physics recon; `ablation_results.missing_data` |
| **C8** No figures / empty notebook | **fixed** | `scripts/generate_figures.py` → `artifacts/figures/F01`–`F17`; notebook updated (execute to save outputs) |

## Major (M1–M16)

| ID | Status | Evidence / files |
|----|--------|------------------|
| **M1** Empirical criteria | **fixed** | `physics/criteria.py`; Russenes/Kidybinski/brittleness in `R_physics` |
| **M2** Tolerances + dual data policy | **fixed** | Tight abs/rel tol; as-given vs recompute; F5 |
| **M3** Physically consistent perturbations | **fixed** | Perturb primaries; recompute derived; `uncertainty/estimates.py` |
| **M4** Component R_* / fusion features | **changed** | Component calibration scripts; set A remains production |
| **M5** Critic escalate to abstain | **fixed** | Severe rules → abstain; M7 abstention rises; `agents/tools.py` |
| **M6** Fold-local ensembles | **fixed** | No full-data ensemble inside OOF; in-memory fold fusion |
| **M7** Ablation M2/M3 active | **fixed** | Definitions + wiring in `eval/ablation.py` |
| **M8** Ordinal variants | **fixed** | Frank–Hall; median vs argmax table in `m1_results.json`; F8 |
| **M9** ≥50 trials; replace ExtraTrees | **fixed** | `n_trials: 50`; MLP; broader SVM; `ml/tuning.py` |
| **M10** Production = mean inner | **fixed** | `fold_winner_instability` in `m1_results.json` |
| **M11** Mondrian + contiguous sets | **fixed** | Production mondrian_split; F12 |
| **M12** Combo/scale + OOD alts | **fixed** | `quality/checks.py` |
| **M13** Triplet sensitivity | **fixed** | `slice6_m13_m14.json` |
| **M14** Label policies | **fixed** | Same artefact; F6 highlights 315–344 |
| **M15** Engineering decision text | **fixed** | Intensity names, set, R_final, action; consistency clamp |
| **M16** LLM stance | **fixed** | Rules FSM delivered; LLM compare STALE; methodology § Delivered system |

## Minor (m1–m11)

| ID | Status | Evidence / files |
|----|--------|------------------|
| **m1** Conflicting numbers | **fixed** | Single snapshot + history appendix in `technical_report.md` |
| **m2** Stale artefacts mixed | **changed** | LLM compare archived STALE; regenerate examples/figures from final models |
| **m3** Thin agent specs | **fixed** | Full specs in `methodology.md` |
| **m4** Fusion maths missing | **fixed** | Equations in `methodology.md` |
| **m5** Unused pins / LangGraph wording | **changed** | Documented unused langgraph/langchain; FSM described accurately; matplotlib/seaborn used for F1–F17 |
| **m6** Absolute paths | **changed** | Config resolve_path; regenerate examples; avoid shipping machine-local paths |
| **m7** Dev files in delivery | **changed** | Document exclude `.cursor/`, `catboost_info/`, context pack from zip |
| **m8** Python version drift | **fixed** | `environment.yml` + `pyproject.toml` aligned to **3.12** |
| **m9** Ablation overwrites fusion.joblib | **fixed** | In-memory fold fusion (Slice 2/5) |
| **m10** In-sample headline examples | **fixed** | Held-out primary examples; production labelled |
| **m11** Failure analysis “not provided” | **fixed** | `failure_analysis.md`; `DELIVERABLES.md` updated |

## Regeneration commands

```bash
.venv/bin/python scripts/generate_figures.py
.venv/bin/python scripts/generate_examples.py
.venv/bin/python -m rockburst.cli train          # long
.venv/bin/python -m rockburst.cli evaluate       # long; or --quick
jupyter nbconvert --to notebook --execute notebooks/end_to_end_demo.ipynb --inplace
```
