# Technical Report — Rockburst Agentic AI

## Lead finding: label noise and nested headline

**Nested-unbiased M1 (headline):** pooled QWK **0.621**, accuracy **0.567** (production mode `catboost__knn`; fold winners mix ExtraTrees/CatBoost/LightGBM + KNN). See `artifacts/baselines/m1_results.json`.

**Label conflicts @ 1% near-dup grouping:** 18 conflicting-label groups / **19** pairs. Direction (later `no` − earlier): **+1 in 18/19**, mean Δ ≈ **1.05**. Rows **315–344** appear systematically one grade higher than earlier near-duplicates in **94%** of those pairs — state without over-interpreting as intentional relabeling.

| Policy | n | M1 QWK | M1 accuracy |
|--------|---|--------|-------------|
| Keep all (default) | 344 | 0.673 | 0.581 |
| Keep one (lowest `no` / conflict group) | 325 | 0.698 | 0.655 |
| Drop conflicting groups | 307 | 0.684 | 0.622 |

Near-dup tolerance **1%** chosen from NN distance gap (dense mass \<1%, sparse 1.5–5% tail), not by maximizing QWK. Pairs 19↔342 / 20↔328 group at 2%; 9↔337 at 5%. Full table: `artifacts/baselines/item5_dup_label_noise.json`.

---

## Dataset
- Source: Table S1 (`sustainability-2534712-supplementary.pdf`), **n = 344**.
- Features: \(\sigma_\theta,\sigma_c,\sigma_t,\mathrm{SCF},B_1,B_2,W_{et},D\); target grade \(0\)–\(3\).
- \(D\) missing in **131** rows.
- Issues retained as agent tests: bad \(B_1\), OOD SCF, conflicting near-dups.

## Training protocol
- Seed `42`. Nested grouped CV: `n_outer=5`, `n_inner=3`, Optuna `n_trials=8` / family.
- **Selection-biased** (do not use as headline): QWK **0.685**, acc **0.616**.
- **Nested-unbiased (headline):** QWK **0.621**, acc **0.567**.
- Conformal: **a priori `marginal_split`** (simplest finite-sample guarantee). Side-by-side coverage: marginal **0.898**, Mondrian **0.927**, CV+ **0.904**. Per-class coverage (Item 4): ≈ 0.90 / 0.88 / 0.90 / 0.93.
- Fusion: constrained non-negative logistic + isotonic; collinearity drops `r_uncertainty`; point estimate weights **only `r_model`**. Mean \(R_{\mathrm{final}}\) **0.572** vs base acc **0.567**; ECE **0.042** (was 0.141). Zero-weight flags: `r_data`, `r_physics`, `r_stability`.

## What each agent contributes (measured)

| Component | Measured effect |
|-----------|-----------------|
| Physics skip / conditional routing | M6 route ≠ M5 on **61%** of cases; **1050** physics skips / 1720 |
| Extra validation | Invoked **684** times |
| Critic + rechecks | Fire **49%**; recheck tools used (phys/data/model = 25/181/630); `decision_changed` **450** (downgrades only) |
| Grade / QWK / abstention M5 vs M6 vs M7 | **Identical** (QWK 0.667, abs 0.053 on full ablation) |
| Fusion `R_data`/`R_physics`/`R_stability` | **Zero weight** in point estimate |
| Critic flag AUROC | **0.592** (weak); large-set rule AUROC 0.626 |
| Risk–coverage: `R_final` vs max-prob | Areas **0.592 vs 0.668** — `R_final` does **not** beat max-prob |

Ablation M1 QWK **0.667** [0.620, 0.712] vs nested **0.621**: different protocol (single production mode on repeated OOF vs per-fold inner winners). Nested remains headline.

## Ablation table (grouped OOF, 5×5)

| Model | QWK | Accuracy | Abstention |
|-------|-----|----------|------------|
| M1–M3 | 0.667 | 0.587 | 0 / 0 / 0 |
| M4 | 0.667 | 0.587 | **0.392** |
| M5–M7 | 0.667 | 0.587 | **0.053** |

Accepted vs abstained (M7): accepted harder-filtered accuracy higher; abstained cases worse (see prior audit).

## Brief §10 constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| No prompt-only agents without tools | **met** | `agents/tools.py`, physics, ML, fusion |
| Do not claim more agents ⇒ better performance | **met** | M7 ≰ M1 on QWK; M5≡M6≡M7 on grade metrics |
| No test data in preprocessing / selection | **met** | Nested inner selection; fold impute/scale |
| No hand-picked reliability weights without validation | **partly** | Learned constrained fusion; heuristic fallback **removed**; zero-weight physics/data still a limitation |
| Do not hide when full system ≤ baseline | **met** | Paired Δ=0; Critic AUROC weak; `R_final` loses to max-prob |
| Not accuracy alone | **met** | QWK, severe-error, coverage, ECE, risk–coverage |
| Abstain when inadequate confidence | **met** | Gates + `max_loops`; Item 4 M7 abs ≈ 0.12 under refit fusion |

## Limitations
- \(n=344\); label noise (≥19 conflict pairs; 315–344 often +1).
- M7 does not improve grade metrics vs M1; routing/decision labels change, grades do not.
- Fusion point estimate ignores `R_data`/`R_physics`/`R_stability`.
- Ablation vs nested gap; sensitivity of hand-set constants was flat on a single-fold probe.
- Gemini quotas; rules planner is the default.

## Artefacts
| Path | Content |
|------|---------|
| `artifacts/baselines/m1_results.json` | Nested/biased M1, fusion, conformal |
| `artifacts/baselines/ablation_results.json` | M1–M7 + critic/coord stats |
| `artifacts/baselines/fusion_item2_refit.json` | Fusion before/after |
| `artifacts/baselines/item4_metrics.json` | Critic AUROC, risk–coverage, per-class |
| `artifacts/baselines/item5_dup_label_noise.json` | Tol sensitivity + label noise |
| `artifacts/baselines/threshold_sensitivity.json` | Constants sensitivity |
| `reports/preregistration.md` | Locked rules |
| `reports/audit_log.md` | Before/after by item |
