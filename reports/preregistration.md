# Pre-registration — follow-up round (conditional coordinator / fusion / thresholds)

Rules below are locked **before** the evaluation that measures them. Justified by the client brief or domain logic, not by test outcomes. If a rule is changed after seeing results, both versions are reported in `audit_log.md`.

---

## Item 1 — Routing and Critic (locked 2026-10-06, before Item 1 measure)

### 1a. Physics skip (cheap pre-check)

**Skip geomechanics** only when a pure deterministic pre-check shows:

1. Exact relations SCF, B1, B2 all agree within config `physics.rel_tol` / `physics.abs_tol` (defaults 0.05).
2. Wet is present and inside `physics.wet_range` (default [0.5, 40]).
3. D is present and inside `physics.d_range` (default [50, 2500]).

Otherwise run the full geomechanics tool. Missing D ⇒ do **not** skip (pre-check fails).

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

## Item 2 — Fusion refit (locked before Item 2 measure)

1. Fit binary logistic for “base prediction correct” on OOF live reliabilities.
2. **No** `class_weight="balanced"`.
3. Non-negativity via **L-BFGS-B bound-constrained** fit (`coef ≥ 0`, intercept free) — not post-hoc clipping.
4. Inner-fold feature ablation / collinearity: if `|corr(r_model, r_uncertainty)| ≥ 0.9` or VIF > 5, keep higher univariate AUC (tie → `r_model`); zero-weight features kept as reported flags.
5. Isotonic calibration of `R_final` fitted on **inner** holdout only.
6. Unfitted stacker: **abstain** (no heuristic `[0.2,0.2,0.25,0.2,0.15]`).
7. Missing-D confidence drop measured on held-out folds (observed D vs masked+imputed).

## Item 3 — Thresholds / conformal (locked before Item 3 measure)

- Decision bands as in Item 1 table; select four cutoffs on **inner** data per outer fold.
- Headline conformal method: **`marginal_split`** a priori (simplest finite-sample guarantee). Mondrian and CV+ reported side-by-side; not chosen by closeness to 0.90.
## Item 5 — Near-dup tolerance (locked before Item 5 measure)

- Choose `near_dup_rel_tol` from NN max-rel-distance gap (dense mass below ~1%, sparse 1.5–5% tail), **not** by maximizing QWK.
- Final choice: **0.01 (1%)** — documented after NN histogram; pairs 19/342 & 20/328 (~1.5%) first group at 2%; 9/337 (~4.4%) at 5%.

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
