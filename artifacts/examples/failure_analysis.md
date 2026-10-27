# Held-out failure analysis (RC7 — accepted wrong)

Standalone write-up regenerable via `scripts/generate_examples.py` (rules FSM; no LLM). Separate from held-out **abstain** examples (failure `case_no` must differ from abstain `case_no`).

- **Case no:** 88
- **Fold / model:** 0 / lightgbm__median
- **True intensity:** strong (grade 3)
- **Predicted intensity:** light (grade 1)
- **|ŷ−y|:** 2 (prefer ≥2; fallback ≥1)
- **R_final:** 0.8251741812528557
- **Fold max-prob:** 0.9685566408969644
- **Recommended action:** accept_with_caution
- **Abstained:** False (must be false for this file)
- **Termination reason:** finalize_unresolved_no_new_evidence

## Inputs

```json
{
  "no": 88.0,
  "sigma_theta": 62.4,
  "sigma_c": 235.0,
  "sigma_t": 9.5,
  "scf": 0.27,
  "b1": 24.74,
  "b2": 0.92,
  "wet": 9.0,
  "d": 330.0,
  "rockburst_grade": 3.0
}
```

## Agent outputs / flags

```json
{
  "route_trace": [
    "rules:data_quality",
    "rules:geomechanics_skipped",
    "rules:predict",
    "rules:uncertainty",
    "rules:extra_validation",
    "rules:critic",
    "rules:recheck_physics",
    "rules:critic",
    "rules:finalize"
  ],
  "termination_reason": "finalize_unresolved_no_new_evidence",
  "critic_recheck_cycles": 1,
  "ood_status": "No",
  "data_reliability": null,
  "physics_score": null,
  "r_model": null,
  "prediction_set": [
    0,
    1,
    2
  ],
  "prediction_set_intensities": [
    "none",
    "light",
    "moderate"
  ],
  "model_disagreement": null,
  "critic_endorse": false,
  "critic_rules_fired": [
    "low_physics",
    "physics_vs_model_confidence",
    "ml_vs_criteria",
    "large_prediction_set",
    "validation_objection"
  ],
  "inter_agent_disagreement": {
    "ml_grade": 1,
    "ml_intensity": "light",
    "criteria_consensus_grade": 2,
    "criteria_intensity": "moderate",
    "criteria_grades": {
      "russenes": 1,
      "kidybinski": 3
    },
    "ml_vs_criteria": true,
    "critic_endorse": false,
    "critic_rules_fired": [
      "low_physics",
      "physics_vs_model_confidence",
      "large_prediction_set",
      "ml_vs_criteria",
      "validation_objection"
    ],
    "parties": [
      "ml_vs_criteria",
      "critic_vs_fusion_path",
      "critic_flags_ml_vs_criteria",
      "geomechanics_agent",
      "predictive_agent",
      "uncertainty_agent"
    ],
    "has_disagreement": true,
    "level": "High"
  },
  "checks_skipped": true,
  "decision_override_reason": null,
  "recommended_action": "accept_with_caution",
  "abstained": false,
  "final_system_confidence": 0.8251741812528557
}
```

## Checks skipped

- True

## Checks fired (Critic / route)

```json
{
  "critic_rules_fired": [
    "low_physics",
    "physics_vs_model_confidence",
    "ml_vs_criteria",
    "large_prediction_set",
    "validation_objection"
  ],
  "checks_skipped": true,
  "route_trace": [
    "rules:data_quality",
    "rules:geomechanics_skipped",
    "rules:predict",
    "rules:uncertainty",
    "rules:extra_validation",
    "rules:critic",
    "rules:recheck_physics",
    "rules:critic",
    "rules:finalize"
  ]
}
```

## Why wrong

Held-out accepted prediction with |ŷ−y|=2 (light vs true strong) at R_final=0.825, fold_max_prob=0.969. Action=accept_with_caution (not abstain).

## Which check would have caught it

- monotonic_tendency / criteria consensus: large |ŷ−y| with high R_final should escalate to abstain before accept
- wide conformal edge-of-set check (predicted grade at edge of |set|≥3)
- ml_vs_criteria already fired — escalate_abstain severity from precision

## Decision text

Predicted intensity: light (grade 1). Conformal set: {none, light, moderate} (grades [0, 1, 2]). System confidence R_final=0.825. Recommended action: accept_with_caution. Hazard statement: Moderate rockburst potential; engineering verification recommended. Inter-agent disagreement (High): ML light vs criteria moderate; Critic objections (low_physics, physics_vs_model_confidence, large_prediction_set, ml_vs_criteria, validation_objection). Accept with caution; verify key measurements. (Critic unresolved (endorse=False, recheck=physics); downgraded accept → accept_with_caution)

Protocol: held-out outer test fold; rules FSM; accepted-wrong only (RC7)
