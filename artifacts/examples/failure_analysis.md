# Held-out failure analysis (C5)

Standalone write-up regenerable from shipped models via `scripts/generate_examples.py` (rules FSM; no LLM).

- **True intensity:** strong (grade 3)
- **Predicted intensity:** none (grade 0)
- **|ŷ−y|:** 3
- **R_final:** 0.6875
- **Fold max-prob:** 0.6621621621621622

## Diagnosis

Held-out severe misclassification (|ŷ−y|=3) at high confidence; point prediction none vs true strong. FSM action=abstain. Inter-agent: {'ml_grade': 0, 'ml_intensity': 'none', 'criteria_consensus_grade': None, 'criteria_intensity': 'unknown', 'criteria_grades': {}, 'ml_vs_criteria': False, 'critic_endorse': False, 'critic_rules_fired': ['ensemble_disagreement', 'large_prediction_set', 'wide_conformal_edge', 'validation_objection'], 'parties': ['critic_vs_fusion_path'], 'has_disagreement': True}.

## Decision text

Predicted intensity: none (grade 0). Conformal set: {none, light, moderate} (grades [0, 1, 2]). System confidence R_final=0.688. Recommended action: abstain. Inter-agent disagreement: Critic objections (ensemble_disagreement, large_prediction_set, wide_conformal_edge, validation_objection). Inadequate confidence; abstain and request additional data/review. (Critic severe rules ['wide_conformal_edge']; escalated accept → abstain)

Protocol: held-out outer test fold; rules FSM; not hard-coded
