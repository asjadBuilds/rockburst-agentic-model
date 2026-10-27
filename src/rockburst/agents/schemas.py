"""Pydantic schemas for structured agent I/O."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DataQualityOutput(BaseModel):
    data_reliability: float = Field(ge=0, le=1)
    ood_flag: bool
    missing_features: list[str] = []
    near_duplicates: list[int] = []
    conflicting_near_dup_labels: bool = False
    train_near_dup_grades: list[int] = []
    ood_scores: dict[str, float] = Field(default_factory=dict)
    ood_combination: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = []


class PhysicsOutput(BaseModel):
    physics_score: float = Field(ge=0, le=1)
    exact_checks: dict[str, Any] = {}
    empirical_checks: dict[str, Any] = {}
    issues: list[str] = []


class PredictiveOutput(BaseModel):
    predicted_grade: int
    probabilities: list[float]
    r_model: float = Field(ge=0, le=1)
    observed_mask: dict[str, bool] = {}
    model_name: str = "baseline"


class UncertaintyOutput(BaseModel):
    uncertainty_score: float
    r_uncertainty: float
    prediction_set: list[int]
    model_disagreement: float
    prediction_stability: str
    stability_score: float
    entropy: float


class CriticOutput(BaseModel):
    challenges: list[str] = []
    contradictions: list[str] = []
    request_recheck: Literal["none", "physics", "data", "model"] = "none"
    endorse: bool = True
    rules_fired: list[str] = []
    escalate_abstain: bool = False
    objection_records: list[dict[str, Any]] = Field(default_factory=list)
    disagreement_level: Literal["Low", "Moderate", "High"] = "Low"
    disagreeing_parties: list[str] = Field(default_factory=list)


class AssessmentOutput(BaseModel):
    predicted_grade: int | None
    prediction_probability: float | None
    prediction_set: list[int] = []
    physics_consistency: float | None = None
    input_reliability: float | None = None
    ood_status: str = "unknown"
    prediction_stability: str = "unknown"
    agent_disagreement: str = "unknown"
    final_system_confidence: float | None = None
    decision: str
    recommended_action: str
    probabilities: list[float] = []
    route_trace: list[str] = []
    abstained: bool = False
