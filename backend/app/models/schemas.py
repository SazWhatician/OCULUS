from datetime import datetime
from typing import Any, Literal, List, Dict, Optional
from pydantic import BaseModel, Field

class TimingMetadata(BaseModel):
    pre_answer_silence_ms: Optional[int] = None
    answer_duration_ms: Optional[int] = None
    word_count: Optional[int] = None
    wpm: Optional[float] = None
    wpm_stdev: Optional[float] = None
    pause_count: Optional[int] = None
    longest_pause_ms: Optional[int] = None
    partial_count: Optional[int] = None
    partial_shrinks: Optional[int] = None
    longest_burst_ms: Optional[int] = None

class ProctoringFinding(BaseModel):
    name: str = "finding"
    evidence_quote: str
    explanation: str

class AnswerProctoring(BaseModel):
    question_index: int
    question_preview: str
    score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "medium", "high"]
    ai_reliance_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Estimated candidate AI reliance percentage for this turn")
    speech_naturalness_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Speech naturalness score")
    findings: List[ProctoringFinding] = Field(default_factory=list)
    heuristic_features: Dict[str, Any] = Field(default_factory=dict)

class ProctoringReport(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall session risk score 0 to 100")
    ai_reliance_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall estimated candidate AI reliance percentage")
    speech_naturalness_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Overall speech naturalness score")
    risk_band: Literal["low", "medium", "high"]
    share_high_risk: float
    risk_factors_breakdown: Dict[str, float] = Field(default_factory=dict, description="Micro-breakdown of risk vectors")
    vision_summary: Dict[str, Any] = Field(default_factory=dict, description="Aggregated video proctoring metrics")
    vision_events: List[Dict[str, Any]] = Field(default_factory=list, description="Visual anomaly timeline events")
    pattern_findings: List[str] = Field(default_factory=list)
    recommendation: str
    caveat: str
    per_answer: List[AnswerProctoring] = Field(default_factory=list)
    baseline_summary: Dict[str, Any] = Field(default_factory=dict)
    model_version: str
    weights_version: str = "default-v1"
    status: Literal["ok", "degraded", "failed"]
    delivery_features_available: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    elapsed_ms: int = 0
