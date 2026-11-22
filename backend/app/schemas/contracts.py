from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class IngestionMode(str, Enum):
    ATTENDEE_BOT = "attendee_bot"
    BROWSER_WEBRTC = "browser_webrtc"
    DEMO_FILE = "demo_file"


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    WAITING_IN_LOBBY = "WAITING_IN_LOBBY"
    IN_CALL = "IN_CALL"
    HOST_DENIED = "HOST_DENIED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SpeakerRole(str, Enum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class VisionAnomalyType(str, Enum):
    GAZE_DEVIATION = "GAZE_DEVIATION"
    HEAD_PITCH_DOWN = "HEAD_PITCH_DOWN"
    MULTI_FACE = "MULTI_FACE"
    DEVICE_DETECTED = "DEVICE_DETECTED"


class RiskBand(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"


class ConfidenceBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AudioChunkEvent(BaseModel):
    idempotency_key: str = ""
    session_id: str
    chunk_index: int = 0
    timestamp_ms: int = 0
    duration_ms: int = 0
    speaker_hint: SpeakerRole = SpeakerRole.UNKNOWN
    storage_key: str
    is_mixed_audio: bool = True

    def model_post_init(self, __context) -> None:
        if not self.idempotency_key:
            self.idempotency_key = f"idem:{self.session_id}:audio:{self.chunk_index}"


class WordTiming(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class UtteranceEvent(BaseModel):
    session_id: str
    utterance_id: str
    speaker: SpeakerRole
    speaker_label: str
    text: str
    start_ms: int
    end_ms: int
    words: List[WordTiming] = Field(default_factory=list)


class VisionFrameEvent(BaseModel):
    idempotency_key: str = ""
    session_id: str
    frame_index: int = 0
    timestamp_ms: int
    ingested_at_ms: int = 0
    storage_key: str
    is_live: bool = False

    def model_post_init(self, __context) -> None:
        if not self.idempotency_key:
            self.idempotency_key = f"idem:{self.session_id}:video:{self.frame_index}"


class VisionAnomalyEvent(BaseModel):
    event_id: str
    session_id: str
    timestamp_ms: int
    anomaly_type: VisionAnomalyType
    duration_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    evidence_frame_key: Optional[str] = None
    vlm_verification_note: Optional[str] = None


class CandidateBaseline(BaseModel):
    session_id: str
    baseline_wpm: float
    mad_wpm: float = 0.0
    baseline_disfluency_rate: float
    baseline_pause_ratio: float
    baseline_ttfs_ms: float = 0.0
    sample_word_count: int = 0
    baseline_gaze_variance: float = 0.0


class HeuristicScores(BaseModel):
    word_count: int
    wpm: float
    wpm_robust_zscore: float
    disfluency_rate: float
    disfluency_drop: float
    ttfs_latency_ms: int = 0
    cadence_uniformity_cv: float = 0.0
    ai_tell_matches: List[str] = Field(default_factory=list)
    first_person_ratio: float = 0.0
    concurrent_vision_flags: int = 0
    deterministic_risk: float = Field(ge=0.0, le=100.0)
    trigger_llm_judge: bool = False
    trigger_reasons: List[str] = Field(default_factory=list)


class LLMJudgeVerdict(BaseModel):
    answer_id: str
    provider_used: str = "groq"
    suspicion_score: float = Field(ge=0.0, le=100.0)
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    reasoning_signals: List[str] = Field(default_factory=list)
    verbatim_evidence_quotes: List[str] = Field(default_factory=list)
    recitation_vs_live_assessment: str = ""
    advisory_follow_up_question: str = ""


class QAPairEvaluation(BaseModel):
    qa_index: int
    question_text: str
    answer_text: str
    question_end_ms: int = 0
    answer_start_ms: int = 0
    answer_end_ms: int = 0
    heuristics: HeuristicScores
    llm_verdict: Optional[LLMJudgeVerdict] = None
    combined_risk_score: float = Field(ge=0.0, le=100.0)


class IntegrityReport(BaseModel):
    session_id: str
    candidate_name: str
    job_title: str
    overall_risk_score: float = Field(ge=0.0, le=100.0)
    risk_band: RiskBand
    advisory_disclaimer: str = (
        "Advisory behavioral inference derived from meeting-plane audio and video signals. "
        "Not operating-system telemetry and not definitive proof of misconduct."
    )
    baseline: Optional[CandidateBaseline] = None
    qa_evaluations: List[QAPairEvaluation] = Field(default_factory=list)
    vision_timeline: List[VisionAnomalyEvent] = Field(default_factory=list)
    generated_at_iso: str = ""
