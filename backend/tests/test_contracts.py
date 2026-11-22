import pytest
from app.config import settings
from app.schemas.contracts import (
    AudioChunkEvent,
    SpeakerRole,
    WordTiming,
    UtteranceEvent,
    VisionFrameEvent,
    VisionAnomalyEvent,
    VisionAnomalyType,
    CandidateBaseline,
    HeuristicScores,
    LLMJudgeVerdict,
    ConfidenceBand,
    QAPairEvaluation,
    IntegrityReport,
    RiskBand,
)


def test_zero_keys_config_boot():
    assert settings.DEPLOY_MODE in ["single_node", "distributed"]
    assert settings.STT_PROVIDER in ["hybrid", "speechmatics", "groq"]
    assert isinstance(settings.SPEECHMATICS_API_KEY, str)
    assert isinstance(settings.GROQ_API_KEY, str)


def test_audio_chunk_idempotency_key_generation():
    chunk = AudioChunkEvent(session_id="sess_test", chunk_index=4, storage_key="audio/chunk_4.wav")
    assert chunk.idempotency_key == "idem:sess_test:audio:4"


def test_utterance_serialization():
    words = [
        WordTiming(word="hello", start_ms=0, end_ms=500, confidence=0.98),
        WordTiming(word="world", start_ms=510, end_ms=900, confidence=0.95),
    ]
    utt = UtteranceEvent(
        session_id="sess_123",
        utterance_id="utt_1",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="hello world",
        start_ms=0,
        end_ms=900,
        words=words,
    )
    json_data = utt.model_dump_json()
    reconstructed = UtteranceEvent.model_validate_json(json_data)
    assert reconstructed.utterance_id == "utt_1"
    assert len(reconstructed.words) == 2
    assert reconstructed.words[0].word == "hello"


def test_score_clamping_and_validation():
    # Negative suspicion score must raise ValidationError
    with pytest.raises(Exception):
        LLMJudgeVerdict(
            answer_id="ans_1",
            suspicion_score=-5.0,
            confidence_band=ConfidenceBand.LOW,
        )

    # Score > 100 must raise ValidationError
    with pytest.raises(Exception):
        IntegrityReport(
            session_id="sess_1",
            candidate_name="Alice",
            job_title="Engineer",
            overall_risk_score=150.0,
            risk_band=RiskBand.ELEVATED,
        )


def test_integrity_report_full_roundtrip():
    report = IntegrityReport(
        session_id="sess_abc",
        candidate_name="Bob",
        job_title="Backend Developer",
        overall_risk_score=28.5,
        risk_band=RiskBand.LOW,
        baseline=CandidateBaseline(
            session_id="sess_abc",
            baseline_wpm=135.0,
            mad_wpm=12.0,
            baseline_disfluency_rate=4.2,
            baseline_pause_ratio=0.15,
        ),
        qa_evaluations=[],
        vision_timeline=[],
    )
    raw_json = report.model_dump_json()
    loaded = IntegrityReport.model_validate_json(raw_json)
    assert loaded.session_id == "sess_abc"
    assert loaded.risk_band == RiskBand.LOW
    assert loaded.baseline.baseline_wpm == 135.0
