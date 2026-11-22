import uuid
import pytest
from app.db.postgres import init_db, async_session_factory, SessionModel, QAEvaluationModel
from app.schemas.contracts import (
    AudioChunkEvent,
    SpeakerRole,
    UtteranceEvent,
    WordTiming,
)
from app.db.storage import get_storage_adapter
from app.workers.stt_worker import STTWorker
from app.workers.scoring_worker import ScoringWorker


@pytest.mark.asyncio
async def test_end_to_end_worker_pipeline():
    await init_db()
    storage = get_storage_adapter()

    # 1. Setup session in DB with unique ID
    session_id = f"test_worker_{uuid.uuid4().hex[:8]}"
    async with async_session_factory() as db:
        session = SessionModel(
            id=session_id,
            candidate_name="Alice Candidate",
            job_title="Full Stack Engineer",
            custom_vocab=["FastAPI", "React"],
            status="IN_CALL",
        )
        db.add(session)
        await db.commit()

    # 2. Test STT Worker
    stt_worker = STTWorker()
    audio_key = f"sessions/{session_id}/audio/chunk_0.wav"
    await storage.put_bytes(audio_key, b"RIFFmockwavdata", "audio/wav")

    event = AudioChunkEvent(
        session_id=session_id,
        chunk_index=0,
        timestamp_ms=0,
        duration_ms=5000,
        storage_key=audio_key,
        is_mixed_audio=True,
    )
    await stt_worker.handle_audio_chunk(event)

    # 3. Test Scoring Worker directly with Q&A utterances
    scoring_worker = ScoringWorker()

    q_utt = UtteranceEvent(
        session_id=session_id,
        utterance_id="utt_q0",
        speaker=SpeakerRole.INTERVIEWER,
        speaker_label="S1",
        text="Could you please introduce yourself and your background?",
        start_ms=0,
        end_ms=3000,
    )
    await scoring_worker.handle_utterance(q_utt)

    a_utt = UtteranceEvent(
        session_id=session_id,
        utterance_id="utt_a0",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="Sure, um, I am a full stack engineer with four years of experience building Python and React applications.",
        start_ms=3500,
        end_ms=9000,
    )
    await scoring_worker.handle_utterance(a_utt)

    # Next question triggers evaluation of previous turn
    q_utt2 = UtteranceEvent(
        session_id=session_id,
        utterance_id="utt_q1",
        speaker=SpeakerRole.INTERVIEWER,
        speaker_label="S1",
        text="How do you handle microservices?",
        start_ms=10000,
        end_ms=12000,
    )
    await scoring_worker.handle_utterance(q_utt2)

    # Verify Q0 baseline was persisted
    async with async_session_factory() as db:
        saved_session = await db.get(SessionModel, session_id)
        assert saved_session is not None
        assert saved_session.baseline_json is not None
        assert "baseline_wpm" in saved_session.baseline_json
