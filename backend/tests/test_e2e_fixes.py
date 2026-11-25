import base64
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.adapters.ingestion.attendee import AttendeeBotAdapter
from app.config import Settings
from app.db.postgres import (
    IntegrityReportModel,
    QAEvaluationModel,
    SessionModel,
    UtteranceModel,
    async_session_factory,
    init_db,
)
from app.db.storage import LocalDiskStorageAdapter
from app.main import app
from app.schemas.contracts import (
    AudioChunkEvent,
    SpeakerRole,
    UtteranceEvent,
    WordTiming,
)
from app.workers import scoring_worker as scoring_mod
from app.workers import stt_worker as stt_mod
from app.workers.scoring_worker import ScoringWorker
from app.workers.stt_worker import STTWorker


def test_env_and_settings_load_tuning_flags_without_blank_key_clobber():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_lines = [line.strip() for line in env_path.read_text(encoding="utf-8").splitlines()]
    # Ensure no empty key definitions remain to clobber keys defined earlier in .env
    assert "GROQ_API_KEY=" not in env_lines
    assert "GEMINI_API_KEY=" not in env_lines
    assert "SPEECHMATICS_API_KEY=" not in env_lines

    cfg = Settings(_env_file=str(env_path))
    assert cfg.SCORING_MAX_LLM_TURNS == 6
    assert cfg.PROCTORING_MAX_LLM_TURNS == 5
    assert cfg.LLM_HIGH_RISK_CONSENSUS is False
    assert cfg.GEMINI_TEXT_MODEL == "gemini-3.5-flash-lite"


def test_storage_rejects_path_traversal(tmp_path):
    storage = LocalDiskStorageAdapter(base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        storage._get_path("../escape.txt")


@pytest.mark.asyncio
async def test_attendee_webhook_security_lookup_and_b64_audio():
    await init_db()
    adapter = AttendeeBotAdapter()
    adapter.api_key = ""
    # Fails closed when API key is missing unless insecure dev explicitly enabled
    assert adapter.verify_webhook_signature("", b"{}") is False

    session_id = f"sess_bot_{uuid.uuid4().hex[:8]}"
    bot_id = f"bot_ext_{uuid.uuid4().hex[:8]}"
    async with async_session_factory() as db:
        db.add(
            SessionModel(
                id=session_id,
                candidate_name="Bot Candidate",
                job_title="Staff Engineer",
                ingestion_mode="attendee_bot",
                bot_external_id=bot_id,
                status="WAITING_IN_LOBBY",
            )
        )
        await db.commit()

    # Bot state change looked up purely from bot_id updates DB session status
    await adapter.handle_webhook_event(
        {"event": "bot.state_change", "bot_id": bot_id, "state": "in_call"}
    )
    async with async_session_factory() as db:
        row = await db.get(SessionModel, session_id)
        assert row is not None
        assert row.status == "IN_CALL"

    # Base64 audio payload stores bytes cleanly & guards unknown speaker_hint
    audio_b64 = base64.b64encode(b"RIFFmockbotdata").decode("ascii")
    await adapter.handle_webhook_event(
        {
            "event": "audio.chunk",
            "bot_id": bot_id,
            "chunk_index": 3,
            "raw_bytes_b64": audio_b64,
            "speaker_hint": "unexpected_role_label",
        }
    )
    stored = await adapter.storage.get_bytes(f"sessions/{session_id}/audio/bot_3.wav")
    assert stored == b"RIFFmockbotdata"


@pytest.mark.asyncio
async def test_stt_worker_is_final_flush_and_idempotent_upsert():
    await init_db()
    session_id = f"sess_stt_{uuid.uuid4().hex[:8]}"
    async with async_session_factory() as db:
        db.add(
            SessionModel(
                id=session_id,
                candidate_name="Replay Candidate",
                job_title="Backend Engineer",
                status="PROCESSING",
                speaker_role_map={"S1": "interviewer", "S2": "candidate"},
            )
        )
        await db.commit()

    worker = STTWorker()
    audio_key = f"sessions/{session_id}/audio/demo_full.wav"
    await worker.storage.put_bytes(audio_key, b"RIFFdemodata", "audio/wav")

    await worker.handle_audio_chunk(
        AudioChunkEvent(
            session_id=session_id,
            chunk_index=0,
            timestamp_ms=0,
            duration_ms=5000,
            storage_key=audio_key,
            is_mixed_audio=True,
            is_final=True,
        )
    )

    # Mock Speechmatics emits 2 speakers; is_final=True must flush the trailing 2nd turn too
    assert worker.stitcher.get_open_utterance(session_id) is None
    async with async_session_factory() as db:
        rows = (
            await db.execute(select(UtteranceModel).where(UtteranceModel.session_id == session_id))
        ).scalars().all()
        assert len(rows) == 2

    # Re-persisting the same utterance_id must upsert instead of raising IntegrityError
    first = rows[0]
    await worker._persist_and_publish_utterance(
        UtteranceEvent(
            session_id=session_id,
            utterance_id=first.id,
            speaker=SpeakerRole(first.speaker),
            speaker_label=first.speaker_label,
            text="Updated text on redelivery",
            start_ms=first.start_ms,
            end_ms=first.end_ms,
            words=[],
        )
    )
    async with async_session_factory() as db:
        updated = await db.get(UtteranceModel, first.id)
        assert updated is not None
        assert updated.text == "Updated text on redelivery"


@pytest.mark.asyncio
async def test_complete_session_flushes_pending_turn_and_recomputes_report():
    await init_db()
    session_id = f"sess_comp_{uuid.uuid4().hex[:8]}"
    async with async_session_factory() as db:
        db.add(
            SessionModel(
                id=session_id,
                candidate_name="Flush Candidate",
                job_title="Platform Engineer",
                status="IN_CALL",
                speaker_role_map={"S1": "interviewer", "S2": "candidate"},
            )
        )
        # Seed a stale partial report from earlier in the interview
        db.add(
            IntegrityReportModel(
                session_id=session_id,
                overall_risk_score=0.0,
                risk_band="LOW",
                report_json={"session_id": session_id, "overall_risk_score": 0.0, "qa_evaluations": []},
            )
        )
        await db.commit()

    scoring = ScoringWorker()
    stt = STTWorker()
    prev_scoring = scoring_mod._active_scoring_worker
    prev_stt = stt_mod._active_stt_worker
    scoring_mod._active_scoring_worker = scoring
    stt_mod._active_stt_worker = stt

    try:
        # Question buffered directly into scoring + trailing candidate answer open in stitcher
        await scoring.handle_utterance(
            UtteranceEvent(
                session_id=session_id,
                utterance_id=f"q0_{uuid.uuid4().hex[:6]}",
                speaker=SpeakerRole.INTERVIEWER,
                speaker_label="S1",
                text="Walk me through your Kubernetes production scaling work?",
                start_ms=0,
                end_ms=3000,
            )
        )
        # Leave final candidate turn inside STT stitcher (never closed by next speaker)
        stt.stitcher.process_incoming(
            UtteranceEvent(
                session_id=session_id,
                utterance_id=f"a0_{uuid.uuid4().hex[:6]}",
                speaker=SpeakerRole.CANDIDATE,
                speaker_label="S2",
                text=(
                    "Um, we migrated our stateful workloads to regional GKE node pools with "
                    "horizontal pod autoscaling based on Kafka consumer lag, cutting p99 latency in half."
                ),
                start_ms=4000,
                end_ms=11000,
                words=[
                    WordTiming(word="Um", start_ms=4000, end_ms=4300),
                    WordTiming(word="we", start_ms=4400, end_ms=4600),
                ],
            )
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            comp_res = await client.post(f"/api/v1/sessions/{session_id}/complete")
            assert comp_res.status_code == 200
            comp_data = comp_res.json()
            assert comp_data["status"] == "COMPLETED"
            assert len(comp_data["report"]["qa_evaluations"]) == 1

            # Swap speaker map retroactively and verify persisted utterances remap
            swap_res = await client.patch(
                f"/api/v1/sessions/{session_id}/speakers",
                json={"speaker_role_map": {"S1": "candidate", "S2": "interviewer"}},
            )
            assert swap_res.status_code == 200
            assert swap_res.json()["roles_changed"] is True

            detail_res = await client.get(f"/api/v1/sessions/{session_id}")
            assert detail_res.status_code == 200
            detail = detail_res.json()
            assert len(detail["utterances"]) == 1
            assert detail["utterances"][0]["speaker"] == "interviewer"
    finally:
        scoring_mod._active_scoring_worker = prev_scoring
        stt_mod._active_stt_worker = prev_stt
        await scoring.llm_router.aclose()
