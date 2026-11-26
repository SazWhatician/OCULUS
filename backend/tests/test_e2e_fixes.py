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
        {"trigger": "bot.state_change", "bot_id": bot_id, "data": {"new_state": "joined_recording"}}
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


def test_wav_duration_measurement_and_cusum_and_stitcher_continuation():
    import io
    import wave
    from app.api.ws import _measure_wav_duration_ms
    from app.services.stitcher import UtteranceStitcher
    from app.workers.scoring_worker import _SessionCusumTracker

    # 1. Exact 2.0s 16kHz mono PCM16 WAV length measured cleanly
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 32000)
    assert _measure_wav_duration_ms(buf.getvalue()) == 2000

    # 2. Stitcher merges a 3.6s mid-clause thinking pause ("because")
    stitcher = UtteranceStitcher(max_gap_ms=2500)
    u1 = UtteranceEvent(
        session_id="s_cont",
        utterance_id="u1",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="We moved the consumer offsets to Redis because",
        start_ms=0,
        end_ms=3000,
    )
    u2 = UtteranceEvent(
        session_id="s_cont",
        utterance_id="u2",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="rebalancing partitions took too long during deploys.",
        start_ms=6600,
        end_ms=10000,
    )
    assert stitcher.process_incoming(u1) == []
    assert stitcher.process_incoming(u2) == []
    flushed = stitcher.flush_session("s_cont")
    assert flushed is not None
    assert "because rebalancing" in flushed.text

    # 3. CUSUM tracker triggers after clean baseline shifts into sustained synthetic delivery
    cusum = _SessionCusumTracker()
    alert0, _, _ = cusum.update(0, 40, 14.0, 0.2, 0.0, 2.5, 0.0, 0, 0)
    assert alert0 is False
    alert1, _, _ = cusum.update(1, 55, 62.0, 2.8, 3.2, 0.0, 2200.0, 1, 2)
    alert2, score2, onset = cusum.update(2, 55, 64.0, 3.0, 3.5, 0.0, 2400.0, 1, 3)
    assert alert1 or alert2
    assert score2 >= 2.6
    assert onset in (1, 2)


@pytest.mark.asyncio
async def test_stt_diarizer_centroid_swap_updates_open_buffer_and_vectors():
    import numpy as np
    from app.workers.stt_worker import STTWorker

    worker = STTWorker()
    sid = "sess_swap_centroids"
    state = worker.diarizer.get_state(sid)
    v_int = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_cand = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    state.centroids[SpeakerRole.INTERVIEWER] = v_int
    state.centroids[SpeakerRole.CANDIDATE] = v_cand
    state.sample_counts[SpeakerRole.INTERVIEWER] = 3
    state.sample_counts[SpeakerRole.CANDIDATE] = 5

    worker.stitcher.process_incoming(
        UtteranceEvent(
            session_id=sid,
            utterance_id="u_open",
            speaker=SpeakerRole.INTERVIEWER,
            speaker_label="S1",
            text="Still speaking here",
            start_ms=0,
            end_ms=2000,
        )
    )

    await worker.swap_session_roles(sid, {"S1": "candidate", "S2": "interviewer"})
    assert np.allclose(state.centroids[SpeakerRole.INTERVIEWER], v_cand)
    assert np.allclose(state.centroids[SpeakerRole.CANDIDATE], v_int)
    assert state.sample_counts[SpeakerRole.INTERVIEWER] == 5
    open_utt = worker.stitcher.get_open_utterance(sid)
    assert open_utt is not None
    assert open_utt.speaker == SpeakerRole.CANDIDATE


def test_groq_whisper_multipart_uses_async_byte_stream():
    import httpx
    from app.adapters.stt.groq_whisper import GroqWhisperAdapter, GROQ_TRANSCRIBE_URL

    adapter = GroqWhisperAdapter()
    client = adapter._get_client()
    multipart_files = [
        ("file", ("audio.wav", b"RIFFtest", "audio/wav")),
        ("model", (None, adapter.model)),
        ("response_format", (None, "verbose_json")),
        ("timestamp_granularities[]", (None, "word")),
        ("timestamp_granularities[]", (None, "segment")),
    ]
    req = client.build_request("POST", GROQ_TRANSCRIBE_URL, files=multipart_files)
    assert isinstance(req.stream, httpx.AsyncByteStream)


@pytest.mark.asyncio
async def test_attendee_participant_resolution_separates_host_candidate_and_bot():
    from app.adapters.ingestion.attendee import register_meeting_participant, resolve_meeting_participant

    await init_db()
    sid = f"sess_part_{uuid.uuid4().hex[:8]}"
    async with async_session_factory() as db:
        db.add(
            SessionModel(
                id=sid,
                candidate_name="Alex Torres",
                interviewer_name="Saswat Mohanty",
                job_title="Senior Engineer",
                ingestion_mode="attendee_bot",
                status="IN_CALL",
            )
        )
        await db.commit()

    await register_meeting_participant(sid, "u_host", "Saswat Mohanty", True)
    await register_meeting_participant(sid, "u_cand", "Alex Torres", False)
    await register_meeting_participant(sid, "u_bot", "OCULUS Sentinel", False)

    role_i, _, bot_i = await resolve_meeting_participant(sid, "u_host")
    role_c, _, bot_c = await resolve_meeting_participant(sid, "u_cand")
    _, _, bot_b = await resolve_meeting_participant(sid, "u_bot")

    assert role_i == SpeakerRole.INTERVIEWER and not bot_i
    assert role_c == SpeakerRole.CANDIDATE and not bot_c
    assert bot_b is True


def test_vision_solve_pnp_pose_recovers_frontal_and_rotated_angles():
    import math
    import cv2
    import numpy as np
    from app.workers.vision_worker import VISION_TELEMETRY_INTERVAL_MS, YUNET_MODEL_POINTS_3D, VisionWorker

    assert VISION_TELEMETRY_INTERVAL_MS >= 500
    w_img, h_img = 640, 360
    focal = float(max(w_img, h_img))
    cam_matrix = np.array(
        [[focal, 0.0, w_img / 2.0], [0.0, focal, h_img / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)

    pts_frontal, _ = cv2.projectPoints(
        YUNET_MODEL_POINTS_3D, np.zeros((3, 1), dtype=np.float64), tvec, cam_matrix, np.zeros((4, 1))
    )
    yaw_f, pitch_f, roll_f = VisionWorker._solve_pnp_pose(
        YUNET_MODEL_POINTS_3D, pts_frontal.reshape(5, 2), w_img, h_img
    )
    assert all(math.isfinite(v) for v in (yaw_f, pitch_f, roll_f))
    assert abs(pitch_f) < 5.0 and abs(yaw_f) < 5.0 and abs(roll_f) < 5.0

    rvec_yaw = np.array([[0.0], [np.deg2rad(25.0)], [0.0]], dtype=np.float64)
    pts_yaw, _ = cv2.projectPoints(
        YUNET_MODEL_POINTS_3D, rvec_yaw, tvec, cam_matrix, np.zeros((4, 1))
    )
    yaw_y, pitch_y, roll_y = VisionWorker._solve_pnp_pose(
        YUNET_MODEL_POINTS_3D, pts_yaw.reshape(5, 2), w_img, h_img
    )
    assert all(math.isfinite(v) for v in (yaw_y, pitch_y, roll_y))
    assert abs(yaw_y) > 15.0 and abs(pitch_y) < 5.0


@pytest.mark.asyncio
async def test_vision_handle_video_frame_broadcasts_throttled_telemetry_and_fallbacks():
    import json
    import cv2
    import numpy as np
    from app.db.redis_bus import redis_bus
    from app.schemas.contracts import VisionFrameEvent
    from app.workers.vision_worker import VisionWorker

    worker = VisionWorker()
    sid = f"sess_vis_{uuid.uuid4().hex[:8]}"
    q = await redis_bus.subscribe_ui(sid)

    try:
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        for x in range(0, 640, 24):
            cv2.line(frame, (x, 0), (x, 360), (220, 220, 220), 2)
        for y in range(0, 360, 24):
            cv2.line(frame, (0, y), (640, y), (40, 40, 40), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Fallback verification when YuNet detector is absent
        saved_yunet = worker._yunet_detector
        worker._yunet_detector = None
        assert worker._extract_yunet_geometry(frame, gray, 640, 360) is None
        iso = worker._extract_opencv_isophote_geometry(gray, 640, 360)
        assert iso["engine"] == "opencv_isophote"
        assert iso["faces_detected"] == 0
        worker._yunet_detector = saved_yunet

        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        jpeg_bytes = buf.tobytes()

        k1 = f"sessions/{sid}/frames/test_0.jpg"
        k2 = f"sessions/{sid}/frames/test_1.jpg"
        await worker.storage.put_bytes(k1, jpeg_bytes, "image/jpeg")
        await worker.storage.put_bytes(k2, jpeg_bytes, "image/jpeg")

        await worker.handle_video_frame(
            VisionFrameEvent(
                session_id=sid,
                frame_index=0,
                timestamp_ms=100,
                ingested_at_ms=0,
                storage_key=k1,
                is_live=True,
            )
        )
        assert q.qsize() == 1
        envelope = json.loads(await q.get())
        assert envelope["event_type"] == "VISION_TELEMETRY"
        data = envelope["data"]
        assert data["engine"] in {"mediapipe_tasks", "mediapipe_facemesh", "opencv_yunet", "opencv_isophote"}
        assert data["camera_active"] is True
        assert data["faces_detected"] == 0
        assert "gaze_direction" in data
        assert "head_pose_yaw" in data and "head_pose_pitch" in data

        # Immediate second frame within throttle window should not re-emit telemetry
        await worker.handle_video_frame(
            VisionFrameEvent(
                session_id=sid,
                frame_index=1,
                timestamp_ms=250,
                ingested_at_ms=0,
                storage_key=k2,
                is_live=True,
            )
        )
        assert q.qsize() == 0
    finally:
        await worker.aclose()

