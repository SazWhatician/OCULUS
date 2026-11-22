import pytest
from app.schemas.contracts import AudioChunkEvent, SpeakerRole
from app.adapters.stt.base import STTRouter
from app.adapters.stt.speechmatics import SpeechmaticsAdapter
from app.adapters.stt.groq_whisper import GroqWhisperAdapter
from app.adapters.llm.base import FallbackLLMRouter
from app.adapters.ingestion.attendee import AttendeeBotAdapter


@pytest.mark.asyncio
async def test_speechmatics_adapter_mock_diarization():
    adapter = SpeechmaticsAdapter()
    results = await adapter.transcribe(
        audio_bytes=b"RIFF...",
        session_id="sess_123",
        offset_ms=0,
        custom_vocab=["Kafka", "Redis"],
        diarize=True,
    )
    assert len(results) == 2
    assert results[0].speaker == SpeakerRole.INTERVIEWER
    assert results[1].speaker == SpeakerRole.CANDIDATE
    assert "Kafka" in results[1].text


@pytest.mark.asyncio
async def test_groq_whisper_adapter_mock():
    adapter = GroqWhisperAdapter()
    results = await adapter.transcribe(
        audio_bytes=b"RIFF...",
        session_id="sess_123",
        offset_ms=1000,
        diarize=False,
    )
    assert len(results) == 1
    assert results[0].start_ms == 1000
    assert "microservices" in results[0].text


@pytest.mark.asyncio
async def test_stt_router_routing():
    router = STTRouter()

    # Case 1: Mixed audio routes to Speechmatics
    mixed_chunk = AudioChunkEvent(
        session_id="s1",
        chunk_index=0,
        timestamp_ms=0,
        storage_key="k1",
        is_mixed_audio=True,
    )
    res_mixed = await router.route_and_transcribe(mixed_chunk, b"wav", ["Kafka"])
    assert len(res_mixed) == 2

    # Case 2: Isolated track routes to Groq Whisper
    isolated_chunk = AudioChunkEvent(
        session_id="s1",
        chunk_index=1,
        timestamp_ms=5000,
        storage_key="k2",
        is_mixed_audio=False,
        speaker_hint=SpeakerRole.CANDIDATE,
    )
    res_iso = await router.route_and_transcribe(isolated_chunk, b"wav", ["Kubernetes"])
    assert len(res_iso) == 1


@pytest.mark.asyncio
async def test_fallback_llm_router():
    router = FallbackLLMRouter()
    verdict = await router.evaluate_turn(
        question_text="How do you handle race conditions?",
        answer_text="It is crucial to note that one must delve into composite indexing.",
        job_title="Senior Engineer",
    )
    assert verdict is not None
    assert verdict.suspicion_score > 50.0
    assert len(verdict.advisory_follow_up_question) > 0


def test_attendee_hmac_verification():
    adapter = AttendeeBotAdapter()
    adapter.api_key = "secret_key"
    body = b'{"event": "bot.state_change"}'
    import hmac, hashlib
    valid_sig = hmac.new(b"secret_key", body, hashlib.sha256).hexdigest()

    assert adapter.verify_webhook_signature(valid_sig, body) is True
    assert adapter.verify_webhook_signature("invalid_signature", body) is False
