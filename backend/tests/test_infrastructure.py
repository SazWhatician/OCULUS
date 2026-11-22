import pytest
import time
from app.db.storage import LocalDiskStorageAdapter
from app.db.redis_bus import RedisBus
from app.schemas.contracts import AudioChunkEvent
from app.services.resilience import CircuitBreaker, CircuitState, QuotaGuard


@pytest.mark.asyncio
async def test_local_storage_adapter(tmp_path):
    storage = LocalDiskStorageAdapter(base_dir=str(tmp_path))
    key = "test/audio.wav"
    data = b"RIFF1234WAVEfmt"

    saved_key = await storage.put_bytes(key, data, content_type="audio/wav")
    assert saved_key == key

    retrieved = await storage.get_bytes(key)
    assert retrieved == data

    url = await storage.get_presigned_url(key)
    assert "test/audio.wav" in url

    deleted = await storage.delete(key)
    assert deleted is True


@pytest.mark.asyncio
async def test_redis_bus_idempotency_and_streams():
    bus = RedisBus()
    # Mock offline mode
    key = "idem:test_sess:1"
    assert await bus.check_and_set_idempotency(key) is True
    assert await bus.check_and_set_idempotency(key) is False  # Duplicate caught!

    event = AudioChunkEvent(session_id="s1", chunk_index=0, storage_key="chunk0.wav")
    msg_id = await bus.publish_stream("stream:audio", event)
    assert msg_id is not None


def test_circuit_breaker_transitions():
    cb = CircuitBreaker("test_api", failure_threshold=2, cooldown_seconds=0.1)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.CLOSED

    # 1 failure -> still closed
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 2 failures -> trips to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # Wait for cooldown
    time.sleep(0.12)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success in probe recovers to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_quota_guard_rate_limiting():
    guard = QuotaGuard("test_guard", max_tokens=2, refill_period_sec=1.0)
    assert await guard.acquire(1) is True
    assert await guard.acquire(1) is True
    assert await guard.acquire(1) is False  # Exhausted!
