import asyncio
import logging
from typing import Optional
from app.config import settings
from app.schemas.contracts import AudioChunkEvent, UtteranceEvent, SessionStatus
from app.db.storage import get_storage_adapter
from app.db.redis_bus import redis_bus
from app.adapters.stt.base import STTRouter
from app.services.stitcher import UtteranceStitcher
from app.db.postgres import async_session_factory, UtteranceModel, SessionModel
from app.telemetry import metrics, log_structured

logger = logging.getLogger("oculus.worker.stt")

_active_stt_worker: Optional["STTWorker"] = None


def get_active_stt_worker() -> Optional["STTWorker"]:
    return _active_stt_worker


class STTWorker:
    def __init__(self):
        self.router = STTRouter()
        self.stitcher = UtteranceStitcher(max_gap_ms=settings.STITCH_GAP_MAX_MS)
        self.storage = get_storage_adapter()

    async def handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        if not event.storage_key:
            if event.is_final:
                await self.flush_session(event.session_id)
            return

        try:
            audio_bytes = await self.storage.get_bytes(event.storage_key)
        except Exception as e:
            logger.error(f"Failed retrieving audio for key {event.storage_key}: {e}")
            return

        # Query session custom vocab and speaker map if present
        custom_vocab = []
        speaker_map = None
        async with async_session_factory() as session:
            db_session = await session.get(SessionModel, event.session_id)
            if db_session:
                custom_vocab = db_session.custom_vocab or []
                speaker_map = db_session.speaker_role_map or {}

        # 1. Transcribe
        raw_utterances = await self.router.route_and_transcribe(
            event=event,
            audio_bytes=audio_bytes,
            custom_vocab=custom_vocab,
        )

        # 2. Stitch and flush
        for utt in raw_utterances:
            flushed_list = self.stitcher.process_incoming(utt, speaker_role_map=speaker_map)
            for flushed in flushed_list:
                await self._persist_and_publish_utterance(flushed)

        if event.is_final:
            await self.flush_session(event.session_id)

        metrics.inc("audio_chunks_processed")

    async def flush_session(self, session_id: str):
        flushed = self.stitcher.flush_session(session_id)
        if flushed is not None:
            await self._persist_and_publish_utterance(flushed)
        return flushed

    async def flush_all_pending(self) -> None:
        for sid in list(self.stitcher._buffers.keys()):
            await self.flush_session(sid)

    async def _persist_and_publish_utterance(self, utt: UtteranceEvent) -> None:
        # Upsert into DB to stay idempotent on stream retries / redelivery
        async with async_session_factory() as session:
            existing = await session.get(UtteranceModel, utt.utterance_id)
            words_payload = [w.model_dump() for w in utt.words]
            speaker_val = utt.speaker.value if hasattr(utt.speaker, "value") else str(utt.speaker)
            if existing is not None:
                existing.speaker = speaker_val
                existing.speaker_label = utt.speaker_label
                existing.text = utt.text
                existing.start_ms = utt.start_ms
                existing.end_ms = utt.end_ms
                existing.words_json = words_payload
            else:
                model = UtteranceModel(
                    id=utt.utterance_id,
                    session_id=utt.session_id,
                    speaker=speaker_val,
                    speaker_label=utt.speaker_label,
                    text=utt.text,
                    start_ms=utt.start_ms,
                    end_ms=utt.end_ms,
                    words_json=words_payload,
                )
                session.add(model)
            await session.commit()

        # Emit to stream:utterances for scoring worker
        await redis_bus.publish_stream("stream:utterances", utt)

        # Broadcast live to frontend WebSocket
        await redis_bus.broadcast_ui(
            session_id=utt.session_id,
            event_type="UTTERANCE",
            data=utt.model_dump(),
        )
        metrics.inc("utterances_stitched")


    async def aclose(self) -> None:
        await self.flush_all_pending()
        aclose_fn = getattr(self.router, "aclose", None)
        if callable(aclose_fn):
            await aclose_fn()


async def run_stt_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    global _active_stt_worker
    worker = STTWorker()
    _active_stt_worker = worker
    logger.info("STTWorker started consuming from 'stream:audio'")
    try:
        await redis_bus.consume_stream(
            stream="stream:audio",
            group="stt_workers",
            consumer="stt_1",
            model_cls=AudioChunkEvent,
            handler=worker.handle_audio_chunk,
            stop_event=stop_event,
        )
    finally:
        try:
            await worker.aclose()
        finally:
            if _active_stt_worker is worker:
                _active_stt_worker = None
