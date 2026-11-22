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


class STTWorker:
    def __init__(self):
        self.router = STTRouter()
        self.stitcher = UtteranceStitcher(max_gap_ms=settings.STITCH_GAP_MAX_MS)
        self.storage = get_storage_adapter()

    async def handle_audio_chunk(self, event: AudioChunkEvent) -> None:
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

        metrics.inc("audio_chunks_processed")

    async def _persist_and_publish_utterance(self, utt: UtteranceEvent) -> None:
        # Write to Postgres
        async with async_session_factory() as session:
            model = UtteranceModel(
                id=utt.utterance_id,
                session_id=utt.session_id,
                speaker=utt.speaker.value,
                speaker_label=utt.speaker_label,
                text=utt.text,
                start_ms=utt.start_ms,
                end_ms=utt.end_ms,
                words_json=[w.model_dump() for w in utt.words],
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


async def run_stt_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    worker = STTWorker()
    logger.info("STTWorker started consuming from 'stream:audio'")
    await redis_bus.consume_stream(
        stream="stream:audio",
        group="stt_workers",
        consumer="stt_1",
        model_cls=AudioChunkEvent,
        handler=worker.handle_audio_chunk,
        stop_event=stop_event,
    )
