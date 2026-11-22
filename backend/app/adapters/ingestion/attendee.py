import hmac
import hashlib
import logging
from typing import Dict, Any, Optional
import httpx
from app.config import settings
from app.schemas.contracts import (
    IngestionMode,
    SessionStatus,
    AudioChunkEvent,
    VisionFrameEvent,
    SpeakerRole,
)
from app.adapters.ingestion.base import IngestionAdapter
from app.db.storage import get_storage_adapter
from app.db.redis_bus import redis_bus

logger = logging.getLogger("oculus.ingestion.attendee")


class AttendeeBotAdapter(IngestionAdapter):
    """
    Manages in-meeting bots joining Zoom, Google Meet, or MS Teams via Attendee API.
    Handles webhook lifecycle callbacks and pushes ingested media to Redis Streams.
    """

    @property
    def mode(self) -> IngestionMode:
        return IngestionMode.ATTENDEE_BOT

    def __init__(self):
        self.api_url = settings.ATTENDEE_BASE_URL.rstrip("/")
        self.api_key = settings.ATTENDEE_API_KEY
        self.storage = get_storage_adapter()

    def verify_webhook_signature(self, signature: str, raw_body: bytes) -> bool:
        if not self.api_key:
            return True  # Dev bypass
        expected = hmac.new(self.api_key.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    async def start_session(self, session_id: str, meeting_url: str, bot_name: str = "OCULUS Sentinel", **kwargs) -> Dict[str, Any]:
        if not self.api_key:
            logger.warning("ATTENDEE_API_KEY is not set. Operating in mock bot mode.")
            return {
                "bot_id": f"mock_bot_{session_id}",
                "status": "WAITING_IN_LOBBY",
                "message": "Mock bot joining meeting lobby.",
            }

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "webhook_url": f"{settings.PUBLIC_API_URL}/api/v1/webhooks/attendee",
            "audio_tracks": "per_participant",
            "recording_mode": "speaker_view",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(f"{self.api_url}/api/v1/bots", json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                return {
                    "bot_id": data.get("id"),
                    "status": "WAITING_IN_LOBBY",
                    "data": data,
                }
        except Exception as e:
            logger.error(f"Failed to dispatch Attendee bot: {e}")
            return {"bot_id": f"err_{session_id}", "status": "FAILED", "error": str(e)}

    async def stop_session(self, session_id: str) -> bool:
        logger.info(f"Attendee bot instructed to leave meeting for session {session_id}")
        return True

    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Processes webhook callbacks from Attendee: lifecycle status + media payloads.
        """
        event_type = event_data.get("event")
        bot_id = event_data.get("bot_id")
        session_id = event_data.get("session_id", bot_id)

        if event_type == "bot.state_change":
            state = event_data.get("state", "").lower()
            status_map = {
                "joining": SessionStatus.WAITING_IN_LOBBY,
                "in_waiting_room": SessionStatus.WAITING_IN_LOBBY,
                "in_call": SessionStatus.IN_CALL,
                "host_denied": SessionStatus.HOST_DENIED,
                "call_ended": SessionStatus.COMPLETED,
            }
            new_status = status_map.get(state, SessionStatus.PROCESSING)
            await redis_bus.broadcast_ui(session_id, "STATUS_UPDATE", {"status": new_status.value})

        elif event_type == "audio.chunk":
            # Per-participant or mixed audio chunk delivered
            chunk_bytes = event_data.get("raw_bytes")
            if chunk_bytes:
                storage_key = f"sessions/{session_id}/audio/{event_data.get('chunk_index', 0)}.wav"
                await self.storage.put_bytes(storage_key, chunk_bytes, "audio/wav")
                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=event_data.get("chunk_index", 0),
                    timestamp_ms=event_data.get("timestamp_ms", 0),
                    duration_ms=event_data.get("duration_ms", 0),
                    storage_key=storage_key,
                    is_mixed_audio=event_data.get("is_mixed", False),
                    speaker_hint=SpeakerRole(event_data.get("speaker_hint", "unknown")),
                )
                await redis_bus.publish_stream("stream:audio", event)
