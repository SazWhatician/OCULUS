import base64
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional
import httpx
from sqlalchemy import select
from app.config import settings
from app.schemas.contracts import (
    IngestionMode,
    SessionStatus,
    AudioChunkEvent,
    VisionFrameEvent,
    SpeakerRole,
)
from app.adapters.ingestion.base import IngestionAdapter
from app.db.postgres import async_session_factory, SessionModel
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
            return bool(
                settings.APP_ENV == "development"
                and getattr(settings, "ATTENDEE_WEBHOOK_ALLOW_INSECURE_DEV", False)
            )
        if not signature:
            return False
        provided = signature.strip()
        if provided.startswith("sha256="):
            provided = provided[7:].strip()
        expected = hmac.new(self.api_key.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, provided)

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

    async def stop_session(self, session_id: str, bot_id: Optional[str] = None) -> bool:
        resolved_bot_id = bot_id
        if not resolved_bot_id:
            async with async_session_factory() as db:
                row = await db.get(SessionModel, session_id)
                if row:
                    resolved_bot_id = row.bot_external_id

        if not self.api_key or not resolved_bot_id or str(resolved_bot_id).startswith(("mock_", "err_")):
            logger.info("Attendee bot mock/local leave complete for session %s", session_id)
            return True

        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(f"{self.api_url}/api/v1/bots/{resolved_bot_id}/leave", headers=headers)
                res.raise_for_status()
            logger.info("Attendee bot %s left meeting for session %s", resolved_bot_id, session_id)
            return True
        except Exception as exc:
            logger.warning("Failed instructing Attendee bot %s to leave (%s): %s", resolved_bot_id, session_id, exc)
            return False

    async def _resolve_session_id(self, event_data: Dict[str, Any]) -> Optional[str]:
        explicit_sid = event_data.get("session_id")
        if explicit_sid:
            return str(explicit_sid)
        bot_id = event_data.get("bot_id")
        if not bot_id:
            return None
        async with async_session_factory() as db:
            res = await db.execute(select(SessionModel).where(SessionModel.bot_external_id == str(bot_id)))
            row = res.scalars().first()
            return row.id if row else None

    @staticmethod
    def _safe_speaker_role(value: Any) -> SpeakerRole:
        if isinstance(value, SpeakerRole):
            return value
        if not value:
            return SpeakerRole.UNKNOWN
        try:
            return SpeakerRole(str(value).strip().lower())
        except ValueError:
            return SpeakerRole.UNKNOWN

    async def _decode_payload_bytes(self, event_data: Dict[str, Any], b64_keys: tuple[str, ...], url_keys: tuple[str, ...]) -> Optional[bytes]:
        for k in b64_keys:
            raw_val = event_data.get(k)
            if isinstance(raw_val, str) and raw_val.strip():
                try:
                    return base64.b64decode(raw_val.split(",")[-1].strip(), validate=False)
                except Exception:
                    return None
        raw_bytes = event_data.get("raw_bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return bytes(raw_bytes)
        if isinstance(raw_bytes, str) and raw_bytes:
            try:
                return base64.b64decode(raw_bytes.split(",")[-1].strip(), validate=True)
            except Exception:
                return raw_bytes.encode("latin-1", errors="ignore")
        for uk in url_keys:
            media_url = event_data.get(uk)
            if isinstance(media_url, str) and media_url.startswith(("http://", "https://")):
                headers = {"Authorization": f"Token {self.api_key}"} if self.api_key else {}
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        res = await client.get(media_url, headers=headers)
                        res.raise_for_status()
                        return res.content
                except Exception as exc:
                    logger.warning("Failed fetching Attendee webhook media from %s: %s", media_url, exc)
                    return None
        return None

    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Processes webhook callbacks from Attendee: lifecycle status + media payloads.
        """
        event_type = event_data.get("event")
        session_id = await self._resolve_session_id(event_data)
        if not session_id:
            logger.warning("Ignoring Attendee webhook %r: unable to resolve session_id or bot_id", event_type)
            return

        if event_type == "bot.state_change":
            state = str(event_data.get("state") or "").lower()
            status_map = {
                "joining": SessionStatus.WAITING_IN_LOBBY,
                "in_waiting_room": SessionStatus.WAITING_IN_LOBBY,
                "in_call": SessionStatus.IN_CALL,
                "host_denied": SessionStatus.HOST_DENIED,
                "call_ended": SessionStatus.COMPLETED,
            }
            new_status = status_map.get(state, SessionStatus.PROCESSING)
            async with async_session_factory() as db:
                row = await db.get(SessionModel, session_id)
                if row:
                    row.status = new_status.value
                    await db.commit()

            await redis_bus.broadcast_ui(session_id, "STATUS_UPDATE", {"status": new_status.value})
            if new_status == SessionStatus.COMPLETED:
                await redis_bus.publish_stream(
                    "stream:audio",
                    AudioChunkEvent(
                        session_id=session_id,
                        chunk_index=999999,
                        timestamp_ms=int(event_data.get("timestamp_ms", 0) or 0),
                        duration_ms=0,
                        storage_key="",
                        is_mixed_audio=True,
                        is_final=True,
                    ),
                )

        elif event_type == "audio.chunk":
            chunk_bytes = await self._decode_payload_bytes(
                event_data,
                b64_keys=("raw_bytes_b64", "audio_b64", "data_b64"),
                url_keys=("audio_url", "url"),
            )
            if chunk_bytes:
                chunk_idx = int(event_data.get("chunk_index", 0) or 0)
                storage_key = f"sessions/{session_id}/audio/bot_{chunk_idx}.wav"
                await self.storage.put_bytes(storage_key, chunk_bytes, "audio/wav")
                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=chunk_idx,
                    timestamp_ms=int(event_data.get("timestamp_ms", 0) or 0),
                    duration_ms=int(event_data.get("duration_ms", 5000) or 5000),
                    storage_key=storage_key,
                    is_mixed_audio=bool(event_data.get("is_mixed", False)),
                    speaker_hint=self._safe_speaker_role(event_data.get("speaker_hint")),
                    is_final=bool(event_data.get("is_final", False)),
                )
                await redis_bus.publish_stream("stream:audio", event)

        elif event_type in ("video.frame", "frame"):
            frame_bytes = await self._decode_payload_bytes(
                event_data,
                b64_keys=("raw_bytes_b64", "frame_b64", "image_b64", "data_b64"),
                url_keys=("frame_url", "image_url", "url"),
            )
            if frame_bytes:
                frame_idx = int(event_data.get("frame_index", 0) or 0)
                fkey = f"sessions/{session_id}/frames/bot_{frame_idx}.jpg"
                await self.storage.put_bytes(fkey, frame_bytes, "image/jpeg")
                ts_ms = int(event_data.get("timestamp_ms", frame_idx * 1000) or 0)
                v_event = VisionFrameEvent(
                    session_id=session_id,
                    frame_index=frame_idx,
                    timestamp_ms=ts_ms,
                    ingested_at_ms=ts_ms,
                    storage_key=fkey,
                    is_live=True,
                )
                await redis_bus.publish_stream("stream:video", v_event)
