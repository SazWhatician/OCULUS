import base64
import hmac
import hashlib
import logging
import uuid
from typing import Any, Dict, Optional, Tuple
import httpx
from sqlalchemy import select
from app.config import settings
from app.schemas.contracts import (
    IngestionMode,
    SessionStatus,
    AudioChunkEvent,
    VisionFrameEvent,
    SpeakerRole,
    UtteranceEvent,
)
from app.adapters.ingestion.base import IngestionAdapter
from app.db.postgres import async_session_factory, SessionModel, UtteranceModel
from app.db.storage import get_storage_adapter
from app.db.redis_bus import redis_bus

logger = logging.getLogger("oculus.ingestion.attendee")

# Shared per-session participant registry keyed by (session_id -> participant_uuid -> info)
_PARTICIPANT_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {}
_BOT_SYNC_DONE: set[str] = set()


def _infer_participant_role(
    participant_name: str,
    is_host: Optional[bool],
    candidate_name: str = "",
    interviewer_name: str = "",
) -> Tuple[SpeakerRole, bool]:
    """Determines whether a Zoom/Meet participant is Bot, Interviewer, or Candidate."""
    pname = (participant_name or "").strip().lower()
    if pname and ("oculus" in pname or "notetaker" in pname or "sentinel" in pname):
        return (SpeakerRole.UNKNOWN, True)

    cname = (candidate_name or "").strip().lower()
    iname = (interviewer_name or "").strip().lower()

    if cname and pname:
        c_tokens = [t for t in cname.split() if len(t) >= 2]
        if cname in pname or pname in cname or (c_tokens and c_tokens[0] in pname):
            return (SpeakerRole.CANDIDATE, False)

    if iname and pname:
        i_tokens = [t for t in iname.split() if len(t) >= 2]
        if iname in pname or pname in iname or (i_tokens and i_tokens[0] in pname):
            return (SpeakerRole.INTERVIEWER, False)

    if is_host is True:
        return (SpeakerRole.INTERVIEWER, False)
    if is_host is False:
        return (SpeakerRole.CANDIDATE, False)
    return (SpeakerRole.UNKNOWN, False)


async def register_meeting_participant(
    session_id: str,
    participant_uuid: str,
    participant_name: str = "",
    is_host: Optional[bool] = None,
) -> Dict[str, Any]:
    sess_map = _PARTICIPANT_MAP.setdefault(session_id, {})
    prev = sess_map.get(participant_uuid) or {}
    merged_name = (participant_name or prev.get("name") or "").strip()
    merged_host = is_host if is_host is not None else prev.get("is_host")

    cname, iname = "", ""
    async with async_session_factory() as db:
        srow = await db.get(SessionModel, session_id)
        if srow:
            cname = srow.candidate_name or ""
            iname = srow.interviewer_name or ""

    role, is_bot = _infer_participant_role(merged_name, merged_host, cname, iname)
    # Fallback when 2 humans are present and only one side's role is resolved
    if not is_bot and role == SpeakerRole.UNKNOWN:
        existing_roles = {v["role"] for k, v in sess_map.items() if k != participant_uuid and not v.get("is_bot")}
        if SpeakerRole.INTERVIEWER in existing_roles and SpeakerRole.CANDIDATE not in existing_roles:
            role = SpeakerRole.CANDIDATE
        elif SpeakerRole.CANDIDATE in existing_roles and SpeakerRole.INTERVIEWER not in existing_roles:
            role = SpeakerRole.INTERVIEWER

    entry = {"name": merged_name, "is_host": merged_host, "role": role, "is_bot": is_bot}
    sess_map[participant_uuid] = entry
    return entry


async def resolve_meeting_participant(
    session_id: str,
    participant_uuid: Optional[str],
    participant_name: str = "",
    is_host: Optional[bool] = None,
) -> Tuple[SpeakerRole, str, bool]:
    """Resolves (SpeakerRole, speaker_name, is_bot) using cached webhooks or Attendee REST API."""
    if not participant_uuid:
        return (SpeakerRole.UNKNOWN, participant_name, False)

    sess_map = _PARTICIPANT_MAP.setdefault(session_id, {})
    if participant_uuid not in sess_map and session_id not in _BOT_SYNC_DONE:
        _BOT_SYNC_DONE.add(session_id)
        await sync_participants_from_attendee(session_id)

    if participant_uuid not in sess_map or (participant_name and not sess_map[participant_uuid].get("name")):
        entry = await register_meeting_participant(session_id, participant_uuid, participant_name, is_host)
    else:
        entry = sess_map[participant_uuid]

    return (entry["role"], entry.get("name") or participant_uuid, bool(entry.get("is_bot")))


async def sync_participants_from_attendee(session_id: str) -> None:
    if not settings.ATTENDEE_API_KEY:
        return
    bot_id = None
    async with async_session_factory() as db:
        srow = await db.get(SessionModel, session_id)
        if srow:
            bot_id = srow.bot_external_id
    if not bot_id or str(bot_id).startswith(("mock_", "err_")):
        return

    base = settings.ATTENDEE_BASE_URL.rstrip("/")
    headers = {"Authorization": f"Token {settings.ATTENDEE_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(f"{base}/api/v1/bots/{bot_id}/participant_events", headers=headers)
            if res.status_code >= 400:
                return
            data = res.json()
            for ev in data.get("results") or []:
                puuid = str(ev.get("participant_uuid") or "")
                if puuid:
                    await register_meeting_participant(
                        session_id=session_id,
                        participant_uuid=puuid,
                        participant_name=str(ev.get("participant_name") or ""),
                        is_host=ev.get("participant_is_host"),
                    )
    except Exception as exc:
        logger.debug("Participant API sync skipped for %s: %s", session_id, exc)


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
        self.webhook_secret = getattr(settings, "ATTENDEE_WEBHOOK_SECRET", "")
        self.storage = get_storage_adapter()

    def verify_webhook_signature(self, signature: str, raw_body: bytes) -> bool:
        secret = self.webhook_secret or self.api_key
        if not secret:
            return bool(
                settings.APP_ENV == "development"
                and getattr(settings, "ATTENDEE_WEBHOOK_ALLOW_INSECURE_DEV", False)
            )
        if not signature:
            return False
        provided = signature.strip()
        if provided.startswith("sha256="):
            provided = provided[7:].strip()
        mac = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256)
        expected_hex = mac.hexdigest()
        expected_b64 = base64.b64encode(mac.digest()).decode("ascii")
        return hmac.compare_digest(expected_hex, provided) or hmac.compare_digest(expected_b64, provided)

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
        public_api = settings.PUBLIC_API_URL.strip().rstrip("/")
        ws_api = public_api.replace("https://", "wss://").replace("http://", "ws://")

        payload: Dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "metadata": {"session_id": session_id},
        }
        triggers = (
            ["bot.state_change", "participant_events.join_leave"]
            if ws_api.startswith("wss://")
            else ["bot.state_change", "participant_events.join_leave", "transcript.update"]
        )
        if public_api.startswith(("https://", "http://")) and "localhost" not in public_api and "127.0.0.1" not in public_api:
            payload["webhooks"] = [
                {
                    "url": f"{public_api}/api/v1/webhooks/attendee",
                    "triggers": triggers,
                }
            ]
        if ws_api.startswith("wss://"):
            payload["websocket_settings"] = {
                "per_participant_audio": {
                    "url": f"{ws_api}/ws/attendee/{session_id}",
                    "sample_rate": 16000,
                },
                "per_participant_video": {
                    "url": f"{ws_api}/ws/attendee-video/{session_id}",
                    "webcam_resolution": "360p",
                },
            }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(f"{self.api_url}/api/v1/bots", json=payload, headers=headers)
                if res.status_code >= 400:
                    err_body = res.text[:500]
                    logger.error("Attendee API HTTP %s error dispatching bot: %s", res.status_code, err_body)
                    return {"bot_id": f"err_{session_id}", "status": "FAILED", "error": err_body}
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
        meta = event_data.get("bot_metadata") or event_data.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("session_id"):
            return str(meta["session_id"])
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
        Processes webhook callbacks from Attendee: lifecycle status + media/transcript payloads.
        """
        event_type = event_data.get("trigger") or event_data.get("event")
        session_id = await self._resolve_session_id(event_data)
        if not session_id:
            logger.warning("Ignoring Attendee webhook %r: unable to resolve session_id or bot_id", event_type)
            return

        data_obj = event_data.get("data") if isinstance(event_data.get("data"), dict) else {}

        if event_type == "bot.state_change":
            state = str(
                data_obj.get("new_state")
                or data_obj.get("state")
                or event_data.get("state")
                or ""
            ).lower()
            status_map = {
                "ready": SessionStatus.WAITING_IN_LOBBY,
                "joining": SessionStatus.WAITING_IN_LOBBY,
                "waiting_room": SessionStatus.WAITING_IN_LOBBY,
                "in_waiting_room": SessionStatus.WAITING_IN_LOBBY,
                "joined_not_recording": SessionStatus.IN_CALL,
                "joined_recording": SessionStatus.IN_CALL,
                "in_call": SessionStatus.IN_CALL,
                "leaving": SessionStatus.PROCESSING,
                "post_processing": SessionStatus.PROCESSING,
                "host_denied": SessionStatus.HOST_DENIED,
                "bot_kicked": SessionStatus.HOST_DENIED,
                "fatal_error": SessionStatus.FAILED,
                "call_ended": SessionStatus.COMPLETED,
                "ended": SessionStatus.COMPLETED,
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

        elif event_type == "participant_events.join_leave":
            puuid = str(data_obj.get("participant_uuid") or "")
            if puuid:
                await register_meeting_participant(
                    session_id=session_id,
                    participant_uuid=puuid,
                    participant_name=str(
                        data_obj.get("participant_name")
                        or data_obj.get("full_name")
                        or data_obj.get("name")
                        or ""
                    ),
                    is_host=data_obj.get("participant_is_host"),
                )

        elif event_type == "transcript.update":
            # Skip duplicate fallback insertion if STTWorker is already handling live WS audio chunks
            speaker_name = str(data_obj.get("speaker_name") or "").strip()
            transcription = data_obj.get("transcription") if isinstance(data_obj.get("transcription"), dict) else data_obj
            text = str((transcription or {}).get("transcript") or data_obj.get("text") or "").strip()
            if not text:
                return
            ts_ms = int(data_obj.get("timestamp_ms", 0) or 0)
            dur_ms = int(data_obj.get("duration_ms", 2000) or 2000)
            speaker_label = "S1" if speaker_name.lower().startswith("interviewer") else "S2"
            role = SpeakerRole.INTERVIEWER if speaker_label == "S1" else SpeakerRole.CANDIDATE
            utt = UtteranceEvent(
                session_id=session_id,
                utterance_id=f"att_{session_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                speaker=role,
                speaker_label=speaker_label,
                text=f"{speaker_name}: {text}" if speaker_name else text,
                start_ms=ts_ms,
                end_ms=ts_ms + max(dur_ms, 500),
                words=[],
            )
            async with async_session_factory() as db:
                if await db.get(UtteranceModel, utt.utterance_id) is None:
                    db.add(
                        UtteranceModel(
                            id=utt.utterance_id,
                            session_id=session_id,
                            speaker=utt.speaker.value,
                            speaker_label=utt.speaker_label,
                            text=utt.text,
                            start_ms=utt.start_ms,
                            end_ms=utt.end_ms,
                            words_json=[],
                        )
                    )
                    await db.commit()
            await redis_bus.publish_stream("stream:utterances", utt)
            await redis_bus.broadcast_ui(session_id, "UTTERANCE", utt.model_dump(mode="json"))

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
