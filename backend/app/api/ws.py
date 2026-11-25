import asyncio
import base64
import json
import logging
import time
from typing import Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.db.postgres import (
    IntegrityReportModel,
    SessionModel,
    UtteranceModel,
    async_session_factory,
)
from app.db.redis_bus import redis_bus
from app.db.storage import get_storage_adapter
from app.schemas.contracts import AudioChunkEvent, VisionFrameEvent

router = APIRouter(tags=["websockets"])
logger = logging.getLogger("oculus.api.ws")
storage = get_storage_adapter()

MAX_FRAME_BYTES = 2 * 1024 * 1024   # 2 MB JPEG cap
MAX_AUDIO_BYTES = 5 * 1024 * 1024   # 5 MB audio chunk cap


def _detect_audio_format(raw: bytes) -> Tuple[str, str]:
    """Detect actual container format from browser MediaRecorder blobs."""
    if raw.startswith(b"\x1a\x45\xdf\xa3"):
        return ("webm", "audio/webm")
    if raw.startswith(b"OggS"):
        return ("ogg", "audio/ogg")
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return ("wav", "audio/wav")
    return ("webm", "audio/webm")


async def _safe_unsubscribe_ui(session_id: str, queue: asyncio.Queue) -> None:
    unsub = getattr(redis_bus, "unsubscribe_ui", None)
    if callable(unsub):
        try:
            res = unsub(session_id, queue)
            if asyncio.iscoroutine(res):
                await res
        except TypeError:
            try:
                res = unsub(session_id)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.debug("Pub/Sub unsubscribe fallback failed for %s: %s", session_id, exc)
        except Exception as exc:
            logger.debug("Pub/Sub unsubscribe failed for %s: %s", session_id, exc)


@router.websocket("/ws/sessions/{session_id}")
async def session_live_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    queue = await redis_bus.subscribe_ui(session_id)

    try:
        # Initial hydration snapshot from DB so page reloads never lose history
        try:
            async with async_session_factory() as db:
                s = await db.get(SessionModel, session_id)
                if s:
                    rep_stmt = select(IntegrityReportModel).where(
                        IntegrityReportModel.session_id == session_id
                    )
                    rep = (await db.execute(rep_stmt)).scalars().first()

                    utt_stmt = (
                        select(UtteranceModel)
                        .where(UtteranceModel.session_id == session_id)
                        .order_by(UtteranceModel.start_ms)
                    )
                    utterances = (await db.execute(utt_stmt)).scalars().all()

                    await websocket.send_text(
                        json.dumps(
                            {
                                "event_type": "HYDRATE_STATE",
                                "session_id": session_id,
                                "data": {
                                    "status": s.status,
                                    "candidate_name": s.candidate_name,
                                    "job_title": s.job_title,
                                    "baseline": s.baseline_json,
                                    "utterances": [
                                        {
                                            "utterance_id": u.id,
                                            "speaker": u.speaker,
                                            "speaker_label": u.speaker_label,
                                            "text": u.text,
                                            "start_ms": u.start_ms,
                                            "end_ms": u.end_ms,
                                        }
                                        for u in utterances
                                    ],
                                    "report": rep.report_json if rep else None,
                                },
                            },
                            default=str,
                        )
                    )
        except Exception as exc:
            logger.error("Error hydrating WS session %s: %s", session_id, exc)

        # Relay Pub/Sub events with 25s heartbeat
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=25.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event_type": "PING"}))
    except WebSocketDisconnect:
        logger.info("Client disconnected from WS session %s", session_id)
    except Exception as exc:
        logger.error("WS error for session %s: %s", session_id, exc)
    finally:
        await _safe_unsubscribe_ui(session_id, queue)


@router.websocket("/ws/ingest/{session_id}")
async def browser_webrtc_ingest_websocket(websocket: WebSocket, session_id: str):
    """
    Accepts live browser WebRTC tab capture chunks:
    5-second MediaRecorder audio blobs + 1 FPS canvas JPEGs.
    """
    await websocket.accept()
    session_start_ms = int(time.time() * 1000)
    audio_seq = 0
    frame_seq = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            now_ms = int(time.time() * 1000)

            if msg.get("bytes"):
                b = msg["bytes"]
                if len(b) > MAX_AUDIO_BYTES:
                    logger.warning("Dropped oversized audio chunk (%d bytes) for %s", len(b), session_id)
                    continue

                ext, mime = _detect_audio_format(b)
                audio_key = f"sessions/{session_id}/audio/live_{now_ms}_{audio_seq}.{ext}"
                await storage.put_bytes(audio_key, b, mime)

                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=audio_seq,
                    timestamp_ms=max(0, now_ms - session_start_ms),
                    duration_ms=5000,
                    storage_key=audio_key,
                    is_mixed_audio=True,
                )
                await redis_bus.publish_stream("stream:audio", event)
                audio_seq += 1

            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON on ingest WS for %s", session_id)
                    continue

                if not isinstance(data, dict):
                    continue

                kind = data.get("kind")
                if kind == "frame":
                    raw_b64 = str(data.get("b64") or "").split(",")[-1].strip()
                    if not raw_b64:
                        continue
                    try:
                        fbytes = base64.b64decode(raw_b64, validate=False)
                    except Exception:
                        logger.warning("Malformed base64 frame on ingest WS for %s", session_id)
                        continue

                    if not fbytes or len(fbytes) > MAX_FRAME_BYTES:
                        logger.warning("Dropped invalid/oversized frame (%d bytes) for %s", len(fbytes), session_id)
                        continue

                    fkey = f"sessions/{session_id}/frames/live_{now_ms}_{frame_seq}.jpg"
                    await storage.put_bytes(fkey, fbytes, "image/jpeg")

                    raw_ts = data.get("ts_ms")
                    try:
                        ts_ms = int(raw_ts) if raw_ts is not None else max(0, now_ms - session_start_ms)
                    except (TypeError, ValueError):
                        ts_ms = max(0, now_ms - session_start_ms)

                    fevent = VisionFrameEvent(
                        session_id=session_id,
                        frame_index=frame_seq,
                        timestamp_ms=max(0, ts_ms),
                        ingested_at_ms=now_ms,  # Unix epoch ms (matches VisionWorker clock)
                        storage_key=fkey,
                        is_live=True,
                    )
                    await redis_bus.publish_stream("stream:video", fevent)
                    frame_seq += 1
                elif kind == "end":
                    break

    except WebSocketDisconnect:
        logger.info("Browser ingest disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("Browser ingest WS error for %s: %s", session_id, exc)
    finally:
        if audio_seq > 0:
            now_ms = int(time.time() * 1000)
            await redis_bus.publish_stream(
                "stream:audio",
                AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=audio_seq,
                    timestamp_ms=max(0, now_ms - session_start_ms),
                    duration_ms=0,
                    storage_key="",
                    is_mixed_audio=True,
                    is_final=True,
                ),
            )