import asyncio
import base64
import io
import json
import logging
import time
import wave
from typing import Any, Dict, Tuple

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.adapters.ingestion.attendee import resolve_meeting_participant
from app.db.postgres import (
    IntegrityReportModel,
    SessionModel,
    UtteranceModel,
    async_session_factory,
)
from app.db.redis_bus import redis_bus
from app.db.storage import get_storage_adapter
from app.schemas.contracts import AudioChunkEvent, SpeakerRole, VisionFrameEvent

router = APIRouter(tags=["websockets"])
logger = logging.getLogger("oculus.api.ws")
storage = get_storage_adapter()

MAX_FRAME_BYTES = 2 * 1024 * 1024   # 2 MB JPEG cap
MAX_AUDIO_BYTES = 5 * 1024 * 1024   # 5 MB audio chunk cap


def _measure_wav_duration_ms(raw: bytes, default_ms: int = 5000) -> int:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate() or 16000
            frames = wf.getnframes()
            if frames > 0:
                return max(500, int((frames / float(rate)) * 1000))
    except Exception:
        pass
    return default_ms


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
        except (WebSocketDisconnect, RuntimeError):
            return
        except Exception as exc:
            logger.error("Error hydrating WS session %s: %s", session_id, exc)

        # Relay Pub/Sub events with 25s heartbeat
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=25.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event_type": "PING"}))
    except (WebSocketDisconnect, RuntimeError):
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
                duration_ms = _measure_wav_duration_ms(b) if ext == "wav" else 5000
                chunk_start_ms = max(0, (now_ms - session_start_ms) - duration_ms)
                audio_key = f"sessions/{session_id}/audio/live_{now_ms}_{audio_seq}.{ext}"
                await storage.put_bytes(audio_key, b, mime)

                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=audio_seq,
                    timestamp_ms=chunk_start_ms,
                    duration_ms=duration_ms,
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


def _encode_pcm16_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _pcm16_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    arr = np.frombuffer(pcm_bytes[: (len(pcm_bytes) // 2) * 2], dtype=np.int16)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))


async def _publish_attendee_video_frame(
    session_id: str,
    data_obj: Dict[str, Any],
    now_ms: int,
    session_start_ms: int,
    frame_seq: int,
) -> bool:
    puuid = str(data_obj.get("participant_uuid") or "")
    sname = str(data_obj.get("speaker_name") or data_obj.get("participant_name") or "")
    role, _, is_bot = await resolve_meeting_participant(session_id, puuid, sname)
    if is_bot or role == SpeakerRole.INTERVIEWER:
        return False

    raw_b64 = (
        data_obj.get("frame")
        or data_obj.get("frame_b64")
        or data_obj.get("image")
        or data_obj.get("chunk")
        or ""
    )
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return False
    try:
        fbytes = base64.b64decode(raw_b64.split(",")[-1].strip())
    except Exception:
        return False
    if not fbytes or len(fbytes) > MAX_FRAME_BYTES:
        return False

    fkey = f"sessions/{session_id}/frames/bot_{now_ms}_{frame_seq}.jpg"
    await storage.put_bytes(fkey, fbytes, "image/jpeg")
    await redis_bus.publish_stream(
        "stream:video",
        VisionFrameEvent(
            session_id=session_id,
            frame_index=frame_seq,
            timestamp_ms=max(0, now_ms - session_start_ms),
            ingested_at_ms=now_ms,
            storage_key=fkey,
            is_live=True,
        ),
    )
    return True


@router.websocket("/ws/attendee/{session_id}")
async def attendee_audio_ws(websocket: WebSocket, session_id: str):
    """
    Receives real-time per-participant OR mixed 16-bit mono PCM audio frames from Attendee Bot
    (`per_participant_audio` / `audio`), gates dead silence, segments at natural pauses / 5s cap,
    and publishes isolated speaker-attributed RIFF/WAVE segments into `stream:audio`.
    """
    await websocket.accept()
    session_start_ms = int(time.time() * 1000)
    buffers: Dict[str, Dict[str, Any]] = {}
    audio_seq = 0
    frame_seq = 0
    last_video_ts = 0

    async def _flush_track(track_key: str, is_final: bool = False) -> None:
        nonlocal audio_seq
        st = buffers.get(track_key)
        if not st:
            return
        buf: bytearray = st["buf"]
        sr: int = st["sample_rate"]
        min_bytes = 3200 if is_final else sr * 2 * 3  # 100ms tail or 3s pause split
        if len(buf) < min_bytes:
            if is_final:
                buf.clear()
            return
        raw_pcm = bytes(buf)
        buf.clear()

        # Skip pure room/digital silence tracks so silent Zoom mics don't trigger STT hallucinations
        if _pcm16_rms(raw_pcm) < 35.0 and not is_final:
            st["start_ms"] = max(0, int(time.time() * 1000) - session_start_ms)
            return

        role: SpeakerRole = st.get("role") or SpeakerRole.UNKNOWN
        is_mixed = track_key == "__mixed__" or role == SpeakerRole.UNKNOWN
        dur_ms = max(100, int((len(raw_pcm) / float(sr * 2)) * 1000))
        wav_bytes = _encode_pcm16_wav(raw_pcm, sample_rate=sr)
        now_ms = int(time.time() * 1000)
        audio_key = f"sessions/{session_id}/audio/bot_{track_key}_{now_ms}_{audio_seq}.wav"
        await storage.put_bytes(audio_key, wav_bytes, "audio/wav")
        await redis_bus.publish_stream(
            "stream:audio",
            AudioChunkEvent(
                session_id=session_id,
                chunk_index=audio_seq,
                timestamp_ms=st["start_ms"],
                duration_ms=dur_ms,
                storage_key=audio_key,
                is_mixed_audio=is_mixed,
                speaker_hint=role,
                is_final=is_final,
            ),
        )
        audio_seq += 1
        st["start_ms"] = max(0, now_ms - session_start_ms)

    async def _handle_video_data(data_obj: Dict[str, Any], now_ms: int) -> None:
        nonlocal frame_seq, last_video_ts
        if (now_ms - last_video_ts) < 950:
            return
        if await _publish_attendee_video_frame(session_id, data_obj, now_ms, session_start_ms, frame_seq):
            last_video_ts = now_ms
            frame_seq += 1

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            now_ms = int(time.time() * 1000)
            if msg.get("bytes"):
                st = buffers.setdefault(
                    "__mixed__",
                    {"buf": bytearray(), "start_ms": max(0, now_ms - session_start_ms), "sample_rate": 16000, "role": SpeakerRole.UNKNOWN},
                )
                if not st["buf"]:
                    st["start_ms"] = max(0, now_ms - session_start_ms)
                st["buf"].extend(msg["bytes"])
                if len(st["buf"]) >= st["sample_rate"] * 2 * 5:
                    await _flush_track("__mixed__", is_final=False)
            elif msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                trigger = str(payload.get("trigger") or "").lower()
                data_obj = payload.get("data") if isinstance(payload.get("data"), dict) else {}

                if "video" in trigger or data_obj.get("frame") or data_obj.get("frame_b64"):
                    await _handle_video_data(data_obj, now_ms)
                    continue

                chunk_b64 = data_obj.get("chunk")
                if not isinstance(chunk_b64, str) or not chunk_b64:
                    continue

                puuid = str(data_obj.get("participant_uuid") or "")
                sname = str(data_obj.get("speaker_name") or data_obj.get("participant_name") or "")
                track_key = puuid if puuid else "__mixed__"

                if track_key not in buffers:
                    role = SpeakerRole.UNKNOWN
                    if puuid:
                        role, _, is_bot = await resolve_meeting_participant(session_id, puuid, sname)
                        if is_bot:
                            continue
                    buffers[track_key] = {
                        "buf": bytearray(),
                        "start_ms": max(0, now_ms - session_start_ms),
                        "sample_rate": 16000,
                        "role": role,
                    }

                st = buffers[track_key]
                if data_obj.get("sample_rate"):
                    try:
                        st["sample_rate"] = max(8000, min(48000, int(data_obj["sample_rate"])))
                    except (TypeError, ValueError):
                        pass

                try:
                    raw_pcm = base64.b64decode(chunk_b64)
                except Exception:
                    continue
                if not raw_pcm:
                    continue

                if not st["buf"]:
                    st["start_ms"] = max(0, now_ms - session_start_ms)
                st["buf"].extend(raw_pcm)

                sr = st["sample_rate"]
                # Split cleanly on pause valley after >=3s OR hard cap at 5.5s
                tail_bytes = sr * 2 // 3  # ~330ms
                is_pause_tail = len(st["buf"]) >= sr * 2 * 3 and _pcm16_rms(bytes(st["buf"][-tail_bytes:])) < 80.0
                if len(st["buf"]) >= int(sr * 2 * 5.5) or is_pause_tail:
                    await _flush_track(track_key, is_final=False)
    except WebSocketDisconnect:
        logger.info("Attendee WS disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("Attendee WS error for %s: %s", session_id, exc)
    finally:
        for k in list(buffers.keys()):
            await _flush_track(k, is_final=True)


@router.websocket("/ws/attendee-video/{session_id}")
async def attendee_video_ws(websocket: WebSocket, session_id: str):
    """
    Receives real-time per-participant Zoom/Meet webcam JPEG frames from Attendee Bot
    (`per_participant_video`), filters strictly to the Candidate participant,
    throttles to 1 FPS, and publishes VisionFrameEvents into `stream:video`.
    """
    await websocket.accept()
    session_start_ms = int(time.time() * 1000)
    frame_seq = 0
    last_frame_ms = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            now_ms = int(time.time() * 1000)
            if (now_ms - last_frame_ms) < 950:
                continue

            text_raw = msg.get("text")
            if not text_raw:
                continue
            try:
                payload = json.loads(text_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            data_obj = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            if await _publish_attendee_video_frame(session_id, data_obj, now_ms, session_start_ms, frame_seq):
                last_frame_ms = now_ms
                frame_seq += 1
    except WebSocketDisconnect:
        logger.info("Attendee video WS disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("Attendee video WS error for %s: %s", session_id, exc)