import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.db.redis_bus import redis_bus
from app.db.postgres import async_session_factory, SessionModel, UtteranceModel, QAEvaluationModel, IntegrityReportModel
from app.db.storage import get_storage_adapter
from app.schemas.contracts import AudioChunkEvent, VisionFrameEvent

router = APIRouter(tags=["websockets"])
logger = logging.getLogger("oculus.api.ws")
storage = get_storage_adapter()


@router.websocket("/ws/sessions/{session_id}")
async def session_live_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    queue = await redis_bus.subscribe_ui(session_id)

    # Initial hydration snapshot from DB so page reloads never lose history
    try:
        async with async_session_factory() as db:
            s = await db.get(SessionModel, session_id)
            if s:
                rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
                rep = (await db.execute(rep_stmt)).scalars().first()

                utt_stmt = select(UtteranceModel).where(UtteranceModel.session_id == session_id).order_by(UtteranceModel.start_ms)
                utterances = (await db.execute(utt_stmt)).scalars().all()

                await websocket.send_text(
                    json.dumps({
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
                    })
                )
    except Exception as e:
        logger.error(f"Error hydrating WS session {session_id}: {e}")

    # Relay Pub/Sub events with 25s heartbeat
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=25.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                # Heartbeat ping
                await websocket.send_text(json.dumps({"event_type": "PING"}))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from WS session {session_id}")
    except Exception as e:
        logger.error(f"WS error for session {session_id}: {e}")


@router.websocket("/ws/ingest/{session_id}")
async def browser_webrtc_ingest_websocket(websocket: WebSocket, session_id: str):
    """
    Accepts live browser WebRTC tab capture chunks:
    5-second WebM audio blobs + 1 FPS canvas JPEGs.
    """
    await websocket.accept()
    audio_seq = 0
    frame_seq = 0

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                # Raw audio chunk
                b = msg["bytes"]
                audio_key = f"sessions/{session_id}/audio/live_{audio_seq}.wav"
                await storage.put_bytes(audio_key, b, "audio/wav")
                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=audio_seq,
                    timestamp_ms=audio_seq * 5000,
                    duration_ms=5000,
                    storage_key=audio_key,
                    is_mixed_audio=True,
                )
                await redis_bus.publish_stream("stream:audio", event)
                audio_seq += 1

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                kind = data.get("kind")
                if kind == "frame":
                    import base64
                    raw_b64 = data.get("b64", "").split(",")[-1]
                    fbytes = base64.b64decode(raw_b64)
                    fkey = f"sessions/{session_id}/frames/live_{frame_seq}.jpg"
                    await storage.put_bytes(fkey, fbytes, "image/jpeg")

                    fevent = VisionFrameEvent(
                        session_id=session_id,
                        frame_index=frame_seq,
                        timestamp_ms=data.get("ts_ms", frame_seq * 1000),
                        ingested_at_ms=int(asyncio.get_event_loop().time() * 1000),
                        storage_key=fkey,
                        is_live=True,
                    )
                    await redis_bus.publish_stream("stream:video", fevent)
                    frame_seq += 1

    except WebSocketDisconnect:
        logger.info(f"Browser ingest disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Browser ingest WS error: {e}")
