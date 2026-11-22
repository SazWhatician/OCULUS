import os
import shutil
import asyncio
import logging
from typing import Dict, Any, Optional
from app.schemas.contracts import IngestionMode, AudioChunkEvent, VisionFrameEvent
from app.adapters.ingestion.base import IngestionAdapter
from app.db.storage import get_storage_adapter
from app.db.redis_bus import redis_bus

logger = logging.getLogger("oculus.ingestion.demo")


class DemoFileAdapter(IngestionAdapter):
    """
    $0 Demo Replay Engine: Accepts mock interview recordings (.mp4, .webm, .wav)
    and segments them into audio chunks and 1 FPS video frames feeding Redis Streams.
    """

    @property
    def mode(self) -> IngestionMode:
        return IngestionMode.DEMO_FILE

    def __init__(self):
        self.storage = get_storage_adapter()

    async def start_session(self, session_id: str, file_path: str, **kwargs) -> Dict[str, Any]:
        asyncio.create_task(self._process_file_replay(session_id, file_path))
        return {
            "session_id": session_id,
            "status": "PROCESSING",
            "message": "Demo file processing started in background.",
        }

    async def stop_session(self, session_id: str) -> bool:
        return True

    async def _process_file_replay(self, session_id: str, file_path: str) -> None:
        try:
            logger.info(f"Replaying demo file for session {session_id}: {file_path}")
            has_ffmpeg = shutil.which("ffmpeg") is not None

            # 1. Process Audio
            audio_out = f"{file_path}.wav"
            if has_ffmpeg:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", file_path, "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", audio_out,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            else:
                # If already wav or ffmpeg not in path, use source directly
                audio_out = file_path

            if os.path.exists(audio_out):
                with open(audio_out, "rb") as f:
                    audio_bytes = f.read()
                storage_key = f"sessions/{session_id}/audio/demo_full.wav"
                await self.storage.put_bytes(storage_key, audio_bytes, "audio/wav")

                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=0,
                    timestamp_ms=0,
                    duration_ms=60000,
                    storage_key=storage_key,
                    is_mixed_audio=True,
                )
                await redis_bus.publish_stream("stream:audio", event)

            # 2. Process Video Frames (1 FPS sample)
            if has_ffmpeg and file_path.lower().endswith((".mp4", ".webm", ".mov")):
                frames_dir = f"{file_path}_frames"
                os.makedirs(frames_dir, exist_ok=True)
                proc_v = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", file_path, "-vf", "fps=1,scale=640:360", f"{frames_dir}/frame_%04d.jpg",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc_v.wait()

                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                for idx, fname in enumerate(frame_files):
                    fpath = os.path.join(frames_dir, fname)
                    with open(fpath, "rb") as f:
                        fbytes = f.read()
                    f_key = f"sessions/{session_id}/frames/frame_{idx}.jpg"
                    await self.storage.put_bytes(f_key, fbytes, "image/jpeg")

                    v_event = VisionFrameEvent(
                        session_id=session_id,
                        frame_index=idx,
                        timestamp_ms=idx * 1000,
                        ingested_at_ms=idx * 1000,
                        storage_key=f_key,
                        is_live=False,
                    )
                    await redis_bus.publish_stream("stream:video", v_event)

                shutil.rmtree(frames_dir, ignore_errors=True)

            logger.info(f"Demo file replay completed ingestion for session {session_id}")
        except Exception as e:
            logger.error(f"Error in demo replay for session {session_id}: {e}", exc_info=True)
