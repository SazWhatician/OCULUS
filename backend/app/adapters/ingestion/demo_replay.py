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
            used_ffmpeg_audio = False
            if has_ffmpeg:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", file_path, "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", audio_out,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                used_ffmpeg_audio = os.path.exists(audio_out)

            if not used_ffmpeg_audio:
                audio_out = file_path

            if os.path.exists(audio_out):
                audio_bytes = await asyncio.to_thread(lambda p=audio_out: open(p, "rb").read())
                ext = os.path.splitext(audio_out)[1].lower() or ".wav"
                mime = "audio/wav" if ext == ".wav" else "audio/webm"
                storage_key = f"sessions/{session_id}/audio/demo_full{ext}"
                await self.storage.put_bytes(storage_key, audio_bytes, mime)

                duration_ms = 60000
                if ext == ".wav":
                    try:
                        import wave
                        with wave.open(audio_out, "rb") as wf:
                            rate = wf.getframerate() or 16000
                            duration_ms = max(1000, int((wf.getnframes() / rate) * 1000))
                    except Exception:
                        pass

                event = AudioChunkEvent(
                    session_id=session_id,
                    chunk_index=0,
                    timestamp_ms=0,
                    duration_ms=duration_ms,
                    storage_key=storage_key,
                    is_mixed_audio=True,
                    is_final=True,
                )
                await redis_bus.publish_stream("stream:audio", event)
                if used_ffmpeg_audio and os.path.exists(audio_out):
                    os.remove(audio_out)

            # 2. Process Video Frames (1 FPS sample via ffmpeg or OpenCV fallback)
            if file_path.lower().endswith((".mp4", ".webm", ".mov", ".avi")):
                frames_dir = f"{file_path}_frames"
                os.makedirs(frames_dir, exist_ok=True)
                frame_bytes_list: list[tuple[int, bytes]] = []

                if has_ffmpeg:
                    proc_v = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-y", "-i", file_path, "-vf", "fps=1,scale=640:360", f"{frames_dir}/frame_%04d.jpg",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc_v.wait()
                    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
                    for idx, fname in enumerate(frame_files):
                        fpath = os.path.join(frames_dir, fname)
                        fbytes = await asyncio.to_thread(lambda p=fpath: open(p, "rb").read())
                        frame_bytes_list.append((idx, fbytes))
                else:
                    def _extract_cv2_frames(path: str) -> list[tuple[int, bytes]]:
                        try:
                            import cv2
                        except ImportError:
                            return []
                        cap = cv2.VideoCapture(path)
                        if not cap.isOpened():
                            return []
                        extracted: list[tuple[int, bytes]] = []
                        try:
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            step = max(1, int(round(fps)))
                            frame_num = 0
                            sec_idx = 0
                            while sec_idx < 300:
                                ok, frame = cap.read()
                                if not ok:
                                    break
                                if frame_num % step == 0:
                                    resized = cv2.resize(frame, (640, 360))
                                    enc_ok, buf = cv2.imencode(".jpg", resized)
                                    if enc_ok:
                                        extracted.append((sec_idx, buf.tobytes()))
                                    sec_idx += 1
                                frame_num += 1
                        finally:
                            cap.release()
                        return extracted

                    frame_bytes_list = await asyncio.to_thread(_extract_cv2_frames, file_path)

                for idx, fbytes in frame_bytes_list:
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
