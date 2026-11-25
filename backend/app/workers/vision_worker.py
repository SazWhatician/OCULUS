import asyncio
import hashlib
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.adapters.llm.gemini_judge import GeminiJudgeAdapter
from app.config import settings
from app.db.postgres import VisionEventModel, async_session_factory
from app.db.redis_bus import redis_bus
from app.db.storage import get_storage_adapter
from app.schemas.contracts import (
    VisionAnomalyEvent as ContractVisionAnomalyEvent,
    VisionAnomalyType,
    VisionFrameEvent,
)
from app.services.vision_proctoring import (
    VisionTelemetryFrame,
    get_vision_service,
)
from app.telemetry import metrics

logger = logging.getLogger("oculus.worker.vision")

EPOCH_MS_SANITY_FLOOR = 1_000_000_000_000  # ~2001-09-09 in Unix epoch ms
LIVE_FRAME_MAX_AGE_MS = 10_000
MAX_THUMBNAIL_SESSIONS = 500


class VisionWorker:
    """
    Zero-GPU 3-Tier Vision Proctoring Worker:
      - Tier 0: Fast blur/occlusion check + frame-delta liveness (offloaded off asyncio loop)
      - Tier 1: Cached CPU facial/eye geometry -> VisionProctoringService 10-channel state machine
      - Tier 2: Budget-capped Gemini Flash VLM verification on flagged anomaly keyframes
      - Storage hygiene: Purges every non-anomalous frame immediately in `finally:`
    """

    def __init__(self) -> None:
        self.storage = get_storage_adapter()
        self.gemini_judge = GeminiJudgeAdapter()
        self.vision_service = get_vision_service()

        # Load OpenCV Haar cascades ONCE at startup (never reload XML per frame)
        self._cv_lock = threading.Lock()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # Bounded tiny thumbnails (64x36) per session for lightweight frame-delta liveness
        self._prev_thumbs: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.max_vlm_calls: int = int(getattr(settings, "MAX_VLM_CALLS_PER_SESSION", 12))
        self.vlm_cooldown_sec: float = float(getattr(settings, "VLM_COOLDOWN_SEC", 30.0))
        self._vlm_tracker: "OrderedDict[str, Tuple[int, float]]" = OrderedDict()

    async def aclose(self) -> None:
        await self.gemini_judge.aclose()

    async def handle_video_frame(self, event: VisionFrameEvent) -> None:
        now_ms = int(time.time() * 1000)
        ingested_at_ms = int(getattr(event, "ingested_at_ms", 0) or 0)

        # Epoch-safe backpressure guard: only drop frames if ingested_at_ms is a valid Unix epoch ms
        if event.is_live and ingested_at_ms >= EPOCH_MS_SANITY_FLOOR:
            age_ms = now_ms - ingested_at_ms
            if age_ms > LIVE_FRAME_MAX_AGE_MS:
                logger.debug(
                    "Dropped stale live frame for %s (age=%dms > %dms)",
                    event.session_id,
                    age_ms,
                    LIVE_FRAME_MAX_AGE_MS,
                )
                await self._safe_delete(event.storage_key)
                return
        elif event.is_live and ingested_at_ms > 0:
            logger.warning(
                "Non-epoch ingested_at_ms=%d received for %s; skipping staleness drop.",
                ingested_at_ms,
                event.session_id,
            )

        try:
            jpeg_bytes = await self.storage.get_bytes(event.storage_key)
        except Exception as exc:
            logger.debug("Failed reading frame %s: %s", event.storage_key, exc)
            return

        retained_evidence = False
        try:
            # Tier 0 & Tier 1 CPU computer vision offloaded to worker thread
            analysis = await asyncio.to_thread(
                self._analyze_frame_sync,
                event.session_id,
                jpeg_bytes,
            )
            if analysis is None:
                return

            telemetry_frame = VisionTelemetryFrame(
                session_id=event.session_id,
                timestamp_ms=event.timestamp_ms,
                camera_active=analysis["camera_active"],
                faces_detected=analysis["faces_detected"],
                gaze_direction=analysis["gaze_direction"],
                gaze_offset_x=analysis["gaze_offset_x"],
                gaze_offset_y=analysis["gaze_offset_y"],
                head_pose_yaw=analysis["head_pose_yaw"],
                head_pose_pitch=analysis["head_pose_pitch"],
                head_pose_roll=analysis["head_pose_roll"],
                liveness_score=analysis["liveness_score"],
            )

            # Delegate debouncing, calibration, occlusion, split-angle, peek-loop & liveness
            # to the centralized VisionProctoringService state machine
            service_events = self.vision_service.ingest_frame_all(telemetry_frame)
            if not service_events:
                return

            # Tier 2: Run at most ONE Gemini Flash VLM check per frame when escalated
            vlm_res: Optional[Dict[str, Any]] = None
            needs_vlm = any(ev.requires_vlm_verification for ev in service_events)
            if needs_vlm and self._can_invoke_vlm(event.session_id):
                self._record_vlm_call(event.session_id)
                vlm_res = await self.gemini_judge.verify_frame(
                    jpeg_bytes,
                    session_id=event.session_id,
                )

            retained_evidence = True
            vlm_note = (vlm_res or {}).get("note")
            vlm_phone_detected = bool((vlm_res or {}).get("phone_or_tablet_visible", False))

            for svc_ev in service_events:
                anomaly_type = svc_ev.anomaly_type or VisionAnomalyType.GAZE_DEVIATION
                confidence = svc_ev.confidence
                if vlm_phone_detected and svc_ev.requires_vlm_verification:
                    anomaly_type = VisionAnomalyType.DEVICE_DETECTED
                    confidence = max(confidence, 0.94)

                event_id = self._deterministic_event_id(
                    session_id=event.session_id,
                    timestamp_ms=event.timestamp_ms,
                    frame_index=getattr(event, "frame_index", 0),
                    category=svc_ev.category,
                )

                v_event = ContractVisionAnomalyEvent(
                    event_id=event_id,
                    session_id=event.session_id,
                    timestamp_ms=event.timestamp_ms,
                    anomaly_type=anomaly_type,
                    duration_ms=svc_ev.duration_ms,
                    confidence=confidence,
                    pitch_deg=analysis["head_pose_pitch"],
                    yaw_deg=analysis["head_pose_yaw"],
                    evidence_frame_key=event.storage_key,
                    vlm_verification_note=vlm_note if svc_ev.requires_vlm_verification else None,
                )

                await self._persist_vision_event(v_event)
                await redis_bus.publish_stream("stream:vision_events", v_event)
                await redis_bus.broadcast_ui(
                    session_id=event.session_id,
                    event_type="VISION_ANOMALY",
                    data=v_event.model_dump(mode="json"),
                )
                metrics.inc("vision_anomalies_detected")

        finally:
            if not retained_evidence:
                await self._safe_delete(event.storage_key)
            metrics.inc("video_frames_processed")

    def _analyze_frame_sync(
        self,
        session_id: str,
        jpeg_bytes: bytes,
    ) -> Optional[Dict[str, Any]]:
        """CPU-bound OpenCV decode + geometry extraction (runs in threadpool)."""
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            return None

        h_img, w_img = gray.shape[:2]
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(np.mean(gray))

        # Camera covered / black / severely occluded frame -> feed camera_active=False
        # so VisionProctoringService debounces and flags Camera Occlusion instead of ignoring it
        if lap_var < 10.0 or mean_luma < 8.0:
            return {
                "camera_active": False,
                "faces_detected": 0,
                "gaze_direction": "center",
                "gaze_offset_x": 0.0,
                "gaze_offset_y": 0.0,
                "head_pose_yaw": 0.0,
                "head_pose_pitch": 0.0,
                "head_pose_roll": 0.0,
                "liveness_score": 0.98,
            }

        # Lightweight frame-delta liveness (catches static photo / frozen virtual camera)
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        prev_thumb = self._prev_thumbs.get(session_id)
        if prev_thumb is not None:
            mean_diff = float(np.mean(cv2.absdiff(thumb, prev_thumb)))
            # Alive webcams have subtle compression/micro-motion (>0.35); static images sit near 0.0
            liveness_score = 0.35 if mean_diff < 0.20 else min(0.99, 0.72 + min(mean_diff / 8.0, 0.27))
        else:
            liveness_score = 0.96

        self._prev_thumbs[session_id] = thumb
        self._prev_thumbs.move_to_end(session_id)
        if len(self._prev_thumbs) > MAX_THUMBNAIL_SESSIONS:
            self._prev_thumbs.popitem(last=False)

        with self._cv_lock:
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60),
            )

        num_faces = len(faces)
        gaze_x = 0.0
        gaze_y = 0.0
        yaw_deg = 0.0
        pitch_deg = 0.0
        roll_deg = 0.0
        gaze_direction = "center"

        if num_faces >= 1:
            # Track primary (largest) face bounding box
            (x, y, w, h) = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            cx = (x + w * 0.5) / max(w_img, 1)
            cy = (y + h * 0.5) / max(h_img, 1)

            # Normalized horizontal/vertical displacement from frame center
            dx = (cx - 0.5) * 2.0   # -1..1
            dy = (cy - 0.5) * 2.0   # -1..1

            # Estimate head pose from face center & upper-face eye asymmetry
            yaw_deg = max(-45.0, min(45.0, dx * 38.0))
            pitch_deg = max(-45.0, min(45.0, -dy * 38.0))

            # Eye-ROI refinement for horizontal/vertical iris bias
            roi_gray = gray[y : y + int(h * 0.60), x : x + w]
            with self._cv_lock:
                eyes = self._eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(18, 18),
                )

            if len(eyes) >= 2:
                top_eyes = sorted(eyes, key=lambda e: int(e[2]) * int(e[3]), reverse=True)[:2]
                top_eyes.sort(key=lambda e: e[0])
                (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = top_eyes
                eye_mid_x = ((ex1 + ew1 * 0.5) + (ex2 + ew2 * 0.5)) * 0.5
                eye_mid_y = ((ey1 + eh1 * 0.5) + (ey2 + eh2 * 0.5)) * 0.5
                gaze_x = max(-1.0, min(1.0, ((eye_mid_x / max(w, 1)) - 0.5) * 2.4 + (dx * 0.35)))
                gaze_y = max(-1.0, min(1.0, ((eye_mid_y / max(h * 0.60, 1)) - 0.52) * 1.8 + max(0.0, dy * 0.45)))
                roll_deg = float(np.degrees(np.arctan2((ey2 + eh2 * 0.5) - (ey1 + eh1 * 0.5), max((ex2 - ex1), 1))))
            else:
                gaze_x = max(-1.0, min(1.0, dx * 0.65))
                gaze_y = max(-1.0, min(1.0, dy * 0.70))

            if gaze_y > 0.30 or pitch_deg < -20.0:
                gaze_direction = "down"
            elif gaze_x > 0.34:
                gaze_direction = "right"
            elif gaze_x < -0.34:
                gaze_direction = "left"

        return {
            "camera_active": True,
            "faces_detected": num_faces,
            "gaze_direction": gaze_direction,
            "gaze_offset_x": round(float(gaze_x), 4),
            "gaze_offset_y": round(float(gaze_y), 4),
            "head_pose_yaw": round(float(yaw_deg), 2),
            "head_pose_pitch": round(float(pitch_deg), 2),
            "head_pose_roll": round(float(roll_deg), 2),
            "liveness_score": round(float(liveness_score), 3),
        }

    def _can_invoke_vlm(self, session_id: str) -> bool:
        count, last_ts = self._vlm_tracker.get(session_id, (0, 0.0))
        if count >= self.max_vlm_calls:
            return False
        return (time.time() - last_ts) >= self.vlm_cooldown_sec

    def _record_vlm_call(self, session_id: str) -> None:
        count, _ = self._vlm_tracker.get(session_id, (0, 0.0))
        self._vlm_tracker[session_id] = (count + 1, time.time())
        self._vlm_tracker.move_to_end(session_id)
        if len(self._vlm_tracker) > MAX_THUMBNAIL_SESSIONS:
            self._vlm_tracker.popitem(last=False)

    @staticmethod
    def _deterministic_event_id(
        session_id: str,
        timestamp_ms: int,
        frame_index: int,
        category: str,
    ) -> str:
        raw = f"{session_id}|{timestamp_ms}|{frame_index}|{category}".encode("utf-8")
        return f"v_evt_{hashlib.sha256(raw).hexdigest()[:16]}"

    @staticmethod
    async def _persist_vision_event(v_event: ContractVisionAnomalyEvent) -> None:
        async with async_session_factory() as db:
            existing = await db.get(VisionEventModel, v_event.event_id)
            if existing is not None:
                return
            db.add(
                VisionEventModel(
                    id=v_event.event_id,
                    session_id=v_event.session_id,
                    timestamp_ms=v_event.timestamp_ms,
                    anomaly_type=getattr(v_event.anomaly_type, "value", str(v_event.anomaly_type)),
                    duration_ms=v_event.duration_ms,
                    confidence=v_event.confidence,
                    pitch_deg=v_event.pitch_deg,
                    yaw_deg=v_event.yaw_deg,
                    evidence_frame_key=v_event.evidence_frame_key,
                    vlm_verification_note=v_event.vlm_verification_note,
                )
            )
            await db.commit()

    async def _safe_delete(self, storage_key: str) -> None:
        try:
            await self.storage.delete(storage_key)
        except Exception as exc:
            logger.debug("Storage cleanup failed for %s: %s", storage_key, exc)


async def run_vision_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    worker = VisionWorker()
    consumer_id = f"vision_{socket.gethostname()}_{os.getpid()}"
    logger.info("VisionWorker '%s' started consuming from 'stream:video'", consumer_id)
    try:
        await redis_bus.consume_stream(
            stream="stream:video",
            group="vision_workers",
            consumer=consumer_id,
            model_cls=VisionFrameEvent,
            handler=worker.handle_video_frame,
            stop_event=stop_event,
        )
    finally:
        await worker.aclose()
        logger.info("VisionWorker '%s' shut down cleanly.", consumer_id)