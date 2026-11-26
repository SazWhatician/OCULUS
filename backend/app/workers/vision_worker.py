import asyncio
import hashlib
import logging
import math
import os
import socket
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import httpx
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

# ---------------------------------------------------------------------------
# Forensic Vision Constants & Canonical 3D Anthropometric Model
# ---------------------------------------------------------------------------
EPOCH_MS_SANITY_FLOOR = 1_000_000_000_000  # ~2001-09-09 in Unix epoch ms
LIVE_FRAME_MAX_AGE_MS = 10_000
MAX_TRACKED_SESSIONS = 500
MAX_DECODE_WIDTH = 640                     # Bounds CPU cost even on 4K webcam streams

# Canonical 3D face points (mm) in OpenCV camera-aligned coordinates for straight-on face:
# +X = image right, +Y = image down, +Z = away from camera (nose tip at origin)
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),        # Nose tip (MP 1)
        (0.0, 63.6, 12.5),      # Chin (MP 152)
        (-43.3, -32.7, 26.0),   # Image-left eye outer corner (MP 33)
        (43.3, -32.7, 26.0),    # Image-right eye outer corner (MP 263)
        (-28.9, 28.9, 24.1),    # Image-left mouth corner (MP 61)
        (28.9, 28.9, 24.1),     # Image-right mouth corner (MP 291)
    ],
    dtype=np.float64,
)
PNP_LANDMARK_IDS = (1, 152, 33, 263, 61, 291)

# 5-point canonical model matching OpenCV YuNet landmark indices:
# [viewer-left eye center, viewer-right eye center, nose tip, viewer-left mouth, viewer-right mouth]
YUNET_MODEL_POINTS_3D = np.array(
    [
        (-31.0, -32.7, 22.0),   # Viewer-left eye center
        (31.0, -32.7, 22.0),    # Viewer-right eye center
        (0.0, 0.0, 0.0),        # Nose tip
        (-28.9, 28.9, 24.1),    # Viewer-left mouth corner
        (28.9, 28.9, 24.1),     # Viewer-right mouth corner
    ],
    dtype=np.float64,
)

MEDIAPIPE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
OPENCV_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
VISION_TELEMETRY_INTERVAL_MS = 900

# MediaPipe 478-point eye & iris indices
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_EYE_EAR_IDS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_IDS = (362, 385, 387, 263, 373, 380)
LEFT_IRIS_IDS = (468, 469, 470, 471, 472)
RIGHT_IRIS_IDS = (473, 474, 475, 476, 477)

# Precomputed 64x64 Hann window & radial bandpass mask for 2D FFT screen-moiré detection
_FFT_SIZE = 64
_HANN_2D = np.outer(np.hanning(_FFT_SIZE), np.hanning(_FFT_SIZE)).astype(np.float32)
_FY = np.fft.fftfreq(_FFT_SIZE)[:, None] * _FFT_SIZE
_FX = np.fft.rfftfreq(_FFT_SIZE)[None, :] * _FFT_SIZE
_FRAD = np.hypot(_FX, _FY)
# Exclude low-frequency facial structure (<7) and 8px JPEG block harmonics near edges (>26)
_FFT_ANNULUS_MASK = (_FRAD >= 7.0) & (_FRAD <= 26.0)

# Anti-proxy identity continuity thresholds
IDENTITY_ENROLL_SAMPLES = 8
IDENTITY_MISMATCH_THRESHOLD = 0.56
IDENTITY_MISMATCH_STREAK = 4
IDENTITY_ALERT_COOLDOWN_MS = 25_000


# ---------------------------------------------------------------------------
# Adaptive One-Euro Filter (Casiez et al., CHI 2012)
# ---------------------------------------------------------------------------
class _OneEuroScalar:
    """Velocity-adaptive low-pass filter: suppresses static jitter with near-zero saccade lag."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.08, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[int] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-3))

    def filter(self, value: float, ts_ms: int) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = value
            self._t_prev = ts_ms
            return value

        dt = (ts_ms - self._t_prev) / 1000.0
        if dt <= 0.0:
            dt = 1.0
        elif dt > 5.0:
            # Reset state after long silence/occlusion gaps
            self._x_prev = value
            self._dx_prev = 0.0
            self._t_prev = ts_ms
            return value

        dt = max(0.02, dt)
        dx = (value - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * value + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = ts_ms
        return x_hat


@dataclass
class _WorkerSessionState:
    prev_thumb: Optional[np.ndarray] = None
    prev_face_gray: Optional[np.ndarray] = None

    # One-Euro filters for jitter-free temporal kinematics
    f_gaze_x: _OneEuroScalar = field(default_factory=lambda: _OneEuroScalar(min_cutoff=1.4, beta=0.35))
    f_gaze_y: _OneEuroScalar = field(default_factory=lambda: _OneEuroScalar(min_cutoff=1.4, beta=0.35))
    f_yaw: _OneEuroScalar = field(default_factory=lambda: _OneEuroScalar(min_cutoff=1.2, beta=0.05))
    f_pitch: _OneEuroScalar = field(default_factory=lambda: _OneEuroScalar(min_cutoff=1.2, beta=0.05))
    f_roll: _OneEuroScalar = field(default_factory=lambda: _OneEuroScalar(min_cutoff=1.0, beta=0.04))

    # Biometric liveness history
    ear_history: Deque[float] = field(default_factory=lambda: deque(maxlen=40))
    moire_ema: float = 0.0

    # Baseline-calibrated chromatic lighting asymmetry (neutralizes static windows)
    glow_baseline_ema: Optional[float] = None
    glow_samples: int = 0

    # Candidate identity continuity (proxy / seat-swap defense)
    id_enroll_vectors: List[np.ndarray] = field(default_factory=list)
    id_template: Optional[np.ndarray] = None
    id_mismatch_streak: int = 0
    last_id_alert_ms: int = -1

    # VLM quota & UI telemetry throttling
    vlm_calls: int = 0
    last_vlm_ts: float = 0.0
    last_telemetry_ms: int = -1


class VisionWorker:
    """
    Zero-GPU Advanced 3-Tier Forensic Vision Worker:
      - Tier 0: Fast occlusion/blur gate + Hann-windowed 2D FFT screen-moiré + frame-delta liveness
      - Tier 1A: MediaPipe 478-point 3D FaceMesh -> LM-refined solvePnP Euler pose, roll-invariant
                 eye-basis iris projection, EAR blink biometrics, One-Euro kinematics filtering,
                 temporal blue-glow asymmetry, and 38-D proxy-swap identity continuity.
      - Tier 1B: Automatic fallback to cached OpenCV Haar + Timm & Barth (2011) vectorized
                 gradient-isophote pupil center localization when MediaPipe is unavailable.
      - Tier 2: Budget-capped Gemini Flash VLM verification on confirmed high-signal keyframes.
    """

    def __init__(self) -> None:
        self.storage = get_storage_adapter()
        self.gemini_judge = GeminiJudgeAdapter()
        self.vision_service = get_vision_service()

        self._cv_lock = threading.Lock()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self._mp_engine_kind: str = "opencv_isophote"
        self._mp_landmarker: Any = None
        self._mp_face_mesh: Any = None
        self._yunet_detector: Any = None
        self._yunet_input_size: Tuple[int, int] = (0, 0)
        self._models_bootstrap_attempted: bool = False
        self._model_bootstrap_lock = asyncio.Lock()

        self._init_landmark_engine()
        self._init_yunet_detector()

        self._sessions: "OrderedDict[str, _WorkerSessionState]" = OrderedDict()
        self.max_vlm_calls: int = int(getattr(settings, "MAX_VLM_CALLS_PER_SESSION", 12))
        self.vlm_cooldown_sec: float = float(getattr(settings, "VLM_COOLDOWN_SEC", 30.0))

    @staticmethod
    def _default_model_dir() -> str:
        return os.path.join(getattr(settings, "LOCAL_STORAGE_DIR", "./tmp_storage"), "models")

    def _resolve_model_path(self, setting_attr: str, default_filename: str) -> str:
        configured = getattr(settings, setting_attr, None)
        if configured and str(configured).strip():
            return str(configured).strip()
        return os.path.join(self._default_model_dir(), default_filename)

    async def _ensure_models_ready(self) -> None:
        if self._models_bootstrap_attempted:
            return
        if not bool(getattr(settings, "ENABLE_VISION_MODEL_AUTO_DOWNLOAD", True)):
            self._models_bootstrap_attempted = True
            return

        async with self._model_bootstrap_lock:
            if self._models_bootstrap_attempted:
                return
            self._models_bootstrap_attempted = True

            # 1. If mediapipe is installed and Tasks model isn't on disk, fetch face_landmarker.task
            if self._mp_landmarker is None:
                try:
                    import mediapipe as mp  # noqa: F401
                    mp_path = self._resolve_model_path("MEDIAPIPE_FACE_LANDMARKER_MODEL", "face_landmarker.task")
                    if not os.path.isfile(mp_path):
                        ok = await self._download_model_file(MEDIAPIPE_TASK_URL, mp_path)
                        if ok:
                            self._init_landmark_engine(explicit_mp_path=mp_path)
                except Exception:
                    pass

            # 2. If MediaPipe is not active, fetch tiny (232KB) OpenCV YuNet ONNX for 5-pt solvePnP
            if self._mp_engine_kind == "opencv_isophote" and self._yunet_detector is None:
                if hasattr(cv2, "FaceDetectorYN"):
                    yunet_path = self._resolve_model_path(
                        "OPENCV_YUNET_MODEL", "face_detection_yunet_2023mar.onnx"
                    )
                    if not os.path.isfile(yunet_path):
                        await self._download_model_file(OPENCV_YUNET_URL, yunet_path)
                    if os.path.isfile(yunet_path):
                        self._init_yunet_detector(explicit_path=yunet_path)

    @staticmethod
    async def _download_model_file(url: str, dest_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            tmp_path = f"{dest_path}.tmp"
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                res = await client.get(url)
                res.raise_for_status()
                content = res.content
            if not content or len(content) < 10_000:
                return False
            def _write() -> None:
                with open(tmp_path, "wb") as f:
                    f.write(content)
                os.replace(tmp_path, dest_path)
            await asyncio.to_thread(_write)
            logger.info("Downloaded cached vision model -> %s (%d bytes)", dest_path, len(content))
            return True
        except Exception as exc:
            logger.warning("Auto-download of vision model %s failed (%s); keeping fallback engine.", url, exc)
            return False

    def _init_yunet_detector(self, explicit_path: Optional[str] = None) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            return
        model_path = explicit_path or self._resolve_model_path(
            "OPENCV_YUNET_MODEL", "face_detection_yunet_2023mar.onnx"
        )
        if not os.path.isfile(model_path):
            return
        try:
            with self._cv_lock:
                self._yunet_detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(MAX_DECODE_WIDTH, 360),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=50,
                )
                self._yunet_input_size = (MAX_DECODE_WIDTH, 360)
            if self._mp_engine_kind == "opencv_isophote":
                self._mp_engine_kind = "opencv_yunet"
            logger.info("VisionWorker initialized OpenCV YuNet 5-point CNN face/pose detector.")
        except Exception as exc:
            logger.warning("OpenCV YuNet init failed (%s); using Haar fallback.", exc)
            self._yunet_detector = None

    def _init_landmark_engine(self, explicit_mp_path: Optional[str] = None) -> None:
        """Initializes highest-fidelity CPU 3D landmark backend available in the runtime."""
        try:
            import mediapipe as mp
        except Exception as exc:
            logger.info("MediaPipe not installed (%s); checking YuNet / isophote CPU engines.", exc)
            return

        # Option A: Modern MediaPipe Tasks FaceLandmarker (auto-cached or custom .task bundle)
        model_path = explicit_mp_path or self._resolve_model_path(
            "MEDIAPIPE_FACE_LANDMARKER_MODEL", "face_landmarker.task"
        )
        if model_path and os.path.isfile(str(model_path)):
            try:
                base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
                opts = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=3,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                with self._cv_lock:
                    self._mp_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
                    self._mp_engine_kind = "mediapipe_tasks"
                logger.info("VisionWorker initialized with MediaPipe Tasks FaceLandmarker (478 3D landmarks).")
                return
            except Exception as exc:
                logger.warning("MediaPipe Tasks FaceLandmarker init failed (%s); trying FaceMesh.", exc)

        # Option B: Legacy MediaPipe Solutions FaceMesh (when available)
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=3,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                self._mp_engine_kind = "mediapipe_facemesh"
                logger.info("VisionWorker initialized with MediaPipe FaceMesh (478 3D landmarks + iris).")
                return
        except Exception as exc:
            logger.warning("MediaPipe FaceMesh init failed (%s); checking YuNet / isophote fallback.", exc)

    async def aclose(self) -> None:
        with self._cv_lock:
            for handle in (self._mp_landmarker, self._mp_face_mesh):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
            self._mp_landmarker = None
            self._mp_face_mesh = None
            self._yunet_detector = None
        await self.gemini_judge.aclose()

    def _get_session_state(self, session_id: str) -> _WorkerSessionState:
        st = self._sessions.get(session_id)
        if st is not None:
            self._sessions.move_to_end(session_id)
            return st
        st = _WorkerSessionState()
        self._sessions[session_id] = st
        if len(self._sessions) > MAX_TRACKED_SESSIONS:
            self._sessions.popitem(last=False)
        return st

    async def handle_video_frame(self, event: VisionFrameEvent) -> None:
        now_ms = int(time.time() * 1000)
        ingested_at_ms = int(getattr(event, "ingested_at_ms", 0) or 0)

        # Epoch-safe backpressure guard: never drops frames on monotonic/test timestamps
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
            await self._ensure_models_ready()
            analysis = await asyncio.to_thread(
                self._analyze_frame_sync,
                event.session_id,
                event.timestamp_ms,
                jpeg_bytes,
            )
            if analysis is None:
                return

            telemetry_frame = VisionTelemetryFrame(
                session_id=event.session_id,
                timestamp_ms=event.timestamp_ms,
                camera_active=analysis["camera_active"],
                faces_detected=analysis["faces_detected"],
                objects_detected=analysis["objects_detected"],
                gaze_direction=analysis["gaze_direction"],
                gaze_offset_x=analysis["gaze_offset_x"],
                gaze_offset_y=analysis["gaze_offset_y"],
                head_pose_yaw=analysis["head_pose_yaw"],
                head_pose_pitch=analysis["head_pose_pitch"],
                head_pose_roll=analysis["head_pose_roll"],
                liveness_score=analysis["liveness_score"],
            )

            service_events = self.vision_service.ingest_frame_all(telemetry_frame)
            identity_alert = analysis.get("identity_mismatch_alert")

            # Broadcast live 1 Hz optical telemetry heartbeat so UI stays active even before anomalies fire
            st = self._get_session_state(event.session_id)
            if event.is_live and (st.last_telemetry_ms < 0 or (now_ms - st.last_telemetry_ms) >= VISION_TELEMETRY_INTERVAL_MS):
                st.last_telemetry_ms = now_ms
                await redis_bus.broadcast_ui(
                    session_id=event.session_id,
                    event_type="VISION_TELEMETRY",
                    data={
                        "timestamp_ms": event.timestamp_ms,
                        "engine": analysis.get("engine", self._mp_engine_kind),
                        "camera_active": analysis["camera_active"],
                        "faces_detected": analysis["faces_detected"],
                        "gaze_direction": analysis["gaze_direction"],
                        "gaze_offset_x": analysis["gaze_offset_x"],
                        "gaze_offset_y": analysis["gaze_offset_y"],
                        "head_pose_yaw": analysis["head_pose_yaw"],
                        "head_pose_pitch": analysis["head_pose_pitch"],
                        "head_pose_roll": analysis["head_pose_roll"],
                        "liveness_score": analysis["liveness_score"],
                    },
                )

            if not service_events and not identity_alert:
                return

            # Tier 2: At most ONE budget-capped Gemini Flash VLM check per flagged frame
            vlm_res: Optional[Dict[str, Any]] = None
            needs_vlm = bool(identity_alert) or any(ev.requires_vlm_verification for ev in service_events)
            if needs_vlm and self._can_invoke_vlm(event.session_id):
                self._record_vlm_call(event.session_id)
                vlm_res = await self.gemini_judge.verify_frame(
                    jpeg_bytes,
                    session_id=event.session_id,
                )

            retained_evidence = True
            vlm_note = (vlm_res or {}).get("note")
            vlm_phone_detected = bool((vlm_res or {}).get("phone_or_tablet_visible", False))
            vlm_visual_injection = bool((vlm_res or {}).get("visual_injection_detected", False))

            # Emit events surfaced by VisionProctoringService
            for svc_ev in service_events:
                anomaly_type = svc_ev.anomaly_type or VisionAnomalyType.GAZE_DEVIATION
                confidence = svc_ev.confidence
                note_out = vlm_note if svc_ev.requires_vlm_verification else None

                if vlm_phone_detected and svc_ev.requires_vlm_verification:
                    anomaly_type = VisionAnomalyType.DEVICE_DETECTED
                    confidence = max(confidence, 0.94)
                elif vlm_visual_injection and svc_ev.requires_vlm_verification:
                    anomaly_type = VisionAnomalyType.LIVENESS_SPOOF
                    confidence = max(confidence, 0.95)

                await self._emit_contract_event(
                    event=event,
                    category=svc_ev.category,
                    anomaly_type=anomaly_type,
                    duration_ms=svc_ev.duration_ms,
                    confidence=confidence,
                    pitch_deg=analysis["head_pose_pitch"],
                    yaw_deg=analysis["head_pose_yaw"],
                    vlm_note=note_out,
                )

            # Emit anti-proxy candidate substitution alert if detected independently
            if identity_alert:
                combined_note = f"{identity_alert}" + (f" | VLM: {vlm_note}" if vlm_note else "")
                await self._emit_contract_event(
                    event=event,
                    category="Candidate Identity Substitution",
                    anomaly_type=VisionAnomalyType.LIVENESS_SPOOF,
                    duration_ms=4_000,
                    confidence=0.86,
                    pitch_deg=analysis["head_pose_pitch"],
                    yaw_deg=analysis["head_pose_yaw"],
                    vlm_note=combined_note[:300],
                )

        finally:
            if not retained_evidence:
                await self._safe_delete(event.storage_key)
            metrics.inc("video_frames_processed")

    # -----------------------------------------------------------------------
    # Tier 0 & Tier 1 Synchronous CPU Computer Vision Pipeline
    # -----------------------------------------------------------------------
    def _analyze_frame_sync(
        self,
        session_id: str,
        timestamp_ms: int,
        jpeg_bytes: bytes,
    ) -> Optional[Dict[str, Any]]:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return None

        h_orig, w_orig = bgr.shape[:2]
        if w_orig > MAX_DECODE_WIDTH:
            scale = MAX_DECODE_WIDTH / float(w_orig)
            bgr = cv2.resize(bgr, (MAX_DECODE_WIDTH, max(1, int(h_orig * scale))), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]

        # Tier 0: Fast occlusion / black-screen / tape-over-lens gate
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(np.mean(gray))
        if lap_var < 10.0 or mean_luma < 8.0:
            return {
                "engine": self._mp_engine_kind,
                "camera_active": False,
                "faces_detected": 0,
                "objects_detected": [],
                "gaze_direction": "center",
                "gaze_offset_x": 0.0,
                "gaze_offset_y": 0.0,
                "head_pose_yaw": 0.0,
                "head_pose_pitch": 0.0,
                "head_pose_roll": 0.0,
                "liveness_score": 0.98,
                "identity_mismatch_alert": None,
            }

        state = self._get_session_state(session_id)

        # Tier 0B: Global frame-delta micro-motion score
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        if state.prev_thumb is not None:
            global_motion = float(np.mean(cv2.absdiff(thumb, state.prev_thumb)))
        else:
            global_motion = 2.0
        state.prev_thumb = thumb

        # Tier 1A (MediaPipe 478 3D Landmarks) -> Tier 1B (OpenCV YuNet 5-pt solvePnP) -> Tier 1C (Haar)
        geom = self._extract_mediapipe_geometry(bgr, w_img, h_img)
        if geom is None:
            geom = self._extract_yunet_geometry(bgr, gray, w_img, h_img)
        if geom is None:
            geom = self._extract_opencv_isophote_geometry(gray, w_img, h_img)

        active_engine = str(geom.get("engine") or self._mp_engine_kind)

        num_faces = geom["faces_detected"]
        raw_yaw = geom["yaw_deg"]
        raw_pitch = geom["pitch_deg"]
        raw_roll = geom["roll_deg"]
        raw_gaze_x = geom["gaze_x"]
        raw_gaze_y = geom["gaze_y"]
        ear = geom["ear"]
        face_bbox = geom["face_bbox"]
        geom_ratios = geom["geom_ratios"]

        # Smooth kinematic estimates with velocity-adaptive One-Euro Filters
        if num_faces >= 1:
            gaze_x = max(-1.0, min(1.0, state.f_gaze_x.filter(raw_gaze_x, timestamp_ms)))
            gaze_y = max(-1.0, min(1.0, state.f_gaze_y.filter(raw_gaze_y, timestamp_ms)))
            yaw_deg = max(-65.0, min(65.0, state.f_yaw.filter(raw_yaw, timestamp_ms)))
            pitch_deg = max(-65.0, min(65.0, state.f_pitch.filter(raw_pitch, timestamp_ms)))
            roll_deg = max(-45.0, min(45.0, state.f_roll.filter(raw_roll, timestamp_ms)))
        else:
            gaze_x = gaze_y = yaw_deg = pitch_deg = roll_deg = 0.0

        # Multi-signal zero-GPU liveness fusion + off-axis screen glow + identity continuity
        objects_detected: List[str] = []
        identity_alert: Optional[str] = None
        liveness_score = 0.96

        if num_faces >= 1 and face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            face_gray = gray[fy : fy + fh, fx : fx + fw]
            face_bgr = bgr[fy : fy + fh, fx : fx + fw]

            # 1. Non-rigid facial micro-motion + 2D FFT screen-moiré + EAR blink dynamics
            liveness_score = self._compute_fused_liveness(
                state=state,
                face_gray=face_gray,
                global_motion=global_motion,
                ear=ear,
            )

            # 2. Baseline-calibrated temporal off-axis blue-glow detection (phone/2nd monitor flash)
            if self._detect_temporal_screen_glow(state, face_bgr, gaze_x, yaw_deg):
                objects_detected.append("second screen")

            # 3. Illumination-invariant candidate identity continuity (proxy seat-swap defense)
            if num_faces == 1 and lap_var >= 35.0 and abs(yaw_deg) <= 16.0 and abs(pitch_deg) <= 16.0:
                identity_alert = self._check_identity_continuity(
                    state=state,
                    face_gray=face_gray,
                    geom_ratios=geom_ratios,
                    timestamp_ms=timestamp_ms,
                )
        else:
            liveness_score = 0.35 if global_motion < 0.20 else 0.92

        gaze_direction = "center"
        if num_faces >= 1:
            if gaze_y > 0.30 or pitch_deg < -20.0:
                gaze_direction = "down"
            elif gaze_x > 0.34:
                gaze_direction = "right"
            elif gaze_x < -0.34:
                gaze_direction = "left"

        return {
            "engine": active_engine,
            "camera_active": True,
            "faces_detected": num_faces,
            "objects_detected": objects_detected,
            "gaze_direction": gaze_direction,
            "gaze_offset_x": round(float(gaze_x), 4),
            "gaze_offset_y": round(float(gaze_y), 4),
            "head_pose_yaw": round(float(yaw_deg), 2),
            "head_pose_pitch": round(float(pitch_deg), 2),
            "head_pose_roll": round(float(roll_deg), 2),
            "liveness_score": round(float(liveness_score), 3),
            "identity_mismatch_alert": identity_alert,
        }

    # -----------------------------------------------------------------------
    # Tier 1A: MediaPipe 478-Point 3D Landmark + EPnP/LM SolvePnP + Iris Basis
    # -----------------------------------------------------------------------
    def _extract_mediapipe_geometry(
        self,
        bgr: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> Optional[Dict[str, Any]]:
        if self._mp_engine_kind == "opencv_isophote":
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces_pts: List[np.ndarray] = []

        try:
            with self._cv_lock:
                if self._mp_engine_kind == "mediapipe_tasks" and self._mp_landmarker is not None:
                    import mediapipe as mp

                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = self._mp_landmarker.detect(mp_img)
                    for lm_list in res.face_landmarks or []:
                        pts = np.array(
                            [(lm.x * w_img, lm.y * h_img, lm.z * w_img) for lm in lm_list],
                            dtype=np.float64,
                        )
                        faces_pts.append(pts)
                elif self._mp_engine_kind == "mediapipe_facemesh" and self._mp_face_mesh is not None:
                    res = self._mp_face_mesh.process(rgb)
                    for face_lms in res.multi_face_landmarks or []:
                        pts = np.array(
                            [(lm.x * w_img, lm.y * h_img, lm.z * w_img) for lm in face_lms.landmark],
                            dtype=np.float64,
                        )
                        faces_pts.append(pts)
        except Exception as exc:
            logger.debug("MediaPipe inference error (%s); falling back to OpenCV.", exc)
            return None

        if not faces_pts:
            return {
                "engine": self._mp_engine_kind,
                "faces_detected": 0,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "ear": None,
                "face_bbox": None,
                "geom_ratios": None,
            }

        # Select primary (largest foreground) face by 2D bounding-box area
        primary_pts = max(
            faces_pts,
            key=lambda p: float((np.max(p[:, 0]) - np.min(p[:, 0])) * (np.max(p[:, 1]) - np.min(p[:, 1]))),
        )

        # Face ROI clamped to frame bounds
        x1 = max(0, int(np.min(primary_pts[:, 0])))
        y1 = max(0, int(np.min(primary_pts[:, 1])))
        x2 = min(w_img, int(np.max(primary_pts[:, 0])))
        y2 = min(h_img, int(np.max(primary_pts[:, 1])))
        face_bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1)) if (x2 > x1 + 12 and y2 > y1 + 12) else None

        # 1. Exact 3D Head Pose via Levenberg-Marquardt solvePnP
        yaw_deg, pitch_deg, roll_deg = self._solve_head_pose_pnp(primary_pts, w_img, h_img)

        # 2. Roll-Invariant Local Eye-Basis Iris Projection
        gaze_x, gaze_y = self._compute_iris_basis_gaze(primary_pts)

        # 3. Eye Aspect Ratio (EAR) for blink-dynamics liveness
        ear = self._compute_mean_ear(primary_pts)

        # 4. 6 Scale-Invariant 3D Inter-Landmark Anthropometric Ratios (anti-proxy signature)
        geom_ratios = self._compute_anthropometric_ratios(primary_pts)

        return {
            "engine": self._mp_engine_kind,
            "faces_detected": len(faces_pts),
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ear": ear,
            "face_bbox": face_bbox,
            "geom_ratios": geom_ratios,
        }

    @staticmethod
    def _solve_pnp_pose(
        model_points_3d: np.ndarray,
        image_points: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> Tuple[float, float, float]:
        focal = float(max(w_img, h_img))
        cam_matrix = np.array(
            [
                [focal, 0.0, w_img * 0.5],
                [0.0, focal, h_img * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Seed in front of camera (60cm) so LM optimization converges to valid cheirality
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            model_points_3d,
            image_points,
            cam_matrix,
            dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 50.0:
            return (0.0, 0.0, 0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        nx, ny, nz = -float(rmat[0, 2]), -float(rmat[1, 2]), -float(rmat[2, 2])
        yaw_deg = math.degrees(math.atan2(nx, max(1e-6, -nz)))
        pitch_deg = math.degrees(math.atan2(-ny, max(1e-6, math.hypot(nx, nz))))
        roll_deg = math.degrees(math.atan2(float(rmat[1, 0]), float(rmat[0, 0])))

        return (
            max(-65.0, min(65.0, yaw_deg)),
            max(-65.0, min(65.0, pitch_deg)),
            max(-45.0, min(45.0, roll_deg)),
        )

    @classmethod
    def _solve_head_pose_pnp(
        cls,
        pts: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> Tuple[float, float, float]:
        if pts.shape[0] < 300:
            return (0.0, 0.0, 0.0)
        image_points = np.array([pts[idx, :2] for idx in PNP_LANDMARK_IDS], dtype=np.float64)
        return cls._solve_pnp_pose(MODEL_POINTS_3D, image_points, w_img, h_img)

    # -----------------------------------------------------------------------
    # Tier 1B: OpenCV YuNet CNN 5-Point Keypoints + solvePnP + Isophote Iris
    # -----------------------------------------------------------------------
    def _extract_yunet_geometry(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> Optional[Dict[str, Any]]:
        if self._yunet_detector is None:
            return None

        try:
            with self._cv_lock:
                if self._yunet_input_size != (w_img, h_img):
                    self._yunet_detector.setInputSize((w_img, h_img))
                    self._yunet_input_size = (w_img, h_img)
                _, faces = self._yunet_detector.detect(bgr)
        except Exception as exc:
            logger.debug("YuNet inference failed (%s); falling back to Haar.", exc)
            return None

        if faces is None or len(faces) == 0:
            return {
                "engine": "opencv_yunet",
                "faces_detected": 0,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "ear": None,
                "face_bbox": None,
                "geom_ratios": None,
            }

        num_faces = int(len(faces))
        primary = max(faces, key=lambda d: float(max(0.0, d[2]) * max(0.0, d[3])))
        fx = max(0, int(primary[0]))
        fy = max(0, int(primary[1]))
        fw = max(1, min(w_img - fx, int(primary[2])))
        fh = max(1, min(h_img - fy, int(primary[3])))
        face_bbox = (fx, fy, fw, fh) if (fw > 12 and fh > 12) else None

        # YuNet landmarks: viewer-left eye, viewer-right eye, nose tip, left mouth, right mouth
        image_points = np.array(
            [
                (float(primary[4]), float(primary[5])),
                (float(primary[6]), float(primary[7])),
                (float(primary[8]), float(primary[9])),
                (float(primary[10]), float(primary[11])),
                (float(primary[12]), float(primary[13])),
            ],
            dtype=np.float64,
        )
        yaw_deg, pitch_deg, roll_deg = self._solve_pnp_pose(
            YUNET_MODEL_POINTS_3D, image_points, w_img, h_img
        )

        p_le, p_re, p_nose, p_lm, p_rm = image_points
        inter_ocular = float(np.linalg.norm(p_re - p_le))
        gaze_x, gaze_y = 0.0, 0.0
        geom_ratios = None

        if inter_ocular >= 12.0:
            crop_w = max(14, int(inter_ocular * 0.40))
            crop_h = max(10, int(inter_ocular * 0.24))
            pupils: List[Tuple[float, float]] = []
            for ex_c, ey_c in (p_le, p_re):
                x1 = max(0, int(ex_c - crop_w * 0.5))
                y1 = max(0, int(ey_c - crop_h * 0.5))
                x2 = min(w_img, x1 + crop_w)
                y2 = min(h_img, y1 + crop_h)
                eye_patch = gray[y1:y2, x1:x2]
                if eye_patch.size >= 64:
                    px_norm, py_norm = self._timm_barth_pupil_center(eye_patch)
                    pupils.append((px_norm, py_norm))
            if pupils:
                mean_px = sum(p[0] for p in pupils) / len(pupils)
                mean_py = sum(p[1] for p in pupils) / len(pupils)
                gaze_x = max(-1.0, min(1.0, (mean_px - 0.5) * 3.0))
                gaze_y = max(-1.0, min(1.0, (mean_py - 0.5) * 3.0))

            eye_mid = 0.5 * (p_le + p_re)
            mouth_mid = 0.5 * (p_lm + p_rm)
            geom_ratios = np.array(
                [
                    float(np.linalg.norm(p_nose - eye_mid)) / inter_ocular,
                    float(np.linalg.norm(mouth_mid - p_nose)) / inter_ocular,
                    float(np.linalg.norm(p_rm - p_lm)) / inter_ocular,
                ],
                dtype=np.float32,
            )

        return {
            "engine": "opencv_yunet",
            "faces_detected": num_faces,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ear": None,
            "face_bbox": face_bbox,
            "geom_ratios": geom_ratios,
        }

    @staticmethod
    def _compute_iris_basis_gaze(pts: np.ndarray) -> Tuple[float, float]:
        """Projects iris centroids onto each eye's roll-invariant (u_hat, v_hat) corner basis."""
        if pts.shape[0] < 478:
            return (0.0, 0.0)

        def _eye_offset(
            left_corner_idx: int,
            right_corner_idx: int,
            iris_ids: Tuple[int, ...],
        ) -> Optional[Tuple[float, float]]:
            p_l = pts[left_corner_idx, :2]
            p_r = pts[right_corner_idx, :2]
            vec = p_r - p_l
            eye_w = float(np.hypot(vec[0], vec[1]))
            if eye_w < 4.0:
                return None

            u_hat = vec / eye_w                                   # Points image-right
            v_hat = np.array([-u_hat[1], u_hat[0]], dtype=np.float64)  # Points image-down
            mid = 0.5 * (p_l + p_r)
            iris = np.mean(pts[list(iris_ids), :2], axis=0)
            rel = iris - mid

            # Normalize by anthropometric sclera travel (~28% width horizontally, ~15% vertically)
            gx = max(-1.0, min(1.0, float(np.dot(rel, u_hat)) / (0.28 * eye_w)))
            gy = max(-1.0, min(1.0, float(np.dot(rel, v_hat)) / (0.15 * eye_w)))
            return (gx, gy)

        left_g = _eye_offset(LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_IDS)
        right_g = _eye_offset(RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS_IDS)
        valid = [g for g in (left_g, right_g) if g is not None]
        if not valid:
            return (0.0, 0.0)

        gx = sum(g[0] for g in valid) / len(valid)
        gy = sum(g[1] for g in valid) / len(valid)
        return (gx, gy)

    @staticmethod
    def _compute_mean_ear(pts: np.ndarray) -> Optional[float]:
        if pts.shape[0] < 400:
            return None

        def _ear(ids: Tuple[int, int, int, int, int, int]) -> float:
            p1, p2, p3, p4, p5, p6 = (pts[i, :2] for i in ids)
            horiz = float(np.linalg.norm(p1 - p4))
            if horiz < 2.0:
                return 0.28
            vert = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
            return vert / (2.0 * horiz)

        return 0.5 * (_ear(LEFT_EYE_EAR_IDS) + _ear(RIGHT_EYE_EAR_IDS))

    @staticmethod
    def _compute_anthropometric_ratios(pts: np.ndarray) -> Optional[np.ndarray]:
        if pts.shape[0] < 400:
            return None
        inter_ocular = float(np.linalg.norm(pts[33, :2] - pts[263, :2]))
        if inter_ocular < 10.0:
            return None
        nose_len = float(np.linalg.norm(pts[168, :2] - pts[1, :2]))
        nose_chin = float(np.linalg.norm(pts[1, :2] - pts[152, :2]))
        mouth_w = float(np.linalg.norm(pts[61, :2] - pts[291, :2]))
        jaw_w = float(np.linalg.norm(pts[234, :2] - pts[454, :2]))
        eye_w_l = float(np.linalg.norm(pts[33, :2] - pts[133, :2]))
        eye_w_r = float(np.linalg.norm(pts[362, :2] - pts[263, :2]))
        return np.array(
            [
                nose_len / inter_ocular,
                nose_chin / inter_ocular,
                mouth_w / inter_ocular,
                jaw_w / inter_ocular,
                eye_w_l / inter_ocular,
                eye_w_r / inter_ocular,
            ],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # Tier 1B: OpenCV Haar + Timm & Barth (2011) Vectorized Isophote Pupil Tracker
    # -----------------------------------------------------------------------
    def _extract_opencv_isophote_geometry(
        self,
        gray: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> Dict[str, Any]:
        with self._cv_lock:
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(56, 56),
            )

        num_faces = len(faces)
        if num_faces == 0:
            return {
                "engine": "opencv_isophote",
                "faces_detected": 0,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "ear": None,
                "face_bbox": None,
                "geom_ratios": None,
            }

        x, y, w, h = (int(v) for v in max(faces, key=lambda b: int(b[2]) * int(b[3])))
        cx = (x + w * 0.5) / max(w_img, 1)
        cy = (y + h * 0.5) / max(h_img, 1)
        dx = (cx - 0.5) * 2.0
        dy = (cy - 0.5) * 2.0

        yaw_deg = max(-45.0, min(45.0, dx * 34.0))
        pitch_deg = max(-45.0, min(45.0, -dy * 34.0))
        roll_deg = 0.0
        gaze_x = max(-1.0, min(1.0, dx * 0.55))
        gaze_y = max(-1.0, min(1.0, dy * 0.60))

        roi_h = max(1, int(h * 0.58))
        roi_gray = gray[y : y + roi_h, x : x + w]
        with self._cv_lock:
            eyes = self._eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(16, 16),
            )

        if len(eyes) >= 2:
            top_eyes = sorted(eyes, key=lambda e: int(e[2]) * int(e[3]), reverse=True)[:2]
            top_eyes.sort(key=lambda e: int(e[0]))
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = (
                tuple(int(v) for v in top_eyes[0]),
                tuple(int(v) for v in top_eyes[1]),
            )
            roll_deg = float(
                math.degrees(
                    math.atan2(
                        (ey2 + eh2 * 0.5) - (ey1 + eh1 * 0.5),
                        max((ex2 + ew2 * 0.5) - (ex1 + ew1 * 0.5), 1.0),
                    )
                )
            )
            # Refine yaw from eye-pair horizontal symmetry inside face bounding box
            eye_mid_rel = (((ex1 + ew1 * 0.5) + (ex2 + ew2 * 0.5)) * 0.5) / max(w, 1)
            yaw_deg = max(-45.0, min(45.0, (eye_mid_rel - 0.5) * 95.0 + dx * 12.0))

            # Timm & Barth gradient-isophote iris center inside each eye crop
            pupils = []
            for ex, ey, ew, eh in ((ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2)):
                # Trim upper eyebrow band (top 22%) and outer padding
                y_pad = int(eh * 0.22)
                eye_patch = roi_gray[ey + y_pad : ey + eh, ex : ex + ew]
                if eye_patch.size >= 64:
                    px_norm, py_norm = self._timm_barth_pupil_center(eye_patch)
                    pupils.append((px_norm, py_norm))

            if pupils:
                mean_px = sum(p[0] for p in pupils) / len(pupils)
                mean_py = sum(p[1] for p in pupils) / len(pupils)
                gaze_x = max(-1.0, min(1.0, (mean_px - 0.5) * 3.0))
                gaze_y = max(-1.0, min(1.0, (mean_py - 0.5) * 3.0))

        return {
            "engine": "opencv_isophote",
            "faces_detected": num_faces,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "ear": None,
            "face_bbox": (x, y, w, h),
            "geom_ratios": None,
        }

    @staticmethod
    def _timm_barth_pupil_center(eye_gray: np.ndarray) -> Tuple[float, float]:
        """
        Vectorized Timm & Barth (2011) isophote gradient dot-product iris locator.
        Runs in <0.4ms on CPU NumPy at 20x12 resolution.
        """
        tw, th = 20, 12
        patch = cv2.resize(eye_gray, (tw, th), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(patch, (3, 3), 0).astype(np.float64)

        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)

        thresh = float(np.mean(mag) + 0.3 * np.std(mag))
        gy_idx, gx_idx = np.where(mag > max(thresh, 1e-3))
        if gx_idx.size < 4:
            return (0.5, 0.5)

        # Keep strongest 80 gradient boundary pixels for bounded O(1) compute
        if gx_idx.size > 80:
            strong = np.argpartition(mag[gy_idx, gx_idx], -80)[-80:]
            gx_idx, gy_idx = gx_idx[strong], gy_idx[strong]

        g_pos = np.stack([gx_idx, gy_idx], axis=1).astype(np.float64)  # (K, 2)
        g_mag = mag[gy_idx, gx_idx][:, None]
        g_dir = np.stack([gx[gy_idx, gx_idx], gy[gy_idx, gx_idx]], axis=1) / (g_mag + 1e-6)

        cy_grid, cx_grid = np.mgrid[0:th, 0:tw]
        centers = np.stack([cx_grid.ravel(), cy_grid.ravel()], axis=1).astype(np.float64)  # (M, 2)

        disp = g_pos[None, :, :] - centers[:, None, :]
        dist = np.hypot(disp[..., 0], disp[..., 1]) + 1e-6
        dots = (disp[..., 0] * g_dir[None, :, 0] + disp[..., 1] * g_dir[None, :, 1]) / dist
        dots = np.maximum(0.0, dots)

        # Dark-pixel weighting + gentle centered Gaussian prior to reject dark corner shadows
        dark_w = (255.0 - blur.ravel()) / 255.0
        prior = np.exp(-(((cx_grid.ravel() / tw - 0.5) ** 2) / 0.18 + ((cy_grid.ravel() / th - 0.5) ** 2) / 0.18))
        scores = dark_w * prior * np.mean(dots * dots, axis=1)

        best = int(np.argmax(scores))
        return (float((best % tw) + 0.5) / tw, float((best // tw) + 0.5) / th)

    # -----------------------------------------------------------------------
    # Zero-GPU Anti-Spoofing, Chromatic Screen-Glow & Identity Continuity
    # -----------------------------------------------------------------------
    @staticmethod
    def _compute_fused_liveness(
        state: _WorkerSessionState,
        face_gray: np.ndarray,
        global_motion: float,
        ear: Optional[float],
    ) -> float:
        # 1. Local facial micro-motion (separates living face from frozen stream)
        face_64 = cv2.resize(face_gray, (_FFT_SIZE, _FFT_SIZE), interpolation=cv2.INTER_AREA)
        if state.prev_face_gray is not None:
            face_motion = float(np.mean(cv2.absdiff(face_64, state.prev_face_gray)))
        else:
            face_motion = 2.0
        state.prev_face_gray = face_64

        if global_motion < 0.20 and face_motion < 0.25:
            motion_score = 0.30
        else:
            motion_score = min(0.99, 0.72 + min(max(global_motion, face_motion) / 8.0, 0.27))

        # 2. Hann-windowed 2D FFT radial spectral spike ratio (LCD/OLED replay moiré check)
        f32 = face_64.astype(np.float32)
        f32 = (f32 - float(np.mean(f32))) * _HANN_2D
        spec = np.abs(np.fft.rfft2(f32))
        annulus = spec[_FFT_ANNULUS_MASK]
        if annulus.size > 0:
            med = float(np.median(annulus)) + 1e-4
            peak_ratio = float(np.max(annulus)) / med
            state.moire_ema = peak_ratio if state.moire_ema == 0.0 else (0.25 * peak_ratio + 0.75 * state.moire_ema)
        moire_penalty = 0.0
        if state.moire_ema > 28.0:
            moire_penalty = min(0.32, (state.moire_ema - 28.0) * 0.015)

        # 3. Eye Aspect Ratio (EAR) blink-variance biometric (catches moving hand-held photo)
        blink_penalty = 0.0
        if ear is not None:
            state.ear_history.append(ear)
            if len(state.ear_history) >= 25:
                ear_std = float(np.std(state.ear_history))
                if ear_std < 0.006:
                    blink_penalty = 0.22

        return max(0.10, min(0.99, motion_score - moire_penalty - blink_penalty))

    @staticmethod
    def _detect_temporal_screen_glow(
        state: _WorkerSessionState,
        face_bgr: np.ndarray,
        gaze_x: float,
        yaw_deg: float,
    ) -> bool:
        """
        Detects sudden asymmetric blue-white monitor/phone glow on one side of the face
        relative to the candidate's own calibrated lighting baseline (window-safe).
        """
        fh, fw = face_bgr.shape[:2]
        if fw < 24 or fh < 24:
            return False

        mid_y1, mid_y2 = int(fh * 0.25), int(fh * 0.75)
        left_cheek = face_bgr[mid_y1:mid_y2, : fw // 2]
        right_cheek = face_bgr[mid_y1:mid_y2, fw // 2 :]
        if left_cheek.size == 0 or right_cheek.size == 0:
            return False

        # BGR channel order: index 0 = Blue, index 2 = Red
        lb, lr = float(np.mean(left_cheek[:, :, 0])), float(np.mean(left_cheek[:, :, 2]))
        rb, rr = float(np.mean(right_cheek[:, :, 0])), float(np.mean(right_cheek[:, :, 2]))
        asym = math.log(((rb + 2.0) / (rr + 2.0)) / ((lb + 2.0) / (lr + 2.0)))

        if state.glow_baseline_ema is None:
            state.glow_baseline_ema = asym
            state.glow_samples = 1
            return False

        delta = asym - state.glow_baseline_ema
        if abs(gaze_x) < 0.22 and abs(yaw_deg) < 14.0:
            alpha = 0.15 if state.glow_samples < 12 else 0.03
            state.glow_baseline_ema = (1.0 - alpha) * state.glow_baseline_ema + alpha * asym
            state.glow_samples += 1
            return False

        # Require both substantial temporal color shift AND directional alignment with off-axis gaze
        if state.glow_samples >= 8 and abs(delta) >= 0.28:
            aligned_right = delta > 0.28 and (gaze_x > 0.25 or yaw_deg > 16.0)
            aligned_left = delta < -0.28 and (gaze_x < -0.25 or yaw_deg < -16.0)
            return aligned_right or aligned_left
        return False

    @staticmethod
    def _check_identity_continuity(
        state: _WorkerSessionState,
        face_gray: np.ndarray,
        geom_ratios: Optional[np.ndarray],
        timestamp_ms: int,
    ) -> Optional[str]:
        """
        Lightweight illumination-invariant 38-D descriptor (32-D Sobel orientation grid + 6-D 3D ratios)
        detecting mid-interview proxy/seat-swap substitution on clean frontal frames.
        """
        patch = cv2.resize(face_gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)  # [0, 1)

        # 2x2 spatial grid x 8 orientation bins = 32-D illumination-invariant HOG descriptor
        hog_parts: List[np.ndarray] = []
        for gy_cell in (0, 16):
            for gx_cell in (0, 16):
                cell_m = mag[gy_cell : gy_cell + 16, gx_cell : gx_cell + 16].ravel()
                cell_a = ang[gy_cell : gy_cell + 16, gx_cell : gx_cell + 16].ravel()
                hist, _ = np.histogram(cell_a, bins=8, range=(0.0, 1.0), weights=cell_m)
                hog_parts.append(hist.astype(np.float32))
        hog_vec = np.concatenate(hog_parts)
        hog_norm = float(np.linalg.norm(hog_vec)) + 1e-6
        hog_vec /= hog_norm

        if geom_ratios is not None:
            g_norm = float(np.linalg.norm(geom_ratios)) + 1e-6
            desc = np.concatenate([hog_vec * 0.70, (geom_ratios / g_norm) * 0.30])
        else:
            desc = hog_vec
        desc = desc / (float(np.linalg.norm(desc)) + 1e-6)

        if state.id_template is None:
            state.id_enroll_vectors.append(desc)
            if len(state.id_enroll_vectors) >= IDENTITY_ENROLL_SAMPLES:
                med_vec = np.median(np.stack(state.id_enroll_vectors, axis=0), axis=0)
                state.id_template = med_vec / (float(np.linalg.norm(med_vec)) + 1e-6)
                state.id_enroll_vectors.clear()
            return None

        # Only compare equal-dimensionality descriptors (e.g. if fallback switched mid-call)
        if desc.shape != state.id_template.shape:
            return None

        sim = float(np.dot(desc, state.id_template))
        if sim < IDENTITY_MISMATCH_THRESHOLD:
            state.id_mismatch_streak += 1
            if (
                state.id_mismatch_streak >= IDENTITY_MISMATCH_STREAK
                and (
                    state.last_id_alert_ms < 0
                    or (timestamp_ms - state.last_id_alert_ms) >= IDENTITY_ALERT_COOLDOWN_MS
                )
            ):
                state.last_id_alert_ms = timestamp_ms
                state.id_mismatch_streak = 0
                return (
                    f"Candidate frontal facial descriptor diverged from enrolled baseline "
                    f"(cosine_sim={sim:.2f} < {IDENTITY_MISMATCH_THRESHOLD:.2f}); verify identity continuity."
                )
        else:
            state.id_mismatch_streak = 0
        return None

    # -----------------------------------------------------------------------
    # Persistence, Publishing & VLM Quota Helpers
    # -----------------------------------------------------------------------
    async def _emit_contract_event(
        self,
        event: VisionFrameEvent,
        category: str,
        anomaly_type: VisionAnomalyType,
        duration_ms: int,
        confidence: float,
        pitch_deg: float,
        yaw_deg: float,
        vlm_note: Optional[str],
    ) -> None:
        event_id = self._deterministic_event_id(
            session_id=event.session_id,
            timestamp_ms=event.timestamp_ms,
            frame_index=getattr(event, "frame_index", 0),
            category=category,
        )
        v_event = ContractVisionAnomalyEvent(
            event_id=event_id,
            session_id=event.session_id,
            timestamp_ms=event.timestamp_ms,
            anomaly_type=anomaly_type,
            duration_ms=duration_ms,
            confidence=confidence,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            evidence_frame_key=event.storage_key,
            vlm_verification_note=vlm_note,
        )
        await self._persist_vision_event(v_event)
        await redis_bus.publish_stream("stream:vision_events", v_event)
        await redis_bus.broadcast_ui(
            session_id=event.session_id,
            event_type="VISION_ANOMALY",
            data=v_event.model_dump(mode="json"),
        )
        metrics.inc("vision_anomalies_detected")

    def _can_invoke_vlm(self, session_id: str) -> bool:
        st = self._get_session_state(session_id)
        if st.vlm_calls >= self.max_vlm_calls:
            return False
        return (time.time() - st.last_vlm_ts) >= self.vlm_cooldown_sec

    def _record_vlm_call(self, session_id: str) -> None:
        st = self._get_session_state(session_id)
        st.vlm_calls += 1
        st.last_vlm_ts = time.time()

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
    logger.info(
        "VisionWorker '%s' started (engine=%s) consuming from 'stream:video'",
        consumer_id,
        worker._mp_engine_kind,
    )
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