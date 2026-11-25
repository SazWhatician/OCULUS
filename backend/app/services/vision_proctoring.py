import logging
import math
import statistics
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from app.schemas.contracts import VisionAnomalyType

logger = logging.getLogger("oculus.vision.proctoring")

# ---------------------------------------------------------------------------
# Forensic Thresholds & Memory Safety Constants
# ---------------------------------------------------------------------------
MAX_ACTIVE_SESSIONS = 500
FRAME_RING_BUFFER_SIZE = 180          # ~90-180s rolling window per session (prevents OOM)
MAX_STORED_EVENTS_PER_SESSION = 500

# Temporal persistence requirements (wall-clock milliseconds, FPS-independent)
STREAK_GAP_TOLERANCE_MS = 1_800
GAZE_PERSISTENCE_MS = 3_000
SPLIT_ANGLE_PERSISTENCE_MS = 2_600    # Slightly faster trigger when head + eyes compound
HEAD_POSE_PERSISTENCE_MS = 3_000
CAMERA_OFF_PERSISTENCE_MS = 2_000
FACE_ABSENT_PERSISTENCE_MS = 3_000
MULTI_FACE_PERSISTENCE_MS = 1_500
DEVICE_PERSISTENCE_MS = 600
LIVENESS_PERSISTENCE_MS = 2_000
FROZEN_FEED_PERSISTENCE_MS = 6_000

# Peek-loop (anti-debouncer evasion) sliding window
PEEK_WINDOW_MS = 20_000
PEEK_MIN_SPAN_MS = 12_000
PEEK_MIN_FRAMES = 8
PEEK_MIN_TRANSITIONS = 3
PEEK_MIN_OFFSCREEN_RATIO = 0.40

# Per-category cooldowns (low-severity alerts never mask critical ones)
CATEGORY_COOLDOWNS_MS: Dict[str, int] = {
    "Phone / Device Detected": 8_000,
    "Multi-Face Detection": 8_000,
    "Liveness / Spoof Risk": 12_000,
    "Frozen Video Feed": 15_000,
    "Camera Occlusion": 10_000,
    "Face Absent From Frame": 10_000,
    "Gaze Deviation (Looking Away)": 6_000,
    "Split-Angle Off-Screen Focus": 7_000,
    "Repetitive Off-Screen Peek Pattern": 15_000,
    "Teleprompter Reading Pattern": 12_000,
    "Head Pose Shift": 8_000,
}

SEVERITY_RANK: Dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}

# Calibrated deviation & anti-poisoning limits
CALIBRATION_TARGET_FRAMES = 15
CALIBRATION_MAX_BUFFER = 45           # Safety cap if early frames are turbulent
CALIBRATION_MAX_GAZE_MAD = 0.14       # Rejects fidgeting/peeking during startup lock
MAX_BASELINE_GAZE_OFFSET = 0.22
MAX_BASELINE_POSE_OFFSET = 14.0
BASELINE_DRIFT_ALPHA = 0.015          # Ultra-slow adaptation ONLY on strictly centered frames

GAZE_X_DELTA_THRESHOLD = 0.34
GAZE_Y_DOWN_DELTA_THRESHOLD = 0.30
HEAD_YAW_DELTA_THRESHOLD = 22.0
HEAD_PITCH_DELTA_THRESHOLD = 20.0
EYE_OFFSET_TO_DEGREES = 35.0          # Approx eye-in-head visual angle scaling
SPLIT_ANGLE_YAW_DEG_THRESHOLD = 25.0
SPLIT_ANGLE_PITCH_DEG_THRESHOLD = 22.0

LIVENESS_MIN_THRESHOLD = 0.45
LIVENESS_CLIFF_DROP_DELTA = 0.35      # Relative drop vs healthy EMA (catches deepfake glitches)

# Tier 2 Gemini VLM gating (protects free-tier quota)
VLM_COOLDOWN_MS = 30_000
VLM_MAX_TRIGGERS_PER_SESSION = 20

PROHIBITED_DEVICE_TOKENS = (
    "phone",
    "cell phone",
    "smartphone",
    "mobile",
    "tablet",
    "ipad",
    "earpiece",
)
SECONDARY_AID_TOKENS = ("notes", "cheat sheet", "book", "notebook", "second screen")

CATEGORY_TO_ANOMALY_TYPE: Dict[str, VisionAnomalyType] = {
    "Phone / Device Detected": VisionAnomalyType.DEVICE_DETECTED,
    "Multi-Face Detection": VisionAnomalyType.MULTI_FACE,
    "Liveness / Spoof Risk": VisionAnomalyType.LIVENESS_SPOOF,
    "Frozen Video Feed": VisionAnomalyType.LIVENESS_SPOOF,
    "Camera Occlusion": VisionAnomalyType.CAMERA_OCCLUSION,
    "Face Absent From Frame": VisionAnomalyType.CAMERA_OCCLUSION,
    "Gaze Deviation (Looking Away)": VisionAnomalyType.GAZE_DEVIATION,
    "Split-Angle Off-Screen Focus": VisionAnomalyType.GAZE_DEVIATION,
    "Repetitive Off-Screen Peek Pattern": VisionAnomalyType.GAZE_DEVIATION,
    "Teleprompter Reading Pattern": VisionAnomalyType.GAZE_DEVIATION,
    "Head Pose Shift": VisionAnomalyType.HEAD_PITCH_DOWN,
}


class VisionTelemetryFrame(BaseModel):
    session_id: str
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    camera_active: bool = True
    faces_detected: int = 1
    phone_detected: bool = False
    objects_detected: List[str] = Field(default_factory=list)
    gaze_direction: str = "center"  # center, left, right, up, down
    gaze_offset_x: float = 0.0      # normalized offset -1.0 to 1.0
    gaze_offset_y: float = 0.0      # normalized offset -1.0 to 1.0
    head_pose_yaw: float = 0.0      # degrees (-90 to +90)
    head_pose_pitch: float = 0.0    # degrees (-90 to +90)
    head_pose_roll: float = 0.0     # degrees (-90 to +90)
    liveness_score: float = 0.98    # 0.0 to 1.0


class VisionAnomalyEvent(BaseModel):
    event_id: str
    session_id: str
    timestamp_offset: str           # "HH:MM:SS"
    timestamp_ms: int
    category: str
    anomaly_type: Optional[VisionAnomalyType] = None
    signal: str
    source: str = "Vision Landmark Gaze Model"
    confidence: float
    severity: str                   # "info", "low", "medium", "high"
    duration_ms: int = 0
    requires_vlm_verification: bool = False

    @model_validator(mode="after")
    def _populate_anomaly_type(self) -> "VisionAnomalyEvent":
        if self.anomaly_type is None:
            self.anomaly_type = CATEGORY_TO_ANOMALY_TYPE.get(
                self.category, VisionAnomalyType.GAZE_DEVIATION
            )
        return self


@dataclass
class _AnomalyStreak:
    start_ms: Optional[int] = None
    last_seen_ms: Optional[int] = None
    frame_count: int = 0
    peak_magnitude: float = 0.0
    last_emitted_ms: int = -1

    def update(self, active: bool, ts_ms: int, magnitude: float = 0.0) -> Tuple[int, int]:
        if not active:
            if self.last_seen_ms is not None and (ts_ms - self.last_seen_ms) > STREAK_GAP_TOLERANCE_MS:
                self.reset()
            return (0, 0)

        if self.start_ms is None or (
            self.last_seen_ms is not None and (ts_ms - self.last_seen_ms) > STREAK_GAP_TOLERANCE_MS
        ):
            self.start_ms = ts_ms
            self.frame_count = 1
            self.peak_magnitude = magnitude
        else:
            self.frame_count += 1
            if magnitude > self.peak_magnitude:
                self.peak_magnitude = magnitude

        self.last_seen_ms = ts_ms
        return (max(0, ts_ms - self.start_ms), self.frame_count)

    def can_emit(self, ts_ms: int, cooldown_ms: int) -> bool:
        return self.last_emitted_ms < 0 or (ts_ms - self.last_emitted_ms) >= cooldown_ms

    def mark_emitted(self, ts_ms: int) -> None:
        self.last_emitted_ms = ts_ms

    def reset(self) -> None:
        self.start_ms = None
        self.last_seen_ms = None
        self.frame_count = 0
        self.peak_magnitude = 0.0


@dataclass
class _SessionVisionState:
    session_id: str
    start_ts_ms: int
    last_ts_ms: int = 0
    event_seq: int = 0

    # O(1) running counters (accurate across multi-hour calls with bounded RAM)
    total_frames: int = 0
    camera_active_frames: int = 0
    face_present_frames: int = 0
    center_gaze_frames: int = 0
    multi_face_frames: int = 0
    low_liveness_frames: int = 0

    recent_frames: Deque[VisionTelemetryFrame] = field(
        default_factory=lambda: deque(maxlen=FRAME_RING_BUFFER_SIZE)
    )
    # Compact (timestamp_ms, is_offscreen) ring for peek-loop duty-cycle forensics
    gaze_history: Deque[Tuple[int, bool]] = field(
        default_factory=lambda: deque(maxlen=FRAME_RING_BUFFER_SIZE)
    )
    events: List[VisionAnomalyEvent] = field(default_factory=list)
    streaks: Dict[str, _AnomalyStreak] = field(default_factory=dict)

    # Online resting-posture calibration + neurodiversity variance tracking
    cal_gaze_x: List[float] = field(default_factory=list)
    cal_gaze_y: List[float] = field(default_factory=list)
    cal_yaw: List[float] = field(default_factory=list)
    cal_pitch: List[float] = field(default_factory=list)
    baseline_locked: bool = False
    base_gaze_x: float = 0.0
    base_gaze_y: float = 0.0
    base_yaw: float = 0.0
    base_pitch: float = 0.0
    baseline_gaze_variance: float = 0.0

    # Healthy liveness EMA for relative deepfake cliff-drop detection
    liveness_ema: float = 0.95

    # VLM budget & throttling
    vlm_triggers_sent: int = 0
    last_vlm_trigger_ms: int = -1

    def streak(self, key: str) -> _AnomalyStreak:
        if key not in self.streaks:
            self.streaks[key] = _AnomalyStreak()
        return self.streaks[key]


class VisionProctoringService:
    """
    Forensic vision proctoring state machine:
      - Bounded ring-buffer memory + LRU session eviction
      - Stability-gated resting calibration + centered-only slow drift adaptation
      - VOR-aware Split-Angle head+eye compounding detector
      - Anti-debouncer 20s Peek-Loop off-screen duty-cycle detector
      - Smoothed horizontal teleprompter saccade + sawtooth asymmetry analyzer
      - Micro-motion variance frozen-feed & relative liveness cliff-drop anti-spoofing
      - Budget-capped Tier 2 Gemini VLM escalation gating
    """

    def __init__(self, max_sessions: int = MAX_ACTIVE_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionVisionState] = OrderedDict()

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def ingest_frame(self, frame: VisionTelemetryFrame) -> Optional[VisionAnomalyEvent]:
        emitted = self.ingest_frame_all(frame)
        if not emitted:
            return None
        return max(emitted, key=lambda e: (SEVERITY_RANK.get(e.severity, 0), e.confidence))

    def ingest_frame_all(self, frame: VisionTelemetryFrame) -> List[VisionAnomalyEvent]:
        clean = self._sanitize_frame(frame)
        state = self._get_or_create_state(clean.session_id, clean.timestamp_ms)

        if state.total_frames > 0 and clean.timestamp_ms < (state.last_ts_ms - 5_000):
            logger.debug(
                "Dropped stale vision frame for %s (ts=%d < head=%d)",
                clean.session_id,
                clean.timestamp_ms,
                state.last_ts_ms,
            )
            return []

        if clean.timestamp_ms < state.start_ts_ms:
            state.start_ts_ms = clean.timestamp_ms
        if clean.timestamp_ms > state.last_ts_ms:
            state.last_ts_ms = clean.timestamp_ms

        state.total_frames += 1
        state.recent_frames.append(clean)

        # Online resting-posture calibration + slow centered-only drift compensation
        self._update_calibration(state, clean)

        rel_gaze_x = clean.gaze_offset_x - state.base_gaze_x
        rel_gaze_y = clean.gaze_offset_y - state.base_gaze_y
        rel_yaw = clean.head_pose_yaw - state.base_yaw
        rel_pitch = clean.head_pose_pitch - state.base_pitch

        if clean.camera_active:
            state.camera_active_frames += 1
            if clean.faces_detected >= 1:
                state.face_present_frames += 1
        if clean.faces_detected > 1:
            state.multi_face_frames += 1

        # Evaluate gaze & VOR-aware split-angle compounding
        is_gaze_deviated, gaze_dir_label = self._evaluate_gaze_state(
            clean, rel_gaze_x, rel_gaze_y, state.baseline_locked
        )
        is_split_angle, split_dir_label, split_deg = self._evaluate_split_angle(
            rel_gaze_x, rel_gaze_y, rel_yaw, rel_pitch
        )
        is_offscreen = is_gaze_deviated or is_split_angle

        if clean.camera_active and clean.faces_detected >= 1:
            state.gaze_history.append((clean.timestamp_ms, is_offscreen))
            if not is_offscreen:
                state.center_gaze_frames += 1
                self._adapt_centered_baseline(state, clean, rel_gaze_x, rel_gaze_y, rel_yaw, rel_pitch)

        ts_ms = clean.timestamp_ms
        ts_offset = self._format_offset(ts_ms - state.start_ts_ms)
        emitted: List[VisionAnomalyEvent] = []

        # -------------------------------------------------------------------
        # Channel 1: Secondary Phone / Prohibited Electronic Device Detection
        # -------------------------------------------------------------------
        matched_devices, matched_aids = self._classify_detected_objects(clean)
        device_active = clean.phone_detected or bool(matched_devices)
        dev_dur, dev_frames = state.streak("device").update(device_active, ts_ms, 1.0)
        if device_active and (dev_frames >= 2 or dev_dur >= DEVICE_PERSISTENCE_MS):
            cat = "Phone / Device Detected"
            if state.streak("device").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                labels = ", ".join(matched_devices) if matched_devices else "phone/mobile device"
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"Secondary electronic device ({labels}) persisted in frame ({dev_dur / 1000:.1f}s)",
                        source="Object / VLM Detector",
                        confidence=0.95 if dev_frames >= 2 else 0.88,
                        severity="high",
                        duration_ms=dev_dur,
                        vlm_eligible=True,
                    )
                )
                state.streak("device").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 2: Multi-Face Detection (Second Person Coaching)
        # -------------------------------------------------------------------
        multi_active = clean.camera_active and clean.faces_detected > 1
        mf_dur, mf_frames = state.streak("multi_face").update(
            multi_active, ts_ms, float(clean.faces_detected)
        )
        if multi_active and (mf_dur >= MULTI_FACE_PERSISTENCE_MS or mf_frames >= 3):
            cat = "Multi-Face Detection"
            if state.streak("multi_face").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"{clean.faces_detected} faces detected continuously for {mf_dur / 1000:.1f}s",
                        source="MediaPipe Multi-Face Detector",
                        confidence=min(0.97, 0.86 + 0.03 * min(mf_frames, 3)),
                        severity="high",
                        duration_ms=mf_dur,
                        vlm_eligible=True,
                    )
                )
                state.streak("multi_face").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 3: Anti-Spoofing (Hard Floor + Relative Liveness Cliff Drop)
        # -------------------------------------------------------------------
        liveness_drop = state.liveness_ema - clean.liveness_score
        hard_spoof = clean.liveness_score < LIVENESS_MIN_THRESHOLD
        cliff_spoof = liveness_drop >= LIVENESS_CLIFF_DROP_DELTA and clean.liveness_score < 0.70
        spoof_active = clean.camera_active and clean.faces_detected >= 1 and (hard_spoof or cliff_spoof)

        if clean.camera_active and clean.faces_detected >= 1:
            if hard_spoof or cliff_spoof:
                state.low_liveness_frames += 1
            elif clean.liveness_score >= 0.75:
                state.liveness_ema = (0.12 * clean.liveness_score) + (0.88 * state.liveness_ema)

        sp_dur, sp_frames = state.streak("liveness").update(
            spoof_active, ts_ms, max(1.0 - clean.liveness_score, liveness_drop)
        )
        if spoof_active and sp_dur >= LIVENESS_PERSISTENCE_MS and sp_frames >= 2:
            cat = "Liveness / Spoof Risk"
            if state.streak("liveness").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                reason_tag = (
                    f"low liveness ({clean.liveness_score:.2f})"
                    if hard_spoof
                    else f"abrupt liveness drop ({state.liveness_ema:.2f} -> {clean.liveness_score:.2f})"
                )
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"Sustained {reason_tag} for {sp_dur / 1000:.1f}s (virtual camera / replay risk)",
                        source="Anti-Spoof Liveness Engine",
                        confidence=round(min(0.95, 0.72 + (1.0 - clean.liveness_score) * 0.32), 2),
                        severity="high",
                        duration_ms=sp_dur,
                        vlm_eligible=True,
                    )
                )
                state.streak("liveness").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 4: Micro-Motion Variance Frozen / Photo-Replay Feed Check
        # -------------------------------------------------------------------
        frozen_active, frozen_span_ms, frozen_count = self._detect_frozen_feed(state, ts_ms)
        if frozen_active:
            cat = "Frozen Video Feed"
            if state.streak("frozen").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Near-zero 3D pose micro-motion across {frozen_count} frames "
                            f"({frozen_span_ms / 1000:.1f}s); feed may be frozen or static replay"
                        ),
                        source="Temporal Micro-Motion Monitor",
                        confidence=0.90,
                        severity="medium",
                        duration_ms=frozen_span_ms,
                        vlm_eligible=False,
                    )
                )
                state.streak("frozen").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 5: Camera Occlusion & Face Absence
        # -------------------------------------------------------------------
        cam_off_dur, _ = state.streak("camera_off").update(not clean.camera_active, ts_ms, 1.0)
        if not clean.camera_active and cam_off_dur >= CAMERA_OFF_PERSISTENCE_MS:
            cat = "Camera Occlusion"
            if state.streak("camera_off").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"Webcam video stream paused or disabled ({cam_off_dur / 1000:.1f}s)",
                        source="Camera Stream Monitor",
                        confidence=0.99,
                        severity="medium",
                        duration_ms=cam_off_dur,
                        vlm_eligible=False,
                    )
                )
                state.streak("camera_off").mark_emitted(ts_ms)

        face_absent = clean.camera_active and clean.faces_detected == 0
        fa_dur, fa_frames = state.streak("face_absent").update(face_absent, ts_ms, 1.0)
        if face_absent and fa_dur >= FACE_ABSENT_PERSISTENCE_MS and fa_frames >= 2:
            cat = "Face Absent From Frame"
            if state.streak("face_absent").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"No candidate face visible while camera is active ({fa_dur / 1000:.1f}s)",
                        source="MediaPipe Face Detector",
                        confidence=0.92,
                        severity="medium",
                        duration_ms=fa_dur,
                        vlm_eligible=False,
                    )
                )
                state.streak("face_absent").mark_emitted(ts_ms)

        if not clean.camera_active or clean.faces_detected == 0:
            state.streak("gaze").update(False, ts_ms)
            state.streak("split_angle").update(False, ts_ms)
            state.streak("head_pose").update(False, ts_ms)
            return emitted

        # -------------------------------------------------------------------
        # Channel 6: Baseline-Relative Sustained Gaze Deviation (>= 3.0s)
        # -------------------------------------------------------------------
        gaze_mag = max(abs(rel_gaze_x), abs(rel_gaze_y))
        gz_dur, gz_frames = state.streak("gaze").update(is_gaze_deviated, ts_ms, gaze_mag)
        if is_gaze_deviated and gz_dur >= GAZE_PERSISTENCE_MS and gz_frames >= 2:
            cat = "Gaze Deviation (Looking Away)"
            if state.streak("gaze").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                is_down = gaze_dir_label.startswith("DOWN")
                severity = "high" if (is_down and (gz_dur >= 4_500 or bool(matched_aids))) else "medium"
                aid_note = f"; nearby visual cue: {', '.join(matched_aids)}" if matched_aids else ""
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Sustained off-screen gaze directed {gaze_dir_label} "
                            f"for {gz_dur / 1000:.1f}s (Δx={rel_gaze_x:+.2f}, Δy={rel_gaze_y:+.2f}){aid_note}"
                        ),
                        source="Calibrated Iris Gaze Model",
                        confidence=round(min(0.95, 0.74 + min(gaze_mag, 0.7) * 0.28), 2),
                        severity=severity,
                        duration_ms=gz_dur,
                        vlm_eligible=is_down or gaze_mag >= 0.50,
                    )
                )
                state.streak("gaze").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 7: VOR-Aware Split-Angle Head+Eye Compounding (>= 2.6s)
        # Catches candidates staying just under individual yaw and iris thresholds
        # -------------------------------------------------------------------
        split_only = is_split_angle and not is_gaze_deviated
        sa_dur, sa_frames = state.streak("split_angle").update(split_only, ts_ms, split_deg)
        if split_only and sa_dur >= SPLIT_ANGLE_PERSISTENCE_MS and sa_frames >= 2:
            cat = "Split-Angle Off-Screen Focus"
            if state.streak("split_angle").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                is_down = split_dir_label == "DOWN"
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Compounded head+eye off-axis focus ({split_dir_label}, ~{split_deg:.0f}° combined) "
                            f"for {sa_dur / 1000:.1f}s (ΔYaw={rel_yaw:+.1f}°, ΔPitch={rel_pitch:+.1f}°, "
                            f"Δx={rel_gaze_x:+.2f}, Δy={rel_gaze_y:+.2f})"
                        ),
                        source="VOR-Aware Gaze+Pose Fusion",
                        confidence=round(min(0.93, 0.76 + min((split_deg - 20.0) / 40.0, 0.17)), 2),
                        severity="medium",
                        duration_ms=sa_dur,
                        vlm_eligible=is_down or split_deg >= 32.0,
                    )
                )
                state.streak("split_angle").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 8: Anti-Debouncer 20s Peek-Loop Duty-Cycle Detector
        # Catches repeated 1.5-2.5s glances that individually reset the 3s timer
        # -------------------------------------------------------------------
        peek_hit, peek_transitions, peek_ratio = self._detect_peek_loop(state, ts_ms)
        if peek_hit:
            cat = "Repetitive Off-Screen Peek Pattern"
            if state.streak("peek_loop").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Repeated off-screen peek cycle detected ({peek_transitions} glance transitions, "
                            f"{int(peek_ratio * 100)}% off-screen across last {PEEK_WINDOW_MS // 1000}s)"
                        ),
                        source="Off-Screen Duty-Cycle Analyzer",
                        confidence=round(min(0.92, 0.76 + max(0.0, peek_ratio - 0.35) * 0.40), 2),
                        severity="medium" if peek_ratio < 0.60 else "high",
                        duration_ms=PEEK_WINDOW_MS,
                        vlm_eligible=is_offscreen,
                    )
                )
                state.streak("peek_loop").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 9: Horizontal Teleprompter Saccade & Sawtooth Analyzer
        # -------------------------------------------------------------------
        scan_detected, scan_sweeps, has_sawtooth = self._detect_teleprompter_scanning(state, ts_ms)
        if scan_detected:
            cat = "Teleprompter Reading Pattern"
            if state.streak("teleprompter").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                style_note = " with carriage-return sawtooth asymmetry" if has_sawtooth else ""
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Rhythmic horizontal eye-scanning pattern detected "
                            f"({scan_sweeps} sweeps across ~8s, flat vertical band{style_note})"
                        ),
                        source="Temporal Gaze Saccade Analyzer",
                        confidence=0.89 if has_sawtooth else 0.81,
                        severity="medium",
                        duration_ms=8_000,
                        vlm_eligible=False,
                    )
                )
                state.streak("teleprompter").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 10: Baseline-Relative Sustained Head Pose Shift (>= 3.0s)
        # -------------------------------------------------------------------
        pose_deviated = (
            abs(rel_yaw) > HEAD_YAW_DELTA_THRESHOLD
            or abs(rel_pitch) > HEAD_PITCH_DELTA_THRESHOLD
        )
        pose_mag = max(abs(rel_yaw), abs(rel_pitch))
        hp_dur, hp_frames = state.streak("head_pose").update(pose_deviated, ts_ms, pose_mag)
        if pose_deviated and hp_dur >= HEAD_POSE_PERSISTENCE_MS and hp_frames >= 2:
            cat = "Head Pose Shift"
            if state.streak("head_pose").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Sustained head rotation for {hp_dur / 1000:.1f}s "
                            f"(ΔYaw={rel_yaw:+.1f}°, ΔPitch={rel_pitch:+.1f}°)"
                        ),
                        source="Calibrated Head Pose Estimator",
                        confidence=round(min(0.92, 0.75 + min(pose_mag / 60.0, 0.17)), 2),
                        severity="medium" if rel_pitch < -24.0 else "low",
                        duration_ms=hp_dur,
                        vlm_eligible=rel_pitch < -24.0,
                    )
                )
                state.streak("head_pose").mark_emitted(ts_ms)

        return emitted

    def get_events_in_window(
        self,
        session_id: str,
        start_ms: int,
        end_ms: int,
    ) -> List[VisionAnomalyEvent]:
        state = self._sessions.get(session_id)
        if not state:
            return []
        return [e for e in state.events if start_ms <= e.timestamp_ms <= end_ms]

    def get_session_summary(
        self,
        session_id: str,
        allow_demo_mock: bool = False,
    ) -> Dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None or state.total_frames == 0:
            if allow_demo_mock:
                return self._build_demo_mock_summary(session_id)
            return {
                "data_status": "NO_TELEMETRY",
                "total_frames_analyzed": 0,
                "visual_attention_score": None,
                "gaze_consistency_pct": None,
                "face_coverage_pct": None,
                "offscreen_duty_cycle_pct": None,
                "calibration_complete": False,
                "baseline_gaze_variance": 0.0,
                "baseline_pose": None,
                "phone_events_count": 0,
                "multi_face_events_count": 0,
                "gaze_deviation_events_count": 0,
                "split_angle_events_count": 0,
                "peek_loop_events_count": 0,
                "teleprompter_scan_events_count": 0,
                "liveness_risk_events_count": 0,
                "face_absent_events_count": 0,
                "camera_off_events_count": 0,
                "vlm_triggers_sent": 0,
                "anomaly_events": [],
            }

        total = state.total_frames
        active_face_denom = max(state.face_present_frames, 1)

        gaze_consistency = round((state.center_gaze_frames / active_face_denom) * 100.0, 1)
        offscreen_duty_cycle = round(100.0 - gaze_consistency, 1)
        face_coverage = round((state.face_present_frames / total) * 100.0, 1)

        events = state.events
        phone_events = sum(1 for e in events if e.category == "Phone / Device Detected")
        multi_face_events = sum(1 for e in events if e.category == "Multi-Face Detection")
        gaze_dev_events = sum(1 for e in events if e.category == "Gaze Deviation (Looking Away)")
        split_angle_events = sum(1 for e in events if e.category == "Split-Angle Off-Screen Focus")
        peek_loop_events = sum(1 for e in events if e.category == "Repetitive Off-Screen Peek Pattern")
        teleprompter_events = sum(1 for e in events if e.category == "Teleprompter Reading Pattern")
        liveness_events = sum(1 for e in events if e.category in ("Liveness / Spoof Risk", "Frozen Video Feed"))
        face_absent_events = sum(1 for e in events if e.category == "Face Absent From Frame")
        camera_off_events = sum(1 for e in events if e.category == "Camera Occlusion")

        penalty = (
            min(phone_events * 22.0, 44.0)
            + min(multi_face_events * 16.0, 32.0)
            + min(liveness_events * 18.0, 36.0)
            + min((gaze_dev_events + split_angle_events) * 5.0, 25.0)
            + min(peek_loop_events * 9.0, 27.0)
            + min(teleprompter_events * 8.0, 24.0)
            + min((camera_off_events + face_absent_events) * 6.0, 18.0)
        )
        if face_coverage < 80.0:
            penalty += (80.0 - face_coverage) * 0.25

        visual_attention = round(max(0.0, min(100.0, 100.0 - penalty)), 1)

        return {
            "data_status": "OK",
            "total_frames_analyzed": total,
            "visual_attention_score": visual_attention,
            "gaze_consistency_pct": gaze_consistency,
            "face_coverage_pct": face_coverage,
            "offscreen_duty_cycle_pct": offscreen_duty_cycle,
            "calibration_complete": state.baseline_locked,
            "baseline_gaze_variance": round(state.baseline_gaze_variance, 4),
            "baseline_pose": {
                "gaze_x": round(state.base_gaze_x, 3),
                "gaze_y": round(state.base_gaze_y, 3),
                "yaw": round(state.base_yaw, 1),
                "pitch": round(state.base_pitch, 1),
            },
            "phone_events_count": phone_events,
            "multi_face_events_count": multi_face_events,
            "gaze_deviation_events_count": gaze_dev_events + split_angle_events + peek_loop_events,
            "split_angle_events_count": split_angle_events,
            "peek_loop_events_count": peek_loop_events,
            "teleprompter_scan_events_count": teleprompter_events,
            "liveness_risk_events_count": liveness_events,
            "face_absent_events_count": face_absent_events,
            "camera_off_events_count": camera_off_events,
            "vlm_triggers_sent": state.vlm_triggers_sent,
            "anomaly_events": [e.model_dump(mode="json") for e in events],
        }

    # -----------------------------------------------------------------------
    # Internal Forensic Engines
    # -----------------------------------------------------------------------
    def _get_or_create_state(self, session_id: str, ts_ms: int) -> _SessionVisionState:
        state = self._sessions.get(session_id)
        if state is not None:
            self._sessions.move_to_end(session_id)
            return state

        state = _SessionVisionState(session_id=session_id, start_ts_ms=ts_ms, last_ts_ms=ts_ms)
        self._sessions[session_id] = state
        if len(self._sessions) > self._max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            logger.info("Evicted oldest vision session '%s' (LRU cap=%d).", evicted_id, self._max_sessions)
        return state

    @staticmethod
    def _sanitize_frame(frame: VisionTelemetryFrame) -> VisionTelemetryFrame:
        def _clamp(val: float, lo: float, hi: float, default: float = 0.0) -> float:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return default
            return max(lo, min(hi, float(val)))

        valid_dirs = {"center", "left", "right", "up", "down"}
        direction = str(frame.gaze_direction or "center").strip().lower()
        if direction not in valid_dirs:
            direction = "center"

        return frame.model_copy(
            update={
                "timestamp_ms": max(0, int(frame.timestamp_ms)),
                "faces_detected": max(0, min(20, int(frame.faces_detected))),
                "gaze_direction": direction,
                "gaze_offset_x": _clamp(frame.gaze_offset_x, -1.0, 1.0, 0.0),
                "gaze_offset_y": _clamp(frame.gaze_offset_y, -1.0, 1.0, 0.0),
                "head_pose_yaw": _clamp(frame.head_pose_yaw, -90.0, 90.0, 0.0),
                "head_pose_pitch": _clamp(frame.head_pose_pitch, -90.0, 90.0, 0.0),
                "head_pose_roll": _clamp(frame.head_pose_roll, -90.0, 90.0, 0.0),
                "liveness_score": _clamp(frame.liveness_score, 0.0, 1.0, 0.98),
            }
        )

    @staticmethod
    def _update_calibration(state: _SessionVisionState, frame: VisionTelemetryFrame) -> None:
        if state.baseline_locked:
            return
        if not frame.camera_active or frame.faces_detected != 1 or frame.phone_detected:
            return
        if abs(frame.head_pose_yaw) > 30.0 or abs(frame.head_pose_pitch) > 30.0:
            return

        state.cal_gaze_x.append(frame.gaze_offset_x)
        state.cal_gaze_y.append(frame.gaze_offset_y)
        state.cal_yaw.append(frame.head_pose_yaw)
        state.cal_pitch.append(frame.head_pose_pitch)

        n = len(state.cal_gaze_x)
        if n < CALIBRATION_TARGET_FRAMES:
            return

        med_x = statistics.median(state.cal_gaze_x)
        mad_x = statistics.median([abs(v - med_x) for v in state.cal_gaze_x])

        # Wait for stable resting window (blocks startup calibration poisoning) unless buffer cap reached
        if mad_x > CALIBRATION_MAX_GAZE_MAD and n < CALIBRATION_MAX_BUFFER:
            return

        med_y = statistics.median(state.cal_gaze_y)
        state.base_gaze_x = max(-MAX_BASELINE_GAZE_OFFSET, min(MAX_BASELINE_GAZE_OFFSET, med_x))
        state.base_gaze_y = max(-MAX_BASELINE_GAZE_OFFSET, min(MAX_BASELINE_GAZE_OFFSET, med_y))
        state.base_yaw = max(
            -MAX_BASELINE_POSE_OFFSET, min(MAX_BASELINE_POSE_OFFSET, statistics.median(state.cal_yaw))
        )
        state.base_pitch = max(
            -MAX_BASELINE_POSE_OFFSET, min(MAX_BASELINE_POSE_OFFSET, statistics.median(state.cal_pitch))
        )
        state.baseline_gaze_variance = float(
            statistics.pvariance(state.cal_gaze_x) + statistics.pvariance(state.cal_gaze_y)
        )
        state.baseline_locked = True
        state.cal_gaze_x.clear()
        state.cal_gaze_y.clear()
        state.cal_yaw.clear()
        state.cal_pitch.clear()

    @staticmethod
    def _adapt_centered_baseline(
        state: _SessionVisionState,
        frame: VisionTelemetryFrame,
        rel_x: float,
        rel_y: float,
        rel_yaw: float,
        rel_pitch: float,
    ) -> None:
        """Ultra-slow EMA posture drift compensation ONLY when candidate is firmly centered."""
        if not state.baseline_locked or frame.faces_detected != 1 or frame.liveness_score < 0.70:
            return
        if abs(rel_x) > 0.12 or abs(rel_y) > 0.12 or abs(rel_yaw) > 10.0 or abs(rel_pitch) > 10.0:
            return

        a = BASELINE_DRIFT_ALPHA
        state.base_gaze_x = max(
            -MAX_BASELINE_GAZE_OFFSET,
            min(MAX_BASELINE_GAZE_OFFSET, (1.0 - a) * state.base_gaze_x + a * frame.gaze_offset_x),
        )
        state.base_gaze_y = max(
            -MAX_BASELINE_GAZE_OFFSET,
            min(MAX_BASELINE_GAZE_OFFSET, (1.0 - a) * state.base_gaze_y + a * frame.gaze_offset_y),
        )
        state.base_yaw = max(
            -MAX_BASELINE_POSE_OFFSET,
            min(MAX_BASELINE_POSE_OFFSET, (1.0 - a) * state.base_yaw + a * frame.head_pose_yaw),
        )
        state.base_pitch = max(
            -MAX_BASELINE_POSE_OFFSET,
            min(MAX_BASELINE_POSE_OFFSET, (1.0 - a) * state.base_pitch + a * frame.head_pose_pitch),
        )

    @staticmethod
    def _evaluate_gaze_state(
        frame: VisionTelemetryFrame,
        rel_x: float,
        rel_y: float,
        baseline_locked: bool = False,
    ) -> Tuple[bool, str]:
        has_numeric_offsets = (
            baseline_locked
            or abs(frame.gaze_offset_x) > 1e-4
            or abs(frame.gaze_offset_y) > 1e-4
        )
        if has_numeric_offsets:
            if rel_y > GAZE_Y_DOWN_DELTA_THRESHOLD:
                return (True, "DOWN (Lap / Secondary Device Area)")
            if abs(rel_x) > GAZE_X_DELTA_THRESHOLD:
                return (True, "RIGHT" if rel_x > 0 else "LEFT")
            return (False, "CENTER")

        if frame.gaze_direction == "down":
            return (True, "DOWN (Lap / Secondary Device Area)")
        if frame.gaze_direction in ("left", "right"):
            return (True, frame.gaze_direction.upper())
        return (False, "CENTER")

    @staticmethod
    def _evaluate_split_angle(
        rel_x: float,
        rel_y: float,
        rel_yaw: float,
        rel_pitch: float,
    ) -> Tuple[bool, str, float]:
        """
        VOR-aware head+eye vector compounding:
          - Opposite-sign head/eye motion (turning head while keeping eyes on webcam) cancels out.
          - Same-direction head+eye offset (split-angle bypass toward a 2nd monitor or lap) sums up.
        """
        # Horizontal split-angle (requires same direction to respect vestibulo-ocular reflex)
        if (rel_yaw * rel_x) > 0.0 and abs(rel_yaw) >= 12.0 and abs(rel_x) >= 0.20:
            horiz_deg = abs(rel_yaw) + (abs(rel_x) * EYE_OFFSET_TO_DEGREES)
            if horiz_deg >= SPLIT_ANGLE_YAW_DEG_THRESHOLD:
                return (True, "RIGHT" if rel_x > 0 else "LEFT", horiz_deg)

        # Vertical downward split-angle (head tilted down + eyes shifted down toward lap/phone)
        if rel_pitch <= -13.0 and rel_y >= 0.20:
            vert_deg = abs(rel_pitch) + (rel_y * EYE_OFFSET_TO_DEGREES)
            if vert_deg >= SPLIT_ANGLE_PITCH_DEG_THRESHOLD:
                return (True, "DOWN", vert_deg)

        return (False, "CENTER", 0.0)

    @staticmethod
    def _detect_peek_loop(state: _SessionVisionState, now_ms: int) -> Tuple[bool, int, float]:
        window = [
            (ts, off) for ts, off in state.gaze_history if (now_ms - ts) <= PEEK_WINDOW_MS
        ]
        if len(window) < PEEK_MIN_FRAMES:
            return (False, 0, 0.0)

        span_ms = window[-1][0] - window[0][0]
        if span_ms < PEEK_MIN_SPAN_MS:
            return (False, 0, 0.0)

        off_count = sum(1 for _, off in window if off)
        off_ratio = off_count / len(window)

        transitions = 0
        prev_off = window[0][1]
        for _, off in window[1:]:
            if off and not prev_off:
                transitions += 1
            prev_off = off

        return (
            transitions >= PEEK_MIN_TRANSITIONS and off_ratio >= PEEK_MIN_OFFSCREEN_RATIO,
            transitions,
            off_ratio,
        )

    @staticmethod
    def _detect_frozen_feed(state: _SessionVisionState, now_ms: int) -> Tuple[bool, int, int]:
        """
        Micro-motion variance detector: catches exact frozen frames AND low-noise photo/loop replays,
        while ignoring zeroed-Euler synthetic stubs (yaw=pitch=roll=0.0).
        """
        window = [
            f
            for f in state.recent_frames
            if (now_ms - f.timestamp_ms) <= (FROZEN_FEED_PERSISTENCE_MS + 1_500)
            and f.camera_active
            and f.faces_detected >= 1
        ]
        if len(window) < 6:
            return (False, 0, 0)

        span_ms = window[-1].timestamp_ms - window[0].timestamp_ms
        if span_ms < FROZEN_FEED_PERSISTENCE_MS:
            return (False, 0, 0)

        # Exclude gaze-only synthetic stubs that leave all 3D Euler angles at exact 0.0
        has_3d_pose = any(
            abs(f.head_pose_yaw) > 1e-4 or abs(f.head_pose_pitch) > 1e-4 or abs(f.head_pose_roll) > 1e-4
            for f in window
        )
        if not has_3d_pose:
            return (False, 0, 0)

        yaw_std = statistics.pstdev([f.head_pose_yaw for f in window])
        pitch_std = statistics.pstdev([f.head_pose_pitch for f in window])
        gaze_x_std = statistics.pstdev([f.gaze_offset_x for f in window])
        gaze_y_std = statistics.pstdev([f.gaze_offset_y for f in window])

        is_frozen = (
            yaw_std < 0.05
            and pitch_std < 0.05
            and gaze_x_std < 0.0025
            and gaze_y_std < 0.0025
        )
        return (is_frozen, span_ms, len(window))

    @staticmethod
    def _classify_detected_objects(frame: VisionTelemetryFrame) -> Tuple[List[str], List[str]]:
        devices: List[str] = []
        aids: List[str] = []
        for raw_obj in frame.objects_detected:
            obj = str(raw_obj).strip().lower()
            if not obj:
                continue
            if any(tok in obj for tok in PROHIBITED_DEVICE_TOKENS):
                devices.append(obj)
            elif any(tok in obj for tok in SECONDARY_AID_TOKENS):
                aids.append(obj)
        return (list(dict.fromkeys(devices)), list(dict.fromkeys(aids)))

    @staticmethod
    def _detect_teleprompter_scanning(
        state: _SessionVisionState,
        now_ms: int,
    ) -> Tuple[bool, int, bool]:
        """
        Detects rhythmic horizontal eye sweeps (reading scrolled LLM text on-screen)
        across the last 8s while vertical eye band stays flat, plus sawtooth carriage-return asymmetry.
        """
        window = [
            f
            for f in state.recent_frames
            if (now_ms - f.timestamp_ms) <= 8_000 and f.camera_active and f.faces_detected == 1
        ]
        if len(window) < 8:
            return (False, 0, False)

        raw_xs = [f.gaze_offset_x - state.base_gaze_x for f in window]
        ys = [f.gaze_offset_y - state.base_gaze_y for f in window]

        if statistics.pstdev(ys) > 0.09:
            return (False, 0, False)

        # Require meaningful horizontal scan width (filters tight sensor noise)
        if (max(raw_xs) - min(raw_xs)) < 0.22:
            return (False, 0, False)

        # Light 3-point smoothing only at higher FPS (>=10 samples in 8s) so 1 FPS sweeps stay sharp
        if len(raw_xs) >= 10:
            xs = [raw_xs[0]] + [
                0.25 * raw_xs[i - 1] + 0.50 * raw_xs[i] + 0.25 * raw_xs[i + 1]
                for i in range(1, len(raw_xs) - 1)
            ] + [raw_xs[-1]]
        else:
            xs = raw_xs

        sweeps = 0
        last_dir = 0
        anchor_x = xs[0]
        pos_steps: List[float] = []
        neg_steps: List[float] = []

        for i in range(1, len(xs)):
            step = xs[i] - xs[i - 1]
            if step > 0.02:
                pos_steps.append(step)
            elif step < -0.02:
                neg_steps.append(abs(step))

            delta = xs[i] - anchor_x
            if abs(delta) >= 0.14:
                curr_dir = 1 if delta > 0 else -1
                if last_dir != 0 and curr_dir != last_dir:
                    sweeps += 1
                last_dir = curr_dir
                anchor_x = xs[i]

        # Sawtooth reading signature: fast carriage returns in one direction vs slower tracking in the other
        has_sawtooth = False
        if pos_steps and neg_steps:
            ratio = max(statistics.fmean(pos_steps), statistics.fmean(neg_steps)) / max(
                min(statistics.fmean(pos_steps), statistics.fmean(neg_steps)), 1e-4
            )
            has_sawtooth = ratio >= 1.65

        return (sweeps >= 3, sweeps, has_sawtooth)

    def _create_event(
        self,
        state: _SessionVisionState,
        ts_ms: int,
        ts_offset: str,
        category: str,
        signal: str,
        source: str,
        confidence: float,
        severity: str,
        duration_ms: int,
        vlm_eligible: bool,
    ) -> VisionAnomalyEvent:
        state.event_seq += 1

        trigger_vlm = False
        if (
            vlm_eligible
            and state.vlm_triggers_sent < VLM_MAX_TRIGGERS_PER_SESSION
            and (state.last_vlm_trigger_ms < 0 or (ts_ms - state.last_vlm_trigger_ms) >= VLM_COOLDOWN_MS)
        ):
            trigger_vlm = True
            state.vlm_triggers_sent += 1
            state.last_vlm_trigger_ms = ts_ms

        event = VisionAnomalyEvent(
            event_id=f"v_evt_{state.event_seq:04d}",
            session_id=state.session_id,
            timestamp_offset=ts_offset,
            timestamp_ms=ts_ms,
            category=category,
            anomaly_type=CATEGORY_TO_ANOMALY_TYPE.get(category, VisionAnomalyType.GAZE_DEVIATION),
            signal=signal,
            source=source,
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            severity=severity,
            duration_ms=max(0, int(duration_ms)),
            requires_vlm_verification=trigger_vlm,
        )

        if len(state.events) < MAX_STORED_EVENTS_PER_SESSION:
            state.events.append(event)
        return event

    @staticmethod
    def _format_offset(elapsed_ms: int) -> str:
        elapsed_sec = max(0, int(elapsed_ms // 1000))
        hours = elapsed_sec // 3600
        mins = (elapsed_sec % 3600) // 60
        secs = elapsed_sec % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    @staticmethod
    def _build_demo_mock_summary(session_id: str) -> Dict[str, Any]:
        return {
            "data_status": "DEMO_MOCK",
            "total_frames_analyzed": 145,
            "visual_attention_score": 78.5,
            "gaze_consistency_pct": 82.4,
            "face_coverage_pct": 98.6,
            "offscreen_duty_cycle_pct": 17.6,
            "calibration_complete": True,
            "baseline_gaze_variance": 0.0042,
            "baseline_pose": {"gaze_x": 0.02, "gaze_y": -0.01, "yaw": 1.2, "pitch": -0.8},
            "phone_events_count": 1,
            "multi_face_events_count": 1,
            "gaze_deviation_events_count": 1,
            "split_angle_events_count": 0,
            "peek_loop_events_count": 0,
            "teleprompter_scan_events_count": 0,
            "liveness_risk_events_count": 0,
            "face_absent_events_count": 0,
            "camera_off_events_count": 0,
            "vlm_triggers_sent": 2,
            "anomaly_events": [
                {
                    "event_id": "v_evt_demo_001",
                    "session_id": session_id,
                    "timestamp_offset": "00:01:12",
                    "timestamp_ms": 72_000,
                    "category": "Gaze Deviation (Looking Away)",
                    "anomaly_type": VisionAnomalyType.GAZE_DEVIATION.value,
                    "signal": "Sustained off-screen gaze directed RIGHT for 3.6s (demo sample)",
                    "source": "Calibrated Iris Gaze Model",
                    "confidence": 0.88,
                    "severity": "medium",
                    "duration_ms": 3_600,
                    "requires_vlm_verification": False,
                },
                {
                    "event_id": "v_evt_demo_002",
                    "session_id": session_id,
                    "timestamp_offset": "00:02:45",
                    "timestamp_ms": 165_000,
                    "category": "Phone / Device Detected",
                    "anomaly_type": VisionAnomalyType.DEVICE_DETECTED.value,
                    "signal": "Secondary electronic device (phone) persisted in frame (demo sample)",
                    "source": "Object / VLM Detector",
                    "confidence": 0.94,
                    "severity": "high",
                    "duration_ms": 1_200,
                    "requires_vlm_verification": True,
                },
            ],
        }


_vision_service: Optional[VisionProctoringService] = None


def get_vision_service() -> VisionProctoringService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionProctoringService()
    return _vision_service