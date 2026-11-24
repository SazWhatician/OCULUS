import logging
import math
import statistics
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("oculus.vision.proctoring")

# ---------------------------------------------------------------------------
# Forensic Thresholds & Memory Safety Constants
# ---------------------------------------------------------------------------
MAX_ACTIVE_SESSIONS = 500
FRAME_RING_BUFFER_SIZE = 180          # ~90-180s rolling window per session (prevents OOM)
MAX_STORED_EVENTS_PER_SESSION = 500

# Temporal persistence requirements (true wall-clock milliseconds, FPS-independent)
STREAK_GAP_TOLERANCE_MS = 1_800       # Bridges transient 1-frame MediaPipe/WebRTC drops
GAZE_PERSISTENCE_MS = 3_000           # Sustained off-screen gaze required (>= 3s)
HEAD_POSE_PERSISTENCE_MS = 3_000      # Sustained head turn required (>= 3s)
CAMERA_OFF_PERSISTENCE_MS = 2_000     # Ignores brief WebRTC reconnect flickers
FACE_ABSENT_PERSISTENCE_MS = 3_000    # Candidate left frame or covered lens
MULTI_FACE_PERSISTENCE_MS = 1_500     # Filters 1-frame background false positives
DEVICE_PERSISTENCE_MS = 600           # Requires >=2 frames or 600ms (filters YOLO flicker)
LIVENESS_PERSISTENCE_MS = 2_000       # Sustained virtual-camera / photo-spoof signal
FROZEN_FEED_PERSISTENCE_MS = 6_000    # Identical non-zero pose frozen across >=6s

# Per-category cooldowns (prevents low-severity alerts from masking critical ones)
CATEGORY_COOLDOWNS_MS: Dict[str, int] = {
    "Phone / Device Detected": 8_000,
    "Multi-Face Detection": 8_000,
    "Liveness / Spoof Risk": 12_000,
    "Frozen Video Feed": 15_000,
    "Camera Occlusion": 10_000,
    "Face Absent From Frame": 10_000,
    "Gaze Deviation (Looking Away)": 6_000,
    "Teleprompter Reading Pattern": 12_000,
    "Head Pose Shift": 8_000,
}

SEVERITY_RANK: Dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}

# Calibrated deviation limits
CALIBRATION_TARGET_FRAMES = 15        # First ~10-15s of active valid frames establish baseline
MAX_BASELINE_GAZE_OFFSET = 0.22       # Caps baseline shift so extreme side-setups can't hide cheating
MAX_BASELINE_POSE_OFFSET = 14.0       # Caps yaw/pitch resting compensation in degrees
GAZE_X_DELTA_THRESHOLD = 0.34
GAZE_Y_DOWN_DELTA_THRESHOLD = 0.30
HEAD_YAW_DELTA_THRESHOLD = 22.0
HEAD_PITCH_DELTA_THRESHOLD = 20.0
LIVENESS_MIN_THRESHOLD = 0.45

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
    signal: str
    source: str = "Vision Landmark Gaze Model"
    confidence: float
    severity: str                   # "info", "low", "medium", "high"
    # Additive forensic metadata (defaults preserve 100% backward compatibility)
    duration_ms: int = 0
    requires_vlm_verification: bool = False


@dataclass
class _AnomalyStreak:
    start_ms: Optional[int] = None
    last_seen_ms: Optional[int] = None
    frame_count: int = 0
    peak_magnitude: float = 0.0
    last_emitted_ms: int = -1

    def update(self, active: bool, ts_ms: int, magnitude: float = 0.0) -> Tuple[int, int]:
        """Returns (continuous_duration_ms, streak_frame_count)."""
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

    # O(1) running counters (accurate across multi-hour interviews with bounded RAM)
    total_frames: int = 0
    camera_active_frames: int = 0
    face_present_frames: int = 0
    center_gaze_frames: int = 0
    multi_face_frames: int = 0
    low_liveness_frames: int = 0

    # Bounded sliding window for gaze/teleprompter/frozen-feed dynamics
    recent_frames: Deque[VisionTelemetryFrame] = field(
        default_factory=lambda: deque(maxlen=FRAME_RING_BUFFER_SIZE)
    )
    events: List[VisionAnomalyEvent] = field(default_factory=list)
    streaks: Dict[str, _AnomalyStreak] = field(default_factory=dict)

    # Personal resting baseline (neutralizes external-monitor camera offset)
    cal_gaze_x: List[float] = field(default_factory=list)
    cal_gaze_y: List[float] = field(default_factory=list)
    cal_yaw: List[float] = field(default_factory=list)
    cal_pitch: List[float] = field(default_factory=list)
    baseline_locked: bool = False
    base_gaze_x: float = 0.0
    base_gaze_y: float = 0.0
    base_yaw: float = 0.0
    base_pitch: float = 0.0

    # Frozen-feed tracker
    last_pose_signature: Optional[Tuple[float, float, float, float, float]] = None

    # VLM budget & throttling
    vlm_triggers_sent: int = 0
    last_vlm_trigger_ms: int = -1

    def streak(self, key: str) -> _AnomalyStreak:
        if key not in self.streaks:
            self.streaks[key] = _AnomalyStreak()
        return self.streaks[key]


class VisionProctoringService:
    """
    Production-hardened real-time & post-session vision proctoring engine:
      - Bounded ring-buffer memory + LRU session eviction (zero OOM risk)
      - Online resting-posture calibration (reduces multi-monitor / neurodivergent false positives)
      - True timestamp-based temporal debouncing per anomaly category (FPS-independent)
      - Multi-signal evaluation without alert masking (critical events never dropped by low alerts)
      - Anti-spoofing (liveness + frozen virtual-camera feed) and teleprompter saccade detection
      - Budget-capped Tier 2 Gemini VLM escalation flagging
    """

    def __init__(self, max_sessions: int = MAX_ACTIVE_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionVisionState] = OrderedDict()

    def clear_session(self, session_id: str) -> None:
        """Explicitly release session memory once final IntegrityReport is persisted."""
        self._sessions.pop(session_id, None)

    def ingest_frame(self, frame: VisionTelemetryFrame) -> Optional[VisionAnomalyEvent]:
        """
        Process a single telemetry frame, persist all triggered events for this frame,
        and return the highest-severity newly emitted event (if any).
        """
        emitted = self.ingest_frame_all(frame)
        if not emitted:
            return None
        return max(emitted, key=lambda e: (SEVERITY_RANK.get(e.severity, 0), e.confidence))

    def ingest_frame_all(self, frame: VisionTelemetryFrame) -> List[VisionAnomalyEvent]:
        """
        Full multi-channel evaluator: evaluates ALL anomaly detectors independently so
        co-occurring events (e.g., Phone + Downward Gaze) are never mutually masked.
        """
        clean = self._sanitize_frame(frame)
        state = self._get_or_create_state(clean.session_id, clean.timestamp_ms)

        # Reject heavily out-of-order stale frames (>5s behind stream head)
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

        # Update online resting-posture calibration during initial active frames
        self._update_calibration(state, clean)

        # Compute baseline-relative gaze & head pose
        rel_gaze_x = clean.gaze_offset_x - state.base_gaze_x
        rel_gaze_y = clean.gaze_offset_y - state.base_gaze_y
        rel_yaw = clean.head_pose_yaw - state.base_yaw
        rel_pitch = clean.head_pose_pitch - state.base_pitch

        # Update accurate O(1) summary counters
        if clean.camera_active:
            state.camera_active_frames += 1
            if clean.faces_detected >= 1:
                state.face_present_frames += 1
        if clean.faces_detected > 1:
            state.multi_face_frames += 1
        if clean.camera_active and clean.faces_detected >= 1 and clean.liveness_score < LIVENESS_MIN_THRESHOLD:
            state.low_liveness_frames += 1

        is_gaze_deviated, gaze_dir_label = self._evaluate_gaze_state(clean, rel_gaze_x, rel_gaze_y)
        if clean.camera_active and clean.faces_detected >= 1 and not is_gaze_deviated:
            state.center_gaze_frames += 1

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
        # Channel 3: Anti-Spoofing (Liveness Score Drop / Virtual Camera)
        # -------------------------------------------------------------------
        spoof_active = (
            clean.camera_active
            and clean.faces_detected >= 1
            and clean.liveness_score < LIVENESS_MIN_THRESHOLD
        )
        sp_dur, sp_frames = state.streak("liveness").update(
            spoof_active, ts_ms, 1.0 - clean.liveness_score
        )
        if spoof_active and sp_dur >= LIVENESS_PERSISTENCE_MS and sp_frames >= 2:
            cat = "Liveness / Spoof Risk"
            if state.streak("liveness").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Sustained low liveness score ({clean.liveness_score:.2f}) for "
                            f"{sp_dur / 1000:.1f}s (possible virtual camera or static photo)"
                        ),
                        source="Anti-Spoof Liveness Engine",
                        confidence=round(min(0.95, 0.70 + (1.0 - clean.liveness_score) * 0.35), 2),
                        severity="high",
                        duration_ms=sp_dur,
                        vlm_eligible=True,
                    )
                )
                state.streak("liveness").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 4: Frozen / Looped Video Feed Tampering
        # -------------------------------------------------------------------
        frozen_active = self._is_pose_frozen(state, clean)
        fr_dur, fr_frames = state.streak("frozen").update(frozen_active, ts_ms, 1.0)
        if frozen_active and fr_dur >= FROZEN_FEED_PERSISTENCE_MS and fr_frames >= 6:
            cat = "Frozen Video Feed"
            if state.streak("frozen").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=f"Zero micro-motion across {fr_frames} frames ({fr_dur / 1000:.1f}s); video feed may be frozen or spoofed",
                        source="Temporal Micro-Motion Monitor",
                        confidence=0.89,
                        severity="medium",
                        duration_ms=fr_dur,
                        vlm_eligible=False,
                    )
                )
                state.streak("frozen").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 5: Camera Occlusion OR Face Absent From Frame
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

        # Skip gaze/head analytics when no face is visible
        if not clean.camera_active or clean.faces_detected == 0:
            state.streak("gaze").update(False, ts_ms)
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
                # Keep severity proportional: never auto-label a side glance as "high" without extreme offset
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
        # Channel 7: Horizontal Teleprompter Saccade Reading Detection
        # -------------------------------------------------------------------
        scan_detected, scan_sweeps = self._detect_teleprompter_scanning(state, ts_ms)
        if scan_detected:
            cat = "Teleprompter Reading Pattern"
            if state.streak("teleprompter").can_emit(ts_ms, CATEGORY_COOLDOWNS_MS[cat]):
                emitted.append(
                    self._create_event(
                        state=state,
                        ts_ms=ts_ms,
                        ts_offset=ts_offset,
                        category=cat,
                        signal=(
                            f"Rhythmic horizontal eye-scanning pattern detected "
                            f"({scan_sweeps} sweeps across ~8s with flat vertical gaze)"
                        ),
                        source="Temporal Gaze Saccade Analyzer",
                        confidence=0.82,
                        severity="medium",
                        duration_ms=8_000,
                        vlm_eligible=False,
                    )
                )
                state.streak("teleprompter").mark_emitted(ts_ms)

        # -------------------------------------------------------------------
        # Channel 8: Baseline-Relative Sustained Head Pose Shift (>= 3.0s)
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
        """Helper for ScoringWorker / heuristics.py to correlate vision flags with a Q&A turn."""
        state = self._sessions.get(session_id)
        if not state:
            return []
        return [e for e in state.events if start_ms <= e.timestamp_ms <= end_ms]

    def get_session_summary(
        self,
        session_id: str,
        allow_demo_mock: bool = False,
    ) -> Dict[str, Any]:
        """
        Aggregate visual telemetry and calculate attention & continuity ratings.
        NEVER fabricates cheating accusations on empty sessions unless allow_demo_mock=True.
        """
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
                "calibration_complete": False,
                "baseline_pose": None,
                "phone_events_count": 0,
                "multi_face_events_count": 0,
                "gaze_deviation_events_count": 0,
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
        face_coverage = round((state.face_present_frames / total) * 100.0, 1)

        events = state.events
        phone_events = sum(1 for e in events if e.category == "Phone / Device Detected")
        multi_face_events = sum(1 for e in events if e.category == "Multi-Face Detection")
        gaze_dev_events = sum(1 for e in events if e.category == "Gaze Deviation (Looking Away)")
        teleprompter_events = sum(1 for e in events if e.category == "Teleprompter Reading Pattern")
        liveness_events = sum(1 for e in events if e.category in ("Liveness / Spoof Risk", "Frozen Video Feed"))
        face_absent_events = sum(1 for e in events if e.category == "Face Absent From Frame")
        camera_off_events = sum(1 for e in events if e.category == "Camera Occlusion")

        # Bounded, event-based visual attention penalty (prevents single walk-by from zeroing score)
        penalty = (
            min(phone_events * 22.0, 44.0)
            + min(multi_face_events * 16.0, 32.0)
            + min(liveness_events * 18.0, 36.0)
            + min(gaze_dev_events * 5.0, 25.0)
            + min(teleprompter_events * 8.0, 24.0)
            + min((camera_off_events + face_absent_events) * 6.0, 18.0)
        )
        # Modest continuity penalty if candidate was off-camera for >20% of the call
        if face_coverage < 80.0:
            penalty += (80.0 - face_coverage) * 0.25

        visual_attention = round(max(0.0, min(100.0, 100.0 - penalty)), 1)

        return {
            "data_status": "OK",
            "total_frames_analyzed": total,
            "visual_attention_score": visual_attention,
            "gaze_consistency_pct": gaze_consistency,
            "face_coverage_pct": face_coverage,
            "calibration_complete": state.baseline_locked,
            "baseline_pose": {
                "gaze_x": round(state.base_gaze_x, 3),
                "gaze_y": round(state.base_gaze_y, 3),
                "yaw": round(state.base_yaw, 1),
                "pitch": round(state.base_pitch, 1),
            },
            "phone_events_count": phone_events,
            "multi_face_events_count": multi_face_events,
            "gaze_deviation_events_count": gaze_dev_events,
            "teleprompter_scan_events_count": teleprompter_events,
            "liveness_risk_events_count": liveness_events,
            "face_absent_events_count": face_absent_events,
            "camera_off_events_count": camera_off_events,
            "vlm_triggers_sent": state.vlm_triggers_sent,
            "anomaly_events": [e.model_dump() for e in events],
        }

    # -----------------------------------------------------------------------
    # Internal Helpers
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
        """Defensively clamps corrupted or adversarial telemetry values (NaN/Inf/out-of-range)."""
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
        # Only sample plausible resting frames (not while looking 70 degrees away)
        if abs(frame.head_pose_yaw) > 30.0 or abs(frame.head_pose_pitch) > 30.0:
            return

        state.cal_gaze_x.append(frame.gaze_offset_x)
        state.cal_gaze_y.append(frame.gaze_offset_y)
        state.cal_yaw.append(frame.head_pose_yaw)
        state.cal_pitch.append(frame.head_pose_pitch)

        if len(state.cal_gaze_x) >= CALIBRATION_TARGET_FRAMES:
            state.base_gaze_x = max(
                -MAX_BASELINE_GAZE_OFFSET,
                min(MAX_BASELINE_GAZE_OFFSET, statistics.median(state.cal_gaze_x)),
            )
            state.base_gaze_y = max(
                -MAX_BASELINE_GAZE_OFFSET,
                min(MAX_BASELINE_GAZE_OFFSET, statistics.median(state.cal_gaze_y)),
            )
            state.base_yaw = max(
                -MAX_BASELINE_POSE_OFFSET,
                min(MAX_BASELINE_POSE_OFFSET, statistics.median(state.cal_yaw)),
            )
            state.base_pitch = max(
                -MAX_BASELINE_POSE_OFFSET,
                min(MAX_BASELINE_POSE_OFFSET, statistics.median(state.cal_pitch)),
            )
            state.baseline_locked = True
            # Free calibration buffers immediately
            state.cal_gaze_x.clear()
            state.cal_gaze_y.clear()
            state.cal_yaw.clear()
            state.cal_pitch.clear()

    @staticmethod
    def _evaluate_gaze_state(
        frame: VisionTelemetryFrame,
        rel_x: float,
        rel_y: float,
    ) -> Tuple[bool, str]:
        if frame.gaze_direction == "down" or rel_y > GAZE_Y_DOWN_DELTA_THRESHOLD:
            return (True, "DOWN (Lap / Secondary Device Area)")
        if abs(rel_x) > GAZE_X_DELTA_THRESHOLD or frame.gaze_direction in ("left", "right"):
            if frame.gaze_direction in ("left", "right"):
                return (True, frame.gaze_direction.upper())
            return (True, "RIGHT" if rel_x > 0 else "LEFT")
        return (False, "CENTER")

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
    def _is_pose_frozen(state: _SessionVisionState, frame: VisionTelemetryFrame) -> bool:
        if not frame.camera_active or frame.faces_detected == 0:
            state.last_pose_signature = None
            return False

        sig = (
            round(frame.gaze_offset_x, 4),
            round(frame.gaze_offset_y, 4),
            round(frame.head_pose_yaw, 3),
            round(frame.head_pose_pitch, 3),
            round(frame.head_pose_roll, 3),
        )
        # Ignore all-zero default stubs sent by minimal clients without pose extraction
        all_zero_stub = sig == (0.0, 0.0, 0.0, 0.0, 0.0) and frame.liveness_score >= 0.85
        is_identical = (state.last_pose_signature == sig) and not all_zero_stub
        state.last_pose_signature = sig
        return is_identical

    @staticmethod
    def _detect_teleprompter_scanning(state: _SessionVisionState, now_ms: int) -> Tuple[bool, int]:
        """
        Detects rhythmic horizontal eye sweeps (reading scrolled LLM text on-screen)
        over the last 8 seconds while vertical gaze stays flat.
        """
        window = [
            f
            for f in state.recent_frames
            if (now_ms - f.timestamp_ms) <= 8_000 and f.camera_active and f.faces_detected == 1
        ]
        if len(window) < 8:
            return (False, 0)

        xs = [f.gaze_offset_x - state.base_gaze_x for f in window]
        ys = [f.gaze_offset_y - state.base_gaze_y for f in window]

        # Reading lines keeps vertical eye band tight while sweeping horizontally
        if statistics.pstdev(ys) > 0.09:
            return (False, 0)

        # Count significant horizontal direction reversals (line sweeps + carriage returns)
        sweeps = 0
        last_dir = 0
        anchor_x = xs[0]
        for x in xs[1:]:
            delta = x - anchor_x
            if abs(delta) >= 0.14:
                curr_dir = 1 if delta > 0 else -1
                if last_dir != 0 and curr_dir != last_dir:
                    sweeps += 1
                last_dir = curr_dir
                anchor_x = x

        return (sweeps >= 3, sweeps)

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
            "calibration_complete": True,
            "baseline_pose": {"gaze_x": 0.02, "gaze_y": -0.01, "yaw": 1.2, "pitch": -0.8},
            "phone_events_count": 1,
            "multi_face_events_count": 1,
            "gaze_deviation_events_count": 1,
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