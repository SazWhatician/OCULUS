import asyncio
from app.services.vision_proctoring import get_vision_service, VisionTelemetryFrame

def test_vision_telemetry():
    svc = get_vision_service()
    session_id = "test_vision_session"

    # Ingest 5 frames: 3 center, 2 off-center right
    for i in range(3):
        svc.ingest_frame(VisionTelemetryFrame(
            session_id=session_id,
            timestamp_ms=1000 + i * 1000,
            gaze_direction="center",
            gaze_offset_x=0.02
        ))

    for i in range(3):
        svc.ingest_frame(VisionTelemetryFrame(
            session_id=session_id,
            timestamp_ms=4000 + i * 1000,
            gaze_direction="right",
            gaze_offset_x=0.55
        ))

    summary = svc.get_session_summary(session_id)
    print("=== Test Vision Proctoring Telemetry ===")
    print("Total frames:", summary["total_frames_analyzed"])
    print("Visual attention score:", summary["visual_attention_score"])
    print("Gaze consistency:", summary["gaze_consistency_pct"])
    print("Anomaly events count:", len(summary["anomaly_events"]))
    assert summary["total_frames_analyzed"] == 6
    assert summary["gaze_consistency_pct"] > 0.0
    print("[PASS] Vision Proctoring Test Passed!")


def test_off_center_camera_calibration_no_false_positives():
    svc = get_vision_service()
    session_id = "test_offcenter_cal_session"
    svc.clear_session(session_id)

    # 15 resting frames with off-center webcam (-0.18, raw label "left") lock baseline
    for i in range(15):
        svc.ingest_frame_all(
            VisionTelemetryFrame(
                session_id=session_id,
                timestamp_ms=1_000 + i * 500,
                gaze_direction="left",
                gaze_offset_x=-0.18,
            )
        )

    # Subsequent frames near the calibrated resting offset emit 0 gaze deviations
    for j in range(10):
        svc.ingest_frame_all(
            VisionTelemetryFrame(
                session_id=session_id,
                timestamp_ms=9_000 + j * 500,
                gaze_direction="left",
                gaze_offset_x=-0.19,
            )
        )

    summary = svc.get_session_summary(session_id)
    assert summary["calibration_complete"] is True
    assert summary["gaze_deviation_events_count"] == 0

if __name__ == "__main__":
    test_vision_telemetry()
