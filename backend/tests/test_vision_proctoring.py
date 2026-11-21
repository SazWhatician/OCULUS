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

if __name__ == "__main__":
    test_vision_telemetry()
