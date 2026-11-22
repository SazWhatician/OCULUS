import time
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("oculus")


class MetricsRegistry:
    def __init__(self):
        self._counters: Dict[str, int] = {
            "sessions_created": 0,
            "sessions_completed": 0,
            "audio_chunks_processed": 0,
            "video_frames_processed": 0,
            "utterances_stitched": 0,
            "vision_anomalies_detected": 0,
            "llm_judge_invocations": 0,
            "prefilter_skips": 0,
        }
        self._gauges: Dict[str, float] = {
            "active_sessions": 0,
            "stream_lag_audio": 0,
            "stream_lag_video": 0,
        }
        self._provider_health: Dict[str, str] = {
            "speechmatics": "UNKNOWN",
            "groq_whisper": "UNKNOWN",
            "groq_llm": "UNKNOWN",
            "gemini_flash": "UNKNOWN",
            "attendee": "UNKNOWN",
        }

    def inc(self, counter: str, delta: int = 1) -> None:
        self._counters[counter] = self._counters.get(counter, 0) + delta

    def set_gauge(self, gauge: str, value: float) -> None:
        self._gauges[gauge] = value

    def set_provider_health(self, provider: str, status: str) -> None:
        self._provider_health[provider] = status

    def snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "provider_health": dict(self._provider_health),
        }


metrics = MetricsRegistry()


def log_structured(
    stage: str,
    message: str,
    session_id: Optional[str] = None,
    worker_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    payload = {
        "ts": time.time(),
        "stage": stage,
        "msg": message,
        "session_id": session_id or "",
        "worker": worker_name or "",
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    if extra:
        payload.update(extra)

    logger.log(level, json.dumps(payload))
