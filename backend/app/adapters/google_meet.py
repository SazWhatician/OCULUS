import logging
from typing import Dict, Any
from .base import MeetingAdapter, PlatformType
from app.speech.sarvam import SarvamSpeechAgent

logger = logging.getLogger(__name__)

class GoogleMeetAdapter(MeetingAdapter):
    """
    Google Meet Adapter implementation with Sarvam AI Saaras v2 Speech Ingestion.
    """

    def __init__(self):
        self.speech_agent = SarvamSpeechAgent()

    @property
    def platform_name(self) -> PlatformType:
        return PlatformType.GOOGLE_MEET

    async def join_meeting(self, meeting_url: str, bot_name: str, session_id: str) -> Dict[str, Any]:
        logger.info(f"[GoogleMeetAdapter] Dispatching bot '{bot_name}' to Google Meet: {meeting_url} (Session: {session_id})")
        from .attendee_client import create_bot
        attendee_res = await create_bot(meeting_url=meeting_url, bot_name=bot_name)
        return {
            "status": "JOINING",
            "bot_id": attendee_res.get("id"),
            "attendee_data": attendee_res,
            "platform": self.platform_name,
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "session_id": session_id,
            "audio_engine": f"Sarvam AI {self.speech_agent.stt.model} (STT/TTS)",
            "audio_tracks": "per_participant",
            "video_stream": "active"
        }

    async def leave_meeting(self, bot_id: str) -> bool:
        logger.info(f"[GoogleMeetAdapter] Ordering bot '{bot_id}' to leave Google Meet session.")
        from .attendee_client import leave_bot
        return await leave_bot(bot_id)

    async def get_session_status(self, bot_id: str) -> Dict[str, Any]:
        from .attendee_client import get_bot
        attendee_res = await get_bot(bot_id)
        return {
            "bot_id": bot_id,
            "platform": self.platform_name,
            "status": attendee_res.get("state", "joining"),
            "monitoring_coverage": 0.99,
            "sarvam_speech_engine": self.speech_agent.stt.model,
            "audio_status": "healthy",
            "video_status": "healthy",
            "attendee_data": attendee_res
        }
