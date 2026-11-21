from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

class PlatformType(str, Enum):
    ZOOM = "zoom"
    TEAMS = "teams"
    GOOGLE_MEET = "google_meet"

class MeetingAdapter(ABC):
    """
    Abstract interface for meeting platform adapters (Zoom, MS Teams, Google Meet).
    """

    @property
    @abstractmethod
    def platform_name(self) -> PlatformType:
        """Return the platform identifier."""
        pass

    @abstractmethod
    async def join_meeting(self, meeting_url: str, bot_name: str, session_id: str) -> Dict[str, Any]:
        """
        Dispatch the bot to join a meeting URL.
        Returns bot dispatch status and metadata.
        """
        pass

    @abstractmethod
    async def leave_meeting(self, bot_id: str) -> bool:
        """
        Command the bot to leave an active meeting.
        """
        pass

    @abstractmethod
    async def get_session_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Fetch real-time status of the bot session (e.g. JOINING, RECORDING, DEGRADED, LEAVING).
        """
        pass
