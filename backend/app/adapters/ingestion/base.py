from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.schemas.contracts import IngestionMode, SessionStatus


class IngestionAdapter(ABC):
    @property
    @abstractmethod
    def mode(self) -> IngestionMode:
        pass

    @abstractmethod
    async def start_session(self, session_id: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def stop_session(self, session_id: str) -> bool:
        pass
