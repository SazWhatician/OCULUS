from abc import ABC, abstractmethod
from typing import List, Optional
from app.config import settings
from app.schemas.contracts import UtteranceEvent, AudioChunkEvent
from app.services.resilience import CircuitBreaker


class STTAdapter(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        session_id: str,
        offset_ms: int = 0,
        custom_vocab: Optional[List[str]] = None,
        diarize: bool = True,
    ) -> List[UtteranceEvent]:
        pass


class STTRouter:
    """
    Hybrid Cost Router:
    - Routes mixed audio (requires diarization) -> SpeechmaticsAdapter
    - Routes isolated per-participant tracks -> GroqWhisperAdapter ($0 daily quota)
    - Automatically falls back to Groq Whisper if Speechmatics key is missing or circuit is open.
    """

    def __init__(self):
        from app.adapters.stt.speechmatics import SpeechmaticsAdapter
        from app.adapters.stt.groq_whisper import GroqWhisperAdapter

        self.speechmatics = SpeechmaticsAdapter()
        self.groq_whisper = GroqWhisperAdapter()
        self.cb_speechmatics = CircuitBreaker("speechmatics", failure_threshold=3, cooldown_seconds=60)
        self.cb_groq = CircuitBreaker("groq_whisper", failure_threshold=5, cooldown_seconds=30)

    async def route_and_transcribe(
        self,
        event: AudioChunkEvent,
        audio_bytes: bytes,
        custom_vocab: Optional[List[str]] = None,
    ) -> List[UtteranceEvent]:
        # Condition 1: Mixed audio requires speaker diarization -> Speechmatics
        if event.is_mixed_audio and self.cb_speechmatics.can_execute():
            try:
                results = await self.speechmatics.transcribe(
                    audio_bytes=audio_bytes,
                    session_id=event.session_id,
                    offset_ms=event.timestamp_ms,
                    custom_vocab=custom_vocab,
                    diarize=True,
                )
                self.cb_speechmatics.record_success()
                return results
            except Exception as e:
                self.cb_speechmatics.record_failure(e)

        # Condition 2: Isolated track or fallback -> Groq Whisper
        if self.cb_groq.can_execute():
            try:
                results = await self.groq_whisper.transcribe(
                    audio_bytes=audio_bytes,
                    session_id=event.session_id,
                    offset_ms=event.timestamp_ms,
                    custom_vocab=custom_vocab,
                    diarize=False,
                )
                self.cb_groq.record_success()
                return results
            except Exception as e:
                self.cb_groq.record_failure(e)

        # Graceful fallback to speechmatics mock
        return await self.speechmatics.transcribe(
            audio_bytes=audio_bytes,
            session_id=event.session_id,
            offset_ms=event.timestamp_ms,
            custom_vocab=custom_vocab,
            diarize=True,
        )
