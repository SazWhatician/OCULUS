import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.config import settings
from app.schemas.contracts import AudioChunkEvent, SpeakerRole, UtteranceEvent
from app.services.resilience import CircuitBreaker

logger = logging.getLogger("oculus.stt.router")


class STTProviderError(RuntimeError):
    """Raised by STT adapters on missing keys, HTTP/API errors, or job rejection so breakers trip."""


def detect_audio_format(audio_bytes: bytes) -> Tuple[str, str]:
    """Detects audio container extension and MIME type from magic bytes."""
    if not audio_bytes:
        return ("wav", "audio/wav")
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return ("webm", "audio/webm")
    if audio_bytes.startswith(b"OggS"):
        return ("ogg", "audio/ogg")
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return ("wav", "audio/wav")
    if audio_bytes.startswith(b"fLaC"):
        return ("flac", "audio/flac")
    if audio_bytes.startswith(b"ID3") or (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return ("mp3", "audio/mpeg")
    if len(audio_bytes) >= 8 and audio_bytes[4:8] == b"ftyp":
        return ("m4a", "audio/mp4")
    return ("wav", "audio/wav")


class STTAdapter(ABC):
    is_available: bool = False

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        session_id: str,
        offset_ms: int = 0,
        custom_vocab: Optional[List[str]] = None,
        diarize: bool = True,
        speaker_hint: SpeakerRole = SpeakerRole.UNKNOWN,
    ) -> List[UtteranceEvent]:
        pass

    async def aclose(self) -> None:
        return None


class STTRouter:
    """
    Hybrid Cost & Resilience STT Router:
      - Respects STT_PROVIDER (`hybrid`, `groq`, `speechmatics`)
      - Skips unavailable providers instead of letting mock responses bypass failover
      - Enforces per-provider timeouts and circuit breakers
      - Production outage safety: raises STTProviderError so Redis Streams PEL retries the chunk
        rather than injecting canned mock transcripts into real interviews
    """

    def __init__(
        self,
        speechmatics: Optional[STTAdapter] = None,
        groq_whisper: Optional[STTAdapter] = None,
    ) -> None:
        if speechmatics is None:
            from app.adapters.stt.speechmatics import SpeechmaticsAdapter
            speechmatics = SpeechmaticsAdapter()
        if groq_whisper is None:
            from app.adapters.stt.groq_whisper import GroqWhisperAdapter
            groq_whisper = GroqWhisperAdapter()

        self.speechmatics = speechmatics
        self.groq_whisper = groq_whisper
        self.cb_speechmatics = CircuitBreaker("speechmatics", failure_threshold=3, cooldown_seconds=60)
        self.cb_groq = CircuitBreaker("groq_whisper", failure_threshold=5, cooldown_seconds=30)

        self.stt_provider_mode: str = str(
            getattr(settings, "STT_PROVIDER", "hybrid")
        ).strip().lower()
        self.provider_timeout_seconds: float = float(
            getattr(settings, "STT_PROVIDER_TIMEOUT_SECONDS", 60.0)
        )
        self.allow_mock_fallback: bool = bool(
            getattr(settings, "STT_ALLOW_MOCK_FALLBACK", True)
        )

    async def aclose(self) -> None:
        await asyncio.gather(
            self.speechmatics.aclose(),
            self.groq_whisper.aclose(),
            return_exceptions=True,
        )

    def _ordered_plan(self, event: AudioChunkEvent) -> List[Tuple[STTAdapter, CircuitBreaker, bool]]:
        """Returns [(adapter, circuit_breaker, diarize_flag)] in priority order."""
        sm = (self.speechmatics, self.cb_speechmatics, bool(event.is_mixed_audio))
        gq = (self.groq_whisper, self.cb_groq, False)

        if self.stt_provider_mode == "groq":
            return [gq, sm]
        if self.stt_provider_mode == "speechmatics":
            return [sm, gq]
        # hybrid default: mixed audio -> Speechmatics first; isolated track -> Groq Whisper first
        return [sm, gq] if event.is_mixed_audio else [gq, sm]

    async def route_and_transcribe(
        self,
        event: AudioChunkEvent,
        audio_bytes: bytes,
        custom_vocab: Optional[List[str]] = None,
    ) -> List[UtteranceEvent]:
        if not audio_bytes:
            return []

        plan = self._ordered_plan(event)
        any_live_configured = any(adapter.is_available for adapter, _, _ in plan)
        speaker_hint = getattr(event, "speaker_hint", SpeakerRole.UNKNOWN) or SpeakerRole.UNKNOWN

        last_error: Optional[Exception] = None
        for adapter, breaker, diarize in plan:
            if not adapter.is_available:
                continue
            if not breaker.can_execute():
                logger.warning("STT circuit open for '%s'; skipping.", adapter.provider_name)
                continue

            try:
                results = await asyncio.wait_for(
                    adapter.transcribe(
                        audio_bytes=audio_bytes,
                        session_id=event.session_id,
                        offset_ms=event.timestamp_ms,
                        custom_vocab=custom_vocab,
                        diarize=diarize,
                        speaker_hint=speaker_hint,
                    ),
                    timeout=self.provider_timeout_seconds,
                )
                breaker.record_success()
                return results
            except Exception as exc:
                last_error = exc
                breaker.record_failure(exc)
                logger.warning(
                    "STT provider '%s' failed (%s: %s); failing over.",
                    adapter.provider_name,
                    type(exc).__name__,
                    exc,
                )

        # Keyless local demo fallback ONLY when zero live STT API keys are configured
        if not any_live_configured and self.allow_mock_fallback:
            logger.info("No STT API keys configured and mock mode enabled; returning simulated transcript.")
            if event.is_mixed_audio and hasattr(self.speechmatics, "_mock_diarized_transcription"):
                return self.speechmatics._mock_diarized_transcription(event.session_id, event.timestamp_ms)
            if hasattr(self.groq_whisper, "_mock_transcription"):
                return self.groq_whisper._mock_transcription(
                    event.session_id, event.timestamp_ms, speaker_hint=speaker_hint
                )
            return []

        raise STTProviderError(
            f"All configured STT providers failed or circuits are open for session {event.session_id}: {last_error}"
        )