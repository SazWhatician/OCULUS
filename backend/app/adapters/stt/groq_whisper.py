import asyncio
import hashlib
import logging
import random
from typing import Any, Dict, List, Optional

import httpx

from app.adapters.stt.base import STTAdapter, STTProviderError, detect_audio_format
from app.config import settings
from app.schemas.contracts import SpeakerRole, UtteranceEvent, WordTiming

logger = logging.getLogger("oculus.stt.groq")

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
MAX_PROMPT_CHARS = 750  # Stays safely under Whisper's 224-token prompt cap


class GroqWhisperAdapter(STTAdapter):
    """
    Groq Whisper Large v3 Turbo adapter:
      - Connection-pooled async HTTP client with 429/5xx exponential backoff
      - Auto-detects WebM/Ogg/WAV/MP3/M4A containers (prevents browser blob HTTP 400s)
      - Filters Whisper silence hallucinations via segment no_speech_prob
      - Raises STTProviderError on failure so STTRouter trips CircuitBreaker accurately
    """

    @property
    def provider_name(self) -> str:
        return "groq_whisper"

    def __init__(self) -> None:
        self.api_key: Optional[str] = getattr(settings, "GROQ_API_KEY", None)
        self.is_available: bool = bool(self.api_key)
        self.model: str = str(getattr(settings, "GROQ_WHISPER_MODEL", DEFAULT_WHISPER_MODEL))
        self.max_retries: int = int(getattr(settings, "GROQ_MAX_RETRIES", 2))
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
                limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
                headers={"Authorization": f"Bearer {self.api_key or ''}"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def transcribe(
        self,
        audio_bytes: bytes,
        session_id: str,
        offset_ms: int = 0,
        custom_vocab: Optional[List[str]] = None,
        diarize: bool = False,
        speaker_hint: SpeakerRole = SpeakerRole.UNKNOWN,
    ) -> List[UtteranceEvent]:
        if not self.is_available:
            raise STTProviderError("GROQ_API_KEY is not configured.")
        if not audio_bytes:
            return []

        ext, mime = detect_audio_format(audio_bytes)
        prompt = self._build_vocab_prompt(custom_vocab)

        form_fields: List[tuple[str, str]] = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("temperature", "0"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        if prompt:
            form_fields.append(("prompt", prompt))

        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                files = {"file": (f"audio.{ext}", audio_bytes, mime)}
                res = await client.post(GROQ_TRANSCRIBE_URL, data=form_fields, files=files)

                if res.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    backoff = self._backoff_seconds(res.headers.get("Retry-After"), attempt)
                    logger.warning("Groq Whisper HTTP %s; retrying in %.2fs", res.status_code, backoff)
                    await asyncio.sleep(backoff)
                    continue

                if res.status_code >= 400:
                    raise STTProviderError(
                        f"Groq Whisper HTTP {res.status_code}: {res.text[:250]}"
                    )

                res_json = res.json()
                return self._parse_whisper_response(
                    res_json=res_json,
                    session_id=session_id,
                    offset_ms=offset_ms,
                    speaker_hint=speaker_hint,
                )

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_seconds(None, attempt))
                    continue
                raise STTProviderError(f"Groq Whisper network error: {exc}") from exc

        raise STTProviderError("Groq Whisper retries exhausted.")

    @staticmethod
    def _parse_whisper_response(
        res_json: Dict[str, Any],
        session_id: str,
        offset_ms: int,
        speaker_hint: SpeakerRole,
    ) -> List[UtteranceEvent]:
        text = str(res_json.get("text") or "").strip()
        if not text:
            return []

        # Drop pure-silence Whisper hallucinations (e.g. phantom "Thank you." on background static)
        segments = res_json.get("segments") or []
        if segments and all(
            float(seg.get("no_speech_prob", 0.0) or 0.0) >= 0.85
            and float(seg.get("avg_logprob", 0.0) or 0.0) <= -0.9
            for seg in segments
        ):
            return []

        words: List[WordTiming] = []
        for w in res_json.get("words") or []:
            tok = str(w.get("word") or "").strip()
            if not tok:
                continue
            s_ms = offset_ms + int(round(float(w.get("start", 0.0) or 0.0) * 1000))
            e_ms = offset_ms + int(round(float(w.get("end", 0.0) or 0.0) * 1000))
            words.append(WordTiming(word=tok, start_ms=s_ms, end_ms=max(e_ms, s_ms)))

        duration_ms = int(round(float(res_json.get("duration", 0.0) or 0.0) * 1000))
        start_ms = words[0].start_ms if words else offset_ms
        end_ms = max(
            words[-1].end_ms if words else 0,
            offset_ms + duration_ms,
            start_ms + 200,
        )

        role = speaker_hint if speaker_hint != SpeakerRole.UNKNOWN else SpeakerRole.CANDIDATE
        label = "S1" if role == SpeakerRole.INTERVIEWER else "S2"
        utt_hash = hashlib.sha256(f"{session_id}|{start_ms}|{text}".encode("utf-8")).hexdigest()[:10]

        return [
            UtteranceEvent(
                session_id=session_id,
                utterance_id=f"groq_{utt_hash}",
                speaker=role,
                speaker_label=label,
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                words=words,
            )
        ]

    @staticmethod
    def _build_vocab_prompt(custom_vocab: Optional[List[str]]) -> str:
        if not custom_vocab:
            return ""
        terms = list(dict.fromkeys(str(t).strip() for t in custom_vocab if str(t).strip()))
        return ", ".join(terms)[:MAX_PROMPT_CHARS]

    @staticmethod
    def _backoff_seconds(retry_after: Optional[str], attempt: int) -> float:
        if retry_after:
            try:
                return max(0.5, min(10.0, float(retry_after)))
            except ValueError:
                pass
        return 0.5 * (2 ** attempt) + random.uniform(0.05, 0.20)

    def _mock_transcription(
        self,
        session_id: str,
        offset_ms: int,
        speaker_hint: SpeakerRole = SpeakerRole.CANDIDATE,
    ) -> List[UtteranceEvent]:
        """Used ONLY by STTRouter when zero STT API keys are configured in local demo mode."""
        role = speaker_hint if speaker_hint != SpeakerRole.UNKNOWN else SpeakerRole.CANDIDATE
        label = "S1" if role == SpeakerRole.INTERVIEWER else "S2"
        return [
            UtteranceEvent(
                session_id=session_id,
                utterance_id=f"groq_{session_id}_{offset_ms}",
                speaker=role,
                speaker_label=label,
                text="In my previous role, I deployed microservices on Kubernetes.",
                start_ms=offset_ms,
                end_ms=offset_ms + 3500,
                words=[
                    WordTiming(word="In", start_ms=offset_ms, end_ms=offset_ms + 200),
                    WordTiming(word="my", start_ms=offset_ms + 250, end_ms=offset_ms + 400),
                    WordTiming(word="previous", start_ms=offset_ms + 450, end_ms=offset_ms + 800),
                    WordTiming(word="role,", start_ms=offset_ms + 850, end_ms=offset_ms + 1200),
                    WordTiming(word="I", start_ms=offset_ms + 1250, end_ms=offset_ms + 1400),
                    WordTiming(word="deployed", start_ms=offset_ms + 1450, end_ms=offset_ms + 1900),
                    WordTiming(word="microservices", start_ms=offset_ms + 1950, end_ms=offset_ms + 2700),
                    WordTiming(word="on", start_ms=offset_ms + 2750, end_ms=offset_ms + 2900),
                    WordTiming(word="Kubernetes.", start_ms=offset_ms + 2950, end_ms=offset_ms + 3500),
                ],
            )
        ]