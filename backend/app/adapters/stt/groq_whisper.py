import logging
from typing import List, Optional
import httpx
from app.config import settings
from app.schemas.contracts import UtteranceEvent, WordTiming, SpeakerRole
from app.adapters.stt.base import STTAdapter

logger = logging.getLogger("oculus.stt.groq")


class GroqWhisperAdapter(STTAdapter):
    """
    Groq Whisper Large v3 Turbo adapter for fast, low-cost per-speaker audio transcription.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.is_available = bool(self.api_key)

    def _mock_transcription(self, session_id: str, offset_ms: int) -> List[UtteranceEvent]:
        return [
            UtteranceEvent(
                session_id=session_id,
                utterance_id=f"groq_{session_id}_{offset_ms}",
                speaker=SpeakerRole.CANDIDATE,
                speaker_label="S2",
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

    async def transcribe(
        self,
        audio_bytes: bytes,
        session_id: str,
        offset_ms: int = 0,
        custom_vocab: Optional[List[str]] = None,
        diarize: bool = False,
    ) -> List[UtteranceEvent]:
        if not self.is_available:
            logger.info("Groq API key not set. Returning mock utterance.")
            return self._mock_transcription(session_id, offset_ms)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        prompt = ", ".join(custom_vocab or [])
        data = {
            "model": "whisper-large-v3-turbo",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
            "prompt": prompt,
        }
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )
                res.raise_for_status()
                res_json = res.json()

                words = []
                for w in res_json.get("words", []):
                    words.append(
                        WordTiming(
                            word=w.get("word", ""),
                            start_ms=offset_ms + int(w.get("start", 0) * 1000),
                            end_ms=offset_ms + int(w.get("end", 0) * 1000),
                        )
                    )

                end_ms = offset_ms + int(res_json.get("duration", 0) * 1000)
                return [
                    UtteranceEvent(
                        session_id=session_id,
                        utterance_id=f"groq_{session_id}_{offset_ms}",
                        speaker=SpeakerRole.CANDIDATE,
                        speaker_label="S2",
                        text=res_json.get("text", "").strip(),
                        start_ms=offset_ms,
                        end_ms=max(end_ms, offset_ms + 1000),
                        words=words,
                    )
                ]
        except Exception as e:
            logger.warning(f"Groq Whisper transcription failed ({e}), falling back to mock: {e}")
            return self._mock_transcription(session_id, offset_ms)
