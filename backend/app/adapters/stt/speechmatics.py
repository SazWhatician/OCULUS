import asyncio
import hashlib
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import httpx

from app.adapters.stt.base import STTAdapter, STTProviderError, detect_audio_format
from app.config import settings
from app.schemas.contracts import SpeakerRole, UtteranceEvent, WordTiming

logger = logging.getLogger("oculus.stt.speechmatics")

DEFAULT_SPEECHMATICS_BASE_URL = "https://asr.api.speechmatics.com/v2"
MAX_CUSTOM_VOCAB_TERMS = 200
SPEAKER_GAP_SPLIT_MS = 3_000


class SpeechmaticsAdapter(STTAdapter):
    """
    Speechmatics Batch v2 STT adapter with native speaker diarization (S1, S2),
    custom technical vocabulary injection, real job polling, and json-v2 parsing.
    """

    @property
    def provider_name(self) -> str:
        return "speechmatics"

    def __init__(self) -> None:
        self.api_key: Optional[str] = getattr(settings, "SPEECHMATICS_API_KEY", None)
        self.is_available: bool = bool(self.api_key)
        self.base_url: str = str(
            getattr(settings, "SPEECHMATICS_BASE_URL", DEFAULT_SPEECHMATICS_BASE_URL)
        ).rstrip("/")
        self.operating_point: str = str(
            getattr(settings, "SPEECHMATICS_OPERATING_POINT", "enhanced")
        ).strip()
        self.language: str = str(
            getattr(settings, "SPEECHMATICS_LANGUAGE", "en")
        ).strip()
        self.poll_timeout_sec: float = float(
            getattr(settings, "SPEECHMATICS_POLL_TIMEOUT_SEC", 45.0)
        )
        self.max_retries: int = int(getattr(settings, "SPEECHMATICS_MAX_RETRIES", 2))
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
        diarize: bool = True,
        speaker_hint: SpeakerRole = SpeakerRole.UNKNOWN,
    ) -> List[UtteranceEvent]:
        if not self.is_available:
            raise STTProviderError("SPEECHMATICS_API_KEY is not configured.")
        if not audio_bytes:
            return []

        job_id = await self._submit_job(
            audio_bytes=audio_bytes,
            custom_vocab=custom_vocab,
            diarize=diarize,
        )
        await self._wait_for_job(job_id)
        transcript_json = await self._fetch_transcript(job_id)

        return self._parse_json_v2_transcript(
            payload=transcript_json,
            session_id=session_id,
            offset_ms=offset_ms,
            diarize=diarize,
            speaker_hint=speaker_hint,
        )

    async def _submit_job(
        self,
        audio_bytes: bytes,
        custom_vocab: Optional[List[str]],
        diarize: bool,
    ) -> str:
        ext, mime = detect_audio_format(audio_bytes)
        cleaned_vocab = list(
            dict.fromkeys(str(v).strip() for v in (custom_vocab or []) if str(v).strip())
        )[:MAX_CUSTOM_VOCAB_TERMS]

        transcription_config: Dict[str, Any] = {
            "language": self.language,
            "diarization": "speaker" if diarize else "none",
            "operating_point": self.operating_point,
        }
        if cleaned_vocab:
            transcription_config["additional_vocab"] = [{"content": term} for term in cleaned_vocab]

        config_payload = {
            "type": "transcription",
            "transcription_config": transcription_config,
        }
        config_json = json.dumps(config_payload)

        client = self._get_client()
        url = f"{self.base_url}/jobs"

        for attempt in range(self.max_retries + 1):
            try:
                files = {
                    "data_file": (f"audio.{ext}", audio_bytes, mime),
                    "config": (None, config_json, "application/json"),
                }
                res = await client.post(url, files=files)

                if res.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    backoff = self._backoff_seconds(res.headers.get("Retry-After"), attempt)
                    logger.warning("Speechmatics submit HTTP %s; retrying in %.2fs", res.status_code, backoff)
                    await asyncio.sleep(backoff)
                    continue

                if res.status_code >= 400:
                    raise STTProviderError(
                        f"Speechmatics job submit HTTP {res.status_code}: {res.text[:250]}"
                    )

                body = res.json()
                job_id = body.get("id")
                if not job_id:
                    raise STTProviderError(f"Speechmatics submit returned no job id: {body}")
                return str(job_id)

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_seconds(None, attempt))
                    continue
                raise STTProviderError(f"Speechmatics submit network error: {exc}") from exc

        raise STTProviderError("Speechmatics submit retries exhausted.")

    async def _wait_for_job(self, job_id: str) -> None:
        client = self._get_client()
        url = f"{self.base_url}/jobs/{job_id}"
        deadline = time.monotonic() + self.poll_timeout_sec
        delay = 0.4

        while time.monotonic() < deadline:
            try:
                res = await client.get(url)
                if res.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(2.0, delay))
                    delay = min(2.0, delay * 1.5)
                    continue
                if res.status_code >= 400:
                    raise STTProviderError(
                        f"Speechmatics poll HTTP {res.status_code} for job {job_id}: {res.text[:200]}"
                    )

                job_info = res.json().get("job") or {}
                status = str(job_info.get("status") or "").lower()
                if status == "done":
                    return
                if status in ("rejected", "deleted", "expired", "failed"):
                    errors = job_info.get("errors") or job_info
                    raise STTProviderError(f"Speechmatics job {job_id} ended with status='{status}': {errors}")

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.debug("Transient poll error for Speechmatics job %s: %s", job_id, exc)

            await asyncio.sleep(delay)
            delay = min(1.5, delay * 1.35)

        raise STTProviderError(
            f"Speechmatics job {job_id} timed out after {self.poll_timeout_sec:.1f}s."
        )

    async def _fetch_transcript(self, job_id: str) -> Dict[str, Any]:
        client = self._get_client()
        url = f"{self.base_url}/jobs/{job_id}/transcript"

        for attempt in range(self.max_retries + 1):
            try:
                res = await client.get(url, params={"format": "json-v2"})
                if res.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_seconds(res.headers.get("Retry-After"), attempt))
                    continue
                if res.status_code >= 400:
                    raise STTProviderError(
                        f"Speechmatics transcript fetch HTTP {res.status_code} for job {job_id}: {res.text[:200]}"
                    )
                return res.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_seconds(None, attempt))
                    continue
                raise STTProviderError(f"Speechmatics transcript fetch error: {exc}") from exc

        raise STTProviderError(f"Speechmatics transcript fetch retries exhausted for job {job_id}.")

    @classmethod
    def _parse_json_v2_transcript(
        cls,
        payload: Dict[str, Any],
        session_id: str,
        offset_ms: int,
        diarize: bool,
        speaker_hint: SpeakerRole,
    ) -> List[UtteranceEvent]:
        results = payload.get("results") or []
        if not results:
            return []

        raw_groups: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for item in results:
            item_type = item.get("type")
            alts = item.get("alternatives") or []
            if not alts:
                continue

            top = alts[0]
            content = str(top.get("content") or "").strip()
            if not content:
                continue

            start_ms = offset_ms + int(round(float(item.get("start_time", 0.0) or 0.0) * 1000))
            end_ms = offset_ms + int(round(float(item.get("end_time", 0.0) or 0.0) * 1000))
            end_ms = max(end_ms, start_ms)

            if item_type == "punctuation":
                if current is not None:
                    if item.get("attaches_to") == "previous" and current["tokens"]:
                        current["tokens"][-1] = f"{current['tokens'][-1]}{content}"
                        if current["words"]:
                            prev_w = current["words"][-1]
                            current["words"][-1] = prev_w.model_copy(
                                update={"word": f"{prev_w.word}{content}", "end_ms": max(prev_w.end_ms, end_ms)}
                            )
                    else:
                        current["tokens"].append(content)
                    current["end_ms"] = max(current["end_ms"], end_ms)
                continue

            if item_type != "word":
                continue

            speaker_label = str(top.get("speaker") or ("S2" if speaker_hint == SpeakerRole.CANDIDATE else "S1")).strip()
            confidence = max(0.0, min(1.0, float(top.get("confidence", 1.0) or 1.0)))

            should_split = (
                current is not None
                and (
                    (diarize and speaker_label != current["speaker_label"])
                    or (start_ms - current["end_ms"]) > SPEAKER_GAP_SPLIT_MS
                )
            )
            if should_split:
                raw_groups.append(current)
                current = None

            if current is None:
                current = {
                    "speaker_label": speaker_label,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "tokens": [],
                    "words": [],
                }

            current["tokens"].append(content)
            current["words"].append(
                WordTiming(
                    word=content,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=round(confidence, 3),
                )
            )
            current["end_ms"] = max(current["end_ms"], end_ms)

        if current is not None:
            raw_groups.append(current)

        if not raw_groups:
            return []

        role_map = cls._infer_speaker_roles(raw_groups, diarize=diarize, speaker_hint=speaker_hint)

        utterances: List[UtteranceEvent] = []
        for idx, grp in enumerate(raw_groups):
            text = " ".join(grp["tokens"]).strip()
            if not text:
                continue
            label = grp["speaker_label"]
            utt_hash = hashlib.sha256(
                f"{session_id}|{grp['start_ms']}|{idx}|{text}".encode("utf-8")
            ).hexdigest()[:10]
            utterances.append(
                UtteranceEvent(
                    session_id=session_id,
                    utterance_id=f"sm_{utt_hash}",
                    speaker=role_map.get(label, SpeakerRole.CANDIDATE),
                    speaker_label=label,
                    text=text,
                    start_ms=grp["start_ms"],
                    end_ms=max(grp["end_ms"], grp["start_ms"] + 100),
                    words=grp["words"],
                )
            )
        return utterances

    @staticmethod
    def _infer_speaker_roles(
        groups: List[Dict[str, Any]],
        diarize: bool,
        speaker_hint: SpeakerRole,
    ) -> Dict[str, SpeakerRole]:
        labels = list(dict.fromkeys(g["speaker_label"] for g in groups))
        if not diarize:
            role = speaker_hint if speaker_hint != SpeakerRole.UNKNOWN else SpeakerRole.CANDIDATE
            return {lbl: role for lbl in labels}

        if len(labels) == 1:
            lbl = labels[0]
            if speaker_hint != SpeakerRole.UNKNOWN:
                return {lbl: speaker_hint}
            combined = " ".join(" ".join(g["tokens"]) for g in groups).strip()
            # Single-speaker chunk ending with a concise question is likely interviewer; otherwise candidate
            is_short_question = combined.endswith("?") and len(combined.split()) <= 35
            return {lbl: SpeakerRole.INTERVIEWER if is_short_question else SpeakerRole.CANDIDATE}

        # Multi-speaker chunk: S1 (or speaker with shortest/question turns) -> INTERVIEWER, S2+ -> CANDIDATE
        role_map: Dict[str, SpeakerRole] = {}
        interviewer_label = labels[0]
        if "S1" in labels:
            interviewer_label = "S1"
        for lbl in labels:
            role_map[lbl] = SpeakerRole.INTERVIEWER if lbl == interviewer_label else SpeakerRole.CANDIDATE
        return role_map

    @staticmethod
    def _backoff_seconds(retry_after: Optional[str], attempt: int) -> float:
        if retry_after:
            try:
                return max(0.5, min(10.0, float(retry_after)))
            except ValueError:
                pass
        return 0.6 * (2 ** attempt) + random.uniform(0.05, 0.25)

    def _mock_diarized_transcription(self, session_id: str, offset_ms: int) -> List[UtteranceEvent]:
        """Used ONLY by STTRouter when zero STT API keys are configured in local demo mode."""
        words_q = [
            WordTiming(word="How", start_ms=offset_ms, end_ms=offset_ms + 300),
            WordTiming(word="do", start_ms=offset_ms + 350, end_ms=offset_ms + 500),
            WordTiming(word="you", start_ms=offset_ms + 550, end_ms=offset_ms + 700),
            WordTiming(word="scale", start_ms=offset_ms + 750, end_ms=offset_ms + 1100),
            WordTiming(word="distributed", start_ms=offset_ms + 1150, end_ms=offset_ms + 1700),
            WordTiming(word="queues?", start_ms=offset_ms + 1750, end_ms=offset_ms + 2300),
        ]
        q_utt = UtteranceEvent(
            session_id=session_id,
            utterance_id=f"utt_{session_id}_{offset_ms}_q",
            speaker=SpeakerRole.INTERVIEWER,
            speaker_label="S1",
            text="How do you scale distributed queues?",
            start_ms=offset_ms,
            end_ms=offset_ms + 2300,
            words=words_q,
        )

        words_a = [
            WordTiming(word="Certainly!", start_ms=offset_ms + 2800, end_ms=offset_ms + 3400),
            WordTiming(word="It", start_ms=offset_ms + 3500, end_ms=offset_ms + 3700),
            WordTiming(word="is", start_ms=offset_ms + 3750, end_ms=offset_ms + 3900),
            WordTiming(word="crucial", start_ms=offset_ms + 3950, end_ms=offset_ms + 4400),
            WordTiming(word="to", start_ms=offset_ms + 4450, end_ms=offset_ms + 4600),
            WordTiming(word="note", start_ms=offset_ms + 4650, end_ms=offset_ms + 4900),
            WordTiming(word="that", start_ms=offset_ms + 4950, end_ms=offset_ms + 5100),
            WordTiming(word="Kafka", start_ms=offset_ms + 5150, end_ms=offset_ms + 5600),
            WordTiming(word="partitions", start_ms=offset_ms + 5650, end_ms=offset_ms + 6200),
            WordTiming(word="play", start_ms=offset_ms + 6250, end_ms=offset_ms + 6500),
            WordTiming(word="a", start_ms=offset_ms + 6550, end_ms=offset_ms + 6650),
            WordTiming(word="pivotal", start_ms=offset_ms + 6700, end_ms=offset_ms + 7100),
            WordTiming(word="role.", start_ms=offset_ms + 7150, end_ms=offset_ms + 7600),
        ]
        a_utt = UtteranceEvent(
            session_id=session_id,
            utterance_id=f"utt_{session_id}_{offset_ms}_a",
            speaker=SpeakerRole.CANDIDATE,
            speaker_label="S2",
            text="Certainly! It is crucial to note that Kafka partitions play a pivotal role.",
            start_ms=offset_ms + 2800,
            end_ms=offset_ms + 7600,
            words=words_a,
        )
        return [q_utt, a_utt]