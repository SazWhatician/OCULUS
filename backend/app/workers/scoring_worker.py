import asyncio
import logging
import os
import re
import socket
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from sqlalchemy import select

from app.adapters.llm.base import FallbackLLMRouter, ForensicReasoningToolkit
from app.config import settings
from app.db.postgres import (
    IntegrityReportModel,
    QAEvaluationModel,
    SessionModel,
    VisionEventModel,
    async_session_factory,
)
from app.db.redis_bus import redis_bus
from app.schemas.contracts import (
    CandidateBaseline,
    QAPairEvaluation,
    SpeakerRole,
    UtteranceEvent,
    VisionAnomalyEvent,
    VisionAnomalyType,
)
from app.services.aggregator import (
    calculate_qa_combined_score,
    generate_integrity_report,
)
from app.services.heuristics import (
    compute_candidate_baseline,
    compute_deterministic_heuristics,
)
from app.telemetry import metrics

logger = logging.getLogger("oculus.worker.scoring")

MAX_ACTIVE_SESSIONS = 500
UTTERANCE_DEDUPE_WINDOW = 256
MIN_WORDS_FOR_BASELINE = 25
DEFAULT_MAX_LLM_TURNS_PER_SESSION = 8
DEFAULT_IDLE_FLUSH_SECONDS = 18.0
DEFAULT_SESSION_TTL_SECONDS = 900.0
SWEEPER_INTERVAL_SECONDS = 5.0
PRE_SPEECH_LOOKBACK_MS = 2_000
MAX_QUESTION_CHARS = 2_500

# Short interviewer active-listening cues that should NOT fragment a candidate's answer
BACKCHANNEL_RE = re.compile(
    r"^(?:uh[- ]?huh|mm[- ]?hmm|mhm|yeah|yep|yes|okay|ok|right|sure|got it|"
    r"i see|makes sense|cool|nice|great|interesting|go on|keep going|alright|"
    r"all right|totally|fair enough|understood|awesome)[\s.,!]*$",
    re.IGNORECASE,
)


@dataclass
class _SessionBuffer:
    session_id: str
    active_q: str = ""
    q_end_ms: int = 0
    ans_chunks: List[UtteranceEvent] = field(default_factory=list)
    ans_start_ms: int = 0
    ans_end_ms: int = 0
    qa_index: int = 0
    llm_calls_used: int = 0
    baseline_calibrated: bool = False
    hydrated_from_db: bool = False
    last_activity_mono: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seen_utt_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=UTTERANCE_DEDUPE_WINDOW))
    seen_utt_set: set[str] = field(default_factory=set)

    def mark_seen(self, utterance_id: str) -> bool:
        """Returns True if utterance_id is new; False if duplicate redelivery."""
        if not utterance_id:
            return True
        if utterance_id in self.seen_utt_set:
            return False
        if len(self.seen_utt_ids) == self.seen_utt_ids.maxlen:
            evicted = self.seen_utt_ids.popleft()
            self.seen_utt_set.discard(evicted)
        self.seen_utt_ids.append(utterance_id)
        self.seen_utt_set.add(utterance_id)
        return True


class ScoringWorker:
    def __init__(self, llm_router: Optional[FallbackLLMRouter] = None) -> None:
        self.llm_router = llm_router or FallbackLLMRouter()
        self._buffers: OrderedDict[str, _SessionBuffer] = OrderedDict()
        self.max_llm_turns_per_session: int = int(
            getattr(settings, "SCORING_MAX_LLM_TURNS", DEFAULT_MAX_LLM_TURNS_PER_SESSION)
        )
        self.idle_flush_seconds: float = float(
            getattr(settings, "SCORING_IDLE_FLUSH_SECONDS", DEFAULT_IDLE_FLUSH_SECONDS)
        )
        self.session_ttl_seconds: float = float(
            getattr(settings, "SCORING_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)
        )

    def _get_buffer(self, session_id: str) -> _SessionBuffer:
        buf = self._buffers.get(session_id)
        if buf is not None:
            self._buffers.move_to_end(session_id)
            return buf

        buf = _SessionBuffer(session_id=session_id)
        self._buffers[session_id] = buf
        if len(self._buffers) > MAX_ACTIVE_SESSIONS:
            evicted_id, _ = self._buffers.popitem(last=False)
            logger.info("Evicted oldest scoring buffer '%s' (cap=%d).", evicted_id, MAX_ACTIVE_SESSIONS)
        return buf

    async def handle_utterance(self, utt: UtteranceEvent) -> None:
        text = (utt.text or "").strip()
        if not text:
            return

        buf = self._get_buffer(utt.session_id)
        async with buf.lock:
            if not buf.mark_seen(utt.utterance_id):
                logger.debug("Skipped duplicate utterance '%s' for %s.", utt.utterance_id, utt.session_id)
                return

            buf.last_activity_mono = time.monotonic()

            if utt.speaker == SpeakerRole.INTERVIEWER:
                # Prevent brief active-listening cues from chopping a candidate's live answer
                if buf.ans_chunks and self._is_backchannel(text):
                    return

                if buf.ans_chunks:
                    await self._evaluate_and_flush_locked(buf)

                combined_q = f"{buf.active_q} {text}".strip()
                buf.active_q = combined_q[-MAX_QUESTION_CHARS:]
                buf.q_end_ms = max(buf.q_end_ms, utt.end_ms)

            elif utt.speaker == SpeakerRole.CANDIDATE:
                if not buf.ans_chunks:
                    buf.ans_start_ms = utt.start_ms
                else:
                    buf.ans_start_ms = min(buf.ans_start_ms, utt.start_ms)
                buf.ans_end_ms = max(buf.ans_end_ms, utt.end_ms)
                buf.ans_chunks.append(utt)

    async def flush_session(self, session_id: str) -> None:
        """Explicitly score any buffered final candidate turn when a session ends."""
        buf = self._buffers.get(session_id)
        if buf is None:
            return
        async with buf.lock:
            if buf.ans_chunks:
                await self._evaluate_and_flush_locked(buf)

    async def flush_all_pending(self) -> None:
        """Flushes all buffered answers safely during worker shutdown."""
        for session_id in list(self._buffers.keys()):
            try:
                await self.flush_session(session_id)
            except Exception as exc:
                logger.error("Shutdown flush failed for session %s: %s", session_id, exc)

    async def sweep_idle_sessions(self) -> None:
        """
        Scores final candidate answers once silence exceeds `idle_flush_seconds`
        and evicts stale session buffers after `session_ttl_seconds`.
        """
        now = time.monotonic()
        for session_id in list(self._buffers.keys()):
            buf = self._buffers.get(session_id)
            if buf is None:
                continue

            idle_for = now - buf.last_activity_mono
            if buf.ans_chunks and idle_for >= self.idle_flush_seconds:
                async with buf.lock:
                    if buf.ans_chunks and (time.monotonic() - buf.last_activity_mono) >= self.idle_flush_seconds:
                        try:
                            await self._evaluate_and_flush_locked(buf)
                        except Exception as exc:
                            logger.error("Idle flush failed for %s: %s", session_id, exc, exc_info=True)

            elif not buf.ans_chunks and idle_for >= self.session_ttl_seconds:
                self._buffers.pop(session_id, None)

    async def aclose(self) -> None:
        await self.flush_all_pending()
        await self.llm_router.aclose()

    async def _hydrate_session_state(self, buf: _SessionBuffer) -> None:
        """Recovers qa_index, baseline flag, and LLM call count from Postgres after worker restart."""
        if buf.hydrated_from_db:
            return

        async with async_session_factory() as db:
            session_obj = await db.get(SessionModel, buf.session_id)
            if session_obj and session_obj.baseline_json:
                buf.baseline_calibrated = True

            res = await db.execute(
                select(QAEvaluationModel).where(QAEvaluationModel.session_id == buf.session_id)
            )
            rows = res.scalars().all()
            if rows:
                buf.qa_index = max(r.qa_index for r in rows) + 1
                buf.llm_calls_used = sum(
                    1
                    for r in rows
                    if isinstance(r.payload_json, dict) and r.payload_json.get("llm_verdict") is not None
                )

        buf.hydrated_from_db = True

    async def _evaluate_and_flush_locked(self, buf: _SessionBuffer) -> None:
        if not buf.ans_chunks:
            return

        await self._hydrate_session_state(buf)

        session_id = buf.session_id
        chunks = list(buf.ans_chunks)
        full_answer = " ".join(u.text.strip() for u in chunks if u.text and u.text.strip()).strip()
        ans_start_ms = buf.ans_start_ms
        ans_end_ms = max(buf.ans_end_ms, ans_start_ms)
        q_end_ms = buf.q_end_ms
        question_text = buf.active_q or "General Question"
        qa_index = buf.qa_index

        # Reset turn buffer before running I/O so an exception never replays corrupted state
        buf.ans_chunks = []
        buf.active_q = ""
        buf.q_end_ms = 0
        buf.ans_start_ms = 0
        buf.ans_end_ms = 0

        if not full_answer:
            return

        buf.qa_index += 1
        word_count = len(full_answer.split())
        duration_ms = max(ans_end_ms - ans_start_ms, 1_000)
        ttfs_latency_ms = max(ans_start_ms - q_end_ms, 0) if (q_end_ms > 0 and ans_start_ms >= q_end_ms) else 0
        word_dicts, inter_word_gaps = self._extract_word_timings_and_gaps(chunks)

        # Accurate bounded vision lookback window (never widens to t=0 when q_end_ms == 0)
        if q_end_ms > 0:
            lookback_start = max(0, min(q_end_ms - PRE_SPEECH_LOOKBACK_MS, ans_start_ms - PRE_SPEECH_LOOKBACK_MS))
            lookback_start = max(lookback_start, ans_start_ms - 15_000, 0)
        else:
            lookback_start = max(0, ans_start_ms - 3_000)

        # Single DB round-trip for session metadata + concurrent vision events
        baseline: Optional[CandidateBaseline] = None
        job_title = "Software Engineer"
        candidate_name = "Candidate"
        concurrent_vision: List[VisionAnomalyEvent] = []

        async with async_session_factory() as db:
            session_obj = await db.get(SessionModel, session_id)
            if session_obj:
                job_title = session_obj.job_title or job_title
                candidate_name = session_obj.candidate_name or candidate_name
                if session_obj.baseline_json:
                    try:
                        baseline = CandidateBaseline.model_validate(session_obj.baseline_json)
                        buf.baseline_calibrated = True
                    except Exception as exc:
                        logger.warning("Corrupt baseline_json for %s: %s", session_id, exc)

            v_stmt = (
                select(VisionEventModel)
                .where(VisionEventModel.session_id == session_id)
                .where(VisionEventModel.timestamp_ms >= lookback_start)
                .where(VisionEventModel.timestamp_ms <= ans_end_ms)
                .order_by(VisionEventModel.timestamp_ms.asc())
            )
            v_res = await db.execute(v_stmt)
            for row in v_res.scalars().all():
                parsed_ev = self._row_to_vision_event(row)
                if parsed_ev is not None:
                    concurrent_vision.append(parsed_ev)

        # Stage 0: Calibrate baseline on first substantive turn (>= 25 words), ignoring mic checks
        is_calibration_turn = False
        if baseline is None and word_count >= MIN_WORDS_FOR_BASELINE:
            baseline = compute_candidate_baseline(
                session_id=session_id,
                q0_text=full_answer,
                duration_ms=duration_ms,
                words=word_dicts or None,
                ttfs_ms=float(ttfs_latency_ms or 1_200),
            )
            buf.baseline_calibrated = True
            is_calibration_turn = True

            async with async_session_factory() as db:
                session_obj = await db.get(SessionModel, session_id)
                if session_obj:
                    session_obj.baseline_json = baseline.model_dump(mode="json")
                    await db.commit()

        # Stage 1: Full deterministic heuristics (now with inter_word_gaps + question_text)
        heuristics = compute_deterministic_heuristics(
            answer_text=full_answer,
            duration_ms=duration_ms,
            baseline=baseline,
            inter_word_gaps=inter_word_gaps or None,
            ttfs_latency_ms=ttfs_latency_ms,
            concurrent_vision_events=concurrent_vision,
            question_text=question_text,
        )

        # Stage 2: Selective, budget-capped LLM Judge with security/hard-vision override
        injection_hits = ForensicReasoningToolkit.detect_injection_attempts(full_answer)
        has_hard_vision = any(
            ev.anomaly_type.value in ("DEVICE_DETECTED", "MULTI_FACE") for ev in concurrent_vision
        )
        within_budget = buf.llm_calls_used < self.max_llm_turns_per_session
        should_invoke_llm = (
            bool(injection_hits)
            or (has_hard_vision and within_budget)
            or (heuristics.trigger_llm_judge and not is_calibration_turn and within_budget)
        )

        llm_verdict = None
        if should_invoke_llm:
            metrics.inc("llm_judge_invocations")
            buf.llm_calls_used += 1
            try:
                llm_verdict = await self.llm_router.evaluate_turn(
                    question_text=question_text,
                    answer_text=full_answer,
                    job_title=job_title,
                    baseline=baseline,
                    heuristics=heuristics,
                )
            except Exception as exc:
                logger.error("LLM router failed on %s Q%d: %s", session_id, qa_index, exc)
                metrics.inc("llm_judge_failures")
        else:
            metrics.inc("prefilter_skips")

        combined_score = calculate_qa_combined_score(heuristics, llm_verdict)
        qa_eval = QAPairEvaluation(
            qa_index=qa_index,
            question_text=question_text,
            answer_text=full_answer,
            question_end_ms=q_end_ms,
            answer_start_ms=ans_start_ms,
            answer_end_ms=ans_end_ms,
            heuristics=heuristics,
            llm_verdict=llm_verdict,
            combined_risk_score=combined_score,
        )

        qa_payload = qa_eval.model_dump(mode="json")

        # Idempotent UPSERT + de-duplicated re-aggregation (fixes SQLAlchemy autoflush double-counting)
        async with async_session_factory() as db:
            qa_id = f"qa_{session_id}_{qa_index}"
            existing_qa = await db.get(QAEvaluationModel, qa_id)
            if existing_qa is None:
                db.add(
                    QAEvaluationModel(
                        id=qa_id,
                        session_id=session_id,
                        qa_index=qa_index,
                        combined_risk_score=combined_score,
                        payload_json=qa_payload,
                    )
                )
            else:
                existing_qa.combined_risk_score = combined_score
                existing_qa.payload_json = qa_payload

            await db.flush()

            all_qas_res = await db.execute(
                select(QAEvaluationModel)
                .where(QAEvaluationModel.session_id == session_id)
                .order_by(QAEvaluationModel.qa_index.asc())
            )
            eval_by_idx: Dict[int, QAPairEvaluation] = {}
            for row in all_qas_res.scalars().all():
                eval_by_idx[row.qa_index] = QAPairEvaluation.model_validate(row.payload_json)
            eval_by_idx[qa_eval.qa_index] = qa_eval
            evaluations = [eval_by_idx[idx] for idx in sorted(eval_by_idx)]

            all_v_res = await db.execute(
                select(VisionEventModel)
                .where(VisionEventModel.session_id == session_id)
                .order_by(VisionEventModel.timestamp_ms.asc())
            )
            all_vision = [
                ev
                for row in all_v_res.scalars().all()
                if (ev := self._row_to_vision_event(row)) is not None
            ]

            report = generate_integrity_report(
                session_id=session_id,
                candidate_name=candidate_name,
                job_title=job_title,
                qa_evaluations=evaluations,
                vision_timeline=all_vision,
                baseline=baseline,
            )
            report_payload = report.model_dump(mode="json")

            rep_model = await db.get(IntegrityReportModel, session_id)
            if rep_model is None:
                db.add(
                    IntegrityReportModel(
                        session_id=session_id,
                        overall_risk_score=report.overall_risk_score,
                        risk_band=report.risk_band.value,
                        report_json=report_payload,
                    )
                )
            else:
                rep_model.overall_risk_score = report.overall_risk_score
                rep_model.risk_band = report.risk_band.value
                rep_model.report_json = report_payload

            await db.commit()

        await redis_bus.broadcast_ui(
            session_id=session_id,
            event_type="QA_EVALUATED",
            data={
                "qa_evaluation": qa_payload,
                "overall_risk_score": report.overall_risk_score,
                "risk_band": report.risk_band.value,
                "baseline_calibrated": buf.baseline_calibrated,
            },
        )

    @staticmethod
    def _is_backchannel(text: str) -> bool:
        if "?" in text or len(text.split()) > 3:
            return False
        return bool(BACKCHANNEL_RE.match(text.strip()))

    @staticmethod
    def _extract_word_timings_and_gaps(
        chunks: List[UtteranceEvent],
    ) -> Tuple[List[Dict[str, int]], List[int]]:
        raw_timings: List[Tuple[int, int]] = []
        for utt in chunks:
            for w in utt.words or []:
                s = w.get("start_ms") if isinstance(w, dict) else getattr(w, "start_ms", None)
                e = w.get("end_ms") if isinstance(w, dict) else getattr(w, "end_ms", None)
                if s is not None and e is not None and int(e) >= int(s) >= 0:
                    raw_timings.append((int(s), int(e)))

        if not raw_timings:
            return ([], [])

        raw_timings.sort(key=lambda pair: pair[0])
        word_dicts = [{"start_ms": s, "end_ms": e} for s, e in raw_timings]
        gaps = [
            raw_timings[i][0] - raw_timings[i - 1][1]
            for i in range(1, len(raw_timings))
            if raw_timings[i][0] > raw_timings[i - 1][1]
        ]
        return (word_dicts, gaps)

    @staticmethod
    def _row_to_vision_event(row: VisionEventModel) -> Optional[VisionAnomalyEvent]:
        try:
            anomaly_type = VisionAnomalyType(row.anomaly_type)
        except ValueError:
            logger.warning("Skipping unrecognized VisionAnomalyType '%s' on row %s.", row.anomaly_type, row.id)
            return None

        return VisionAnomalyEvent(
            event_id=row.id,
            session_id=row.session_id,
            timestamp_ms=row.timestamp_ms,
            anomaly_type=anomaly_type,
            duration_ms=row.duration_ms,
            confidence=row.confidence,
            evidence_frame_key=row.evidence_frame_key,
            vlm_verification_note=row.vlm_verification_note,
        )


async def _idle_sweeper_loop(
    worker: ScoringWorker,
    stop_event: Optional[asyncio.Event],
) -> None:
    while stop_event is None or not stop_event.is_set():
        try:
            if stop_event is not None:
                await asyncio.wait_for(stop_event.wait(), timeout=SWEEPER_INTERVAL_SECONDS)
                break
            await asyncio.sleep(SWEEPER_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass

        try:
            await worker.sweep_idle_sessions()
        except Exception as exc:
            logger.error("Idle sweeper error: %s", exc, exc_info=True)


async def run_scoring_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    worker = ScoringWorker()
    consumer_id = f"scoring_{socket.gethostname()}_{os.getpid()}"
    sweeper_task = asyncio.create_task(_idle_sweeper_loop(worker, stop_event))
    logger.info("ScoringWorker '%s' started consuming from 'stream:utterances'", consumer_id)

    try:
        await redis_bus.consume_stream(
            stream="stream:utterances",
            group="scoring_workers",
            consumer=consumer_id,
            model_cls=UtteranceEvent,
            handler=worker.handle_utterance,
            stop_event=stop_event,
        )
    finally:
        sweeper_task.cancel()
        await asyncio.gather(sweeper_task, return_exceptions=True)
        await worker.aclose()
        logger.info("ScoringWorker '%s' shut down cleanly.", consumer_id)