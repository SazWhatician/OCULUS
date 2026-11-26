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
from app.services.vision_proctoring import get_vision_service
from app.telemetry import metrics

try:
    from app.services.heuristics import clear_session_style_profile
except ImportError:  # Backward compatibility if older heuristics.py is loaded
    def clear_session_style_profile(session_id: str) -> None:
        return None

logger = logging.getLogger("oculus.worker.scoring")

_active_scoring_worker: Optional["ScoringWorker"] = None


def get_active_scoring_worker() -> Optional["ScoringWorker"]:
    return _active_scoring_worker


MAX_ACTIVE_SESSIONS = 500
UTTERANCE_DEDUPE_WINDOW = 256
MIN_WORDS_FOR_BASELINE = 12
TARGET_BASELINE_WORDS = 65
DEFAULT_MAX_LLM_TURNS_PER_SESSION = 8
DEFAULT_LLM_CONCURRENCY = 1
DEFAULT_IDLE_FLUSH_SECONDS = 18.0
DEFAULT_SESSION_TTL_SECONDS = 900.0
SWEEPER_INTERVAL_SECONDS = 5.0
PRE_SPEECH_LOOKBACK_MS = 2_000
VISION_TAIL_WINDOW_MS = 1_500
STALE_QUESTION_RESET_MS = 25_000
MAX_QUESTION_CHARS = 2_500

# Page's One-Sided CUSUM Regime-Shift Constants
CUSUM_SLACK_K = 0.65
CUSUM_ALERT_H = 2.60
CUSUM_MIN_TURN_WORDS = 25

HARD_VISION_TYPES = frozenset({
    "DEVICE_DETECTED",
    "PHONE_DETECTED",
    "MULTI_FACE",
    "LIVENESS_SPOOF",
})

# Short interviewer active-listening cues that must NOT fragment a candidate's live answer
BACKCHANNEL_RE = re.compile(
    r"^(?:uh[- ]?huh|mm[- ]?hmm|mhm|yeah|yep|yes|okay|ok|right|sure|got it|"
    r"i see|makes sense|cool|nice|great|interesting|go on|keep going|alright|"
    r"all right|totally|fair enough|understood|awesome)[\s.,!]*$",
    re.IGNORECASE,
)


def _coerce_role(raw_role: Any) -> SpeakerRole:
    if isinstance(raw_role, SpeakerRole):
        return raw_role
    if isinstance(raw_role, str):
        norm = raw_role.strip().lower()
        try:
            return SpeakerRole(norm)
        except ValueError:
            try:
                return SpeakerRole[norm.upper()]
            except KeyError:
                return SpeakerRole.UNKNOWN
    return SpeakerRole.UNKNOWN


def _normalize_anomaly_type(raw_type: Any) -> Optional[VisionAnomalyType]:
    if isinstance(raw_type, VisionAnomalyType):
        return raw_type
    if isinstance(raw_type, str):
        val = raw_type.strip().upper()
        try:
            return VisionAnomalyType(val)
        except ValueError:
            try:
                return VisionAnomalyType[val]
            except KeyError:
                return None
    return None


@dataclass
class _SessionCusumTracker:
    """
    One-sided Page's CUSUM detector tracking mid-interview behavioral regime shifts
    (e.g., honest Q0-Q1 followed by AI-assisted Q2-Q4) relative to the candidate's
    own calm-turn risk baseline.
    """
    cusum: float = 0.0
    honest_risk_ema: float = 14.0
    honest_turns: int = 0
    onset_qa_index: Optional[int] = None

    def update(
        self,
        qa_index: int,
        word_count: int,
        deterministic_risk: float,
        wpm_z: float,
        disfluency_drop: float,
        disfluency_rate: float,
        ttfs_excess_ms: float,
        concurrent_vision_flags: int,
        ai_tell_count: int,
    ) -> Tuple[bool, float, Optional[int]]:
        """Returns (regime_shift_detected, cusum_score, onset_qa_index)."""
        if word_count < CUSUM_MIN_TURN_WORDS:
            return (False, round(self.cusum, 2), self.onset_qa_index)

        z_wpm = min(3.5, max(0.0, abs(wpm_z) - 1.0))
        z_disf = min(3.0, max(0.0, disfluency_drop / 1.8))
        z_ttfs = min(2.5, max(0.0, ttfs_excess_ms / 1_500.0)) if disfluency_rate < 1.2 else 0.0
        z_risk = min(3.5, max(0.0, (deterministic_risk - self.honest_risk_ema) / 16.0))
        z_vis = 1.1 if concurrent_vision_flags > 0 else 0.0

        evidence = (0.30 * z_wpm) + (0.30 * z_disf) + (0.15 * z_ttfs) + (0.25 * z_risk) + z_vis
        prev = self.cusum
        self.cusum = max(0.0, prev + (evidence - CUSUM_SLACK_K))

        if self.cusum == 0.0:
            self.onset_qa_index = None
        elif prev == 0.0 and self.cusum > 0.0:
            self.onset_qa_index = qa_index

        # Only update honest reference EMA on genuinely calm turns so sustained cheating cannot drag it up
        if deterministic_risk < 25.0 and ai_tell_count == 0 and concurrent_vision_flags == 0:
            alpha = 0.35 if self.honest_turns < 3 else 0.18
            self.honest_risk_ema = (1.0 - alpha) * self.honest_risk_ema + alpha * deterministic_risk
            self.honest_turns += 1

        alert = self.cusum >= CUSUM_ALERT_H and self.onset_qa_index is not None
        return (alert, round(self.cusum, 2), self.onset_qa_index)


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
    llm_attempts: int = 0
    baseline_calibrated: bool = False
    hydrated_from_db: bool = False
    cusum: _SessionCusumTracker = field(default_factory=_SessionCusumTracker)
    last_activity_mono: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seen_utt_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=UTTERANCE_DEDUPE_WINDOW))
    seen_utt_set: set[str] = field(default_factory=set)

    def mark_seen(self, utterance_id: str) -> bool:
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
    """
    Multimodal Live Scoring Worker:
      - Anti-poisoned Empirical Bayes baseline calibration + multi-turn clean refinement
      - Page's One-Sided CUSUM mid-interview behavioral change-point detection
      - 3-tier priority-reserved LLM budget manager + RPM concurrency semaphore
      - DB + in-memory vision window fusion with +1.5s debounce tail & 3D pose retention
      - Idempotent DB UPSERT, restart hydration, lock-safe idle/TTL sweeping, and clean shutdown
    """

    def __init__(self, llm_router: Optional[FallbackLLMRouter] = None) -> None:
        self.llm_router = llm_router or FallbackLLMRouter()
        self._buffers: "OrderedDict[str, _SessionBuffer]" = OrderedDict()
        self.max_llm_turns_per_session: int = max(
            1, int(getattr(settings, "SCORING_MAX_LLM_TURNS", DEFAULT_MAX_LLM_TURNS_PER_SESSION))
        )
        self.reserved_high_priority_slots: int = min(
            2, max(1, self.max_llm_turns_per_session // 3)
        ) if self.max_llm_turns_per_session >= 3 else 0
        self.idle_flush_seconds: float = float(
            getattr(settings, "SCORING_IDLE_FLUSH_SECONDS", DEFAULT_IDLE_FLUSH_SECONDS)
        )
        self.session_ttl_seconds: float = float(
            getattr(settings, "SCORING_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)
        )
        llm_conc = max(1, int(getattr(settings, "SCORING_LLM_CONCURRENCY", DEFAULT_LLM_CONCURRENCY)))
        self._llm_semaphore = asyncio.Semaphore(llm_conc)

    def _get_buffer(self, session_id: str) -> _SessionBuffer:
        buf = self._buffers.get(session_id)
        if buf is not None:
            self._buffers.move_to_end(session_id)
            return buf

        buf = _SessionBuffer(session_id=session_id)
        self._buffers[session_id] = buf
        if len(self._buffers) > MAX_ACTIVE_SESSIONS:
            evicted_id, _ = self._buffers.popitem(last=False)
            clear_session_style_profile(evicted_id)
            logger.info("Evicted oldest scoring buffer '%s' (cap=%d).", evicted_id, MAX_ACTIVE_SESSIONS)
        return buf

    async def handle_utterance(self, utt: UtteranceEvent) -> None:
        text = (utt.text or "").strip()
        if not text:
            return

        role = _coerce_role(utt.speaker)
        buf = self._get_buffer(utt.session_id)

        async with buf.lock:
            if not buf.mark_seen(utt.utterance_id):
                logger.debug("Skipped duplicate utterance '%s' for %s.", utt.utterance_id, utt.session_id)
                return

            buf.last_activity_mono = time.monotonic()

            if role == SpeakerRole.INTERVIEWER:
                if buf.ans_chunks and self._is_backchannel(text):
                    return

                if buf.ans_chunks:
                    await self._evaluate_and_flush_locked(buf)

                # Reset stale unanswered question if a long gap elapsed before a fresh prompt
                if buf.active_q and buf.q_end_ms > 0 and (utt.start_ms - buf.q_end_ms) > STALE_QUESTION_RESET_MS:
                    buf.active_q = ""

                combined_q = f"{buf.active_q} {text}".strip()
                buf.active_q = combined_q[-MAX_QUESTION_CHARS:]
                buf.q_end_ms = max(buf.q_end_ms, utt.end_ms)

            elif role == SpeakerRole.CANDIDATE:
                if not buf.ans_chunks:
                    buf.ans_start_ms = utt.start_ms
                else:
                    buf.ans_start_ms = min(buf.ans_start_ms, utt.start_ms)
                buf.ans_end_ms = max(buf.ans_end_ms, utt.end_ms)
                buf.ans_chunks.append(utt)

    async def flush_session(self, session_id: str) -> None:
        buf = self._buffers.get(session_id)
        if buf is None:
            return
        async with buf.lock:
            if buf.ans_chunks:
                await self._evaluate_and_flush_locked(buf)

    async def flush_all_pending(self) -> None:
        for session_id in list(self._buffers.keys()):
            try:
                await self.flush_session(session_id)
            except Exception as exc:
                logger.error("Shutdown flush failed for session %s: %s", session_id, exc)

    async def sweep_idle_sessions(self) -> None:
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
                async with buf.lock:
                    if not buf.ans_chunks and (time.monotonic() - buf.last_activity_mono) >= self.session_ttl_seconds:
                        self._buffers.pop(session_id, None)
                        clear_session_style_profile(session_id)

    def reset_session(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)
        clear_session_style_profile(session_id)

    async def aclose(self) -> None:
        await self.flush_all_pending()
        await self.llm_router.aclose()

    async def _hydrate_session_state(self, buf: _SessionBuffer) -> None:
        """Recovers qa_index, baseline status, LLM quota, and CUSUM trajectory after worker restart."""
        if buf.hydrated_from_db:
            return

        async with async_session_factory() as db:
            session_obj = await db.get(SessionModel, buf.session_id)
            if session_obj and session_obj.baseline_json:
                buf.baseline_calibrated = True

            res = await db.execute(
                select(QAEvaluationModel)
                .where(QAEvaluationModel.session_id == buf.session_id)
                .order_by(QAEvaluationModel.qa_index.asc())
            )
            rows = res.scalars().all()
            if rows:
                buf.qa_index = max(r.qa_index for r in rows) + 1
                buf.llm_calls_used = sum(
                    1
                    for r in rows
                    if isinstance(r.payload_json, dict) and r.payload_json.get("llm_verdict") is not None
                )
                buf.llm_attempts = buf.llm_calls_used

                # Replay lightweight CUSUM state across prior persisted turns
                for r in rows:
                    payload = r.payload_json if isinstance(r.payload_json, dict) else {}
                    h = payload.get("heuristics") if isinstance(payload.get("heuristics"), dict) else {}
                    if h:
                        buf.cusum.update(
                            qa_index=r.qa_index,
                            word_count=int(h.get("word_count", 0) or 0),
                            deterministic_risk=float(h.get("deterministic_risk", 0.0) or 0.0),
                            wpm_z=float(h.get("wpm_robust_zscore", 0.0) or 0.0),
                            disfluency_drop=float(h.get("disfluency_drop", 0.0) or 0.0),
                            disfluency_rate=float(h.get("disfluency_rate", 3.0) or 3.0),
                            ttfs_excess_ms=max(0.0, float(h.get("ttfs_latency_ms", 0) or 0) - 1_200.0),
                            concurrent_vision_flags=int(h.get("concurrent_vision_flags", 0) or 0),
                            ai_tell_count=len(h.get("ai_tell_matches") or []),
                        )

        buf.hydrated_from_db = True

    async def _evaluate_and_flush_locked(self, buf: _SessionBuffer) -> None:
        if not buf.ans_chunks:
            return

        await self._hydrate_session_state(buf)

        session_id = buf.session_id
        chunks = sorted(buf.ans_chunks, key=lambda u: (u.start_ms, u.end_ms))
        full_answer = self._join_answer_chunks(chunks)
        ans_start_ms = buf.ans_start_ms
        ans_end_ms = max(buf.ans_end_ms, ans_start_ms)
        q_end_ms = buf.q_end_ms
        question_text = buf.active_q or "General Question"
        qa_index = buf.qa_index

        # Reset turn buffer before I/O so an exception never replays corrupted state
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

        # Bounded lookback + 1.5s tail captures debounced gaze events completing right at turn end
        if q_end_ms > 0:
            lookback_start = max(0, min(q_end_ms - PRE_SPEECH_LOOKBACK_MS, ans_start_ms - PRE_SPEECH_LOOKBACK_MS))
            lookback_start = max(lookback_start, ans_start_ms - 15_000, 0)
        else:
            lookback_start = max(0, ans_start_ms - 3_000)
        window_end_ms = ans_end_ms + VISION_TAIL_WINDOW_MS

        baseline: Optional[CandidateBaseline] = None
        job_title = "Software Engineer"
        candidate_name = "Candidate"
        db_vision_events: List[VisionAnomalyEvent] = []

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
                .where(VisionEventModel.timestamp_ms <= window_end_ms)
                .order_by(VisionEventModel.timestamp_ms.asc())
            )
            v_res = await db.execute(v_stmt)
            for row in v_res.scalars().all():
                parsed_ev = self._row_to_vision_event(row)
                if parsed_ev is not None:
                    db_vision_events.append(parsed_ev)

        # Fuse Postgres vision rows with live in-memory VisionProctoringService events (single-node lag guard)
        concurrent_vision = self._merge_live_vision_events(
            session_id=session_id,
            start_ms=lookback_start,
            end_ms=window_end_ms,
            db_events=db_vision_events,
        )

        injection_hits = ForensicReasoningToolkit.detect_injection_attempts(full_answer)
        has_hard_vision = any(
            getattr(ev.anomaly_type, "value", str(ev.anomaly_type)).upper() in HARD_VISION_TYPES
            for ev in concurrent_vision
        )

        # Stage 0: Anti-Poisoned Empirical Bayes Baseline Calibration
        is_calibration_turn = False
        baseline_dirty = False
        if baseline is None and word_count >= MIN_WORDS_FOR_BASELINE:
            # Pre-screen against population priors so a cheated Q0 cannot poison the session baseline
            probe_h = compute_deterministic_heuristics(
                answer_text=full_answer,
                duration_ms=duration_ms,
                baseline=None,
                inter_word_gaps=inter_word_gaps or None,
                ttfs_latency_ms=ttfs_latency_ms,
                concurrent_vision_events=concurrent_vision,
                question_text=question_text,
            )
            is_poisoned = (
                bool(injection_hits)
                or has_hard_vision
                or len(probe_h.ai_tell_matches) >= 2
                or probe_h.deterministic_risk >= 38.0
            )
            if not is_poisoned:
                baseline = compute_candidate_baseline(
                    session_id=session_id,
                    q0_text=full_answer,
                    duration_ms=duration_ms,
                    words=word_dicts or None,
                    ttfs_ms=float(ttfs_latency_ms or 1_200),
                )
                baseline = self._attach_vision_baseline_variance(session_id, baseline)
                buf.baseline_calibrated = True
                is_calibration_turn = True
                baseline_dirty = True
            else:
                logger.info(
                    "Skipped Q0 baseline lock for %s (anti-poisoning guard: risk=%.1f, tells=%d, hard_vision=%s).",
                    session_id,
                    probe_h.deterministic_risk,
                    len(probe_h.ai_tell_matches),
                    has_hard_vision,
                )

        # Stage 1: Deterministic Forensic Heuristics
        heuristics = compute_deterministic_heuristics(
            answer_text=full_answer,
            duration_ms=duration_ms,
            baseline=baseline,
            inter_word_gaps=inter_word_gaps or None,
            ttfs_latency_ms=ttfs_latency_ms,
            concurrent_vision_events=concurrent_vision,
            question_text=question_text,
        )

        # Online Bayesian Baseline Refinement: stabilize short Q0 baselines using subsequent clean turns
        if (
            baseline is not None
            and not is_calibration_turn
            and baseline.sample_word_count < TARGET_BASELINE_WORDS
            and word_count >= 25
            and heuristics.deterministic_risk < 18.0
            and not heuristics.ai_tell_matches
            and not concurrent_vision
            and not injection_hits
        ):
            baseline = self._refine_baseline_with_clean_turn(
                baseline=baseline,
                clean_text=full_answer,
                duration_ms=duration_ms,
                word_dicts=word_dicts or None,
                ttfs_ms=float(ttfs_latency_ms or baseline.baseline_ttfs_ms),
            )
            baseline = self._attach_vision_baseline_variance(session_id, baseline)
            baseline_dirty = True

        if baseline_dirty and baseline is not None:
            async with async_session_factory() as db:
                session_obj = await db.get(SessionModel, session_id)
                if session_obj:
                    session_obj.baseline_json = baseline.model_dump(mode="json")
                    await db.commit()

        # Cross-Turn Page's CUSUM Change-Point Detection
        base_ttfs = baseline.baseline_ttfs_ms if baseline else 1_200.0
        ttfs_excess_ms = max(0.0, float(ttfs_latency_ms) - max(base_ttfs, 1_000.0))
        regime_shift, cusum_score, onset_idx = (False, round(buf.cusum.cusum, 2), buf.cusum.onset_qa_index)
        if not is_calibration_turn:
            regime_shift, cusum_score, onset_idx = buf.cusum.update(
                qa_index=qa_index,
                word_count=word_count,
                deterministic_risk=heuristics.deterministic_risk,
                wpm_z=heuristics.wpm_robust_zscore,
                disfluency_drop=heuristics.disfluency_drop,
                disfluency_rate=heuristics.disfluency_rate,
                ttfs_excess_ms=ttfs_excess_ms,
                concurrent_vision_flags=heuristics.concurrent_vision_flags,
                ai_tell_count=len(heuristics.ai_tell_matches),
            )
            if regime_shift:
                note = (
                    f"Mid-interview behavioral regime shift detected "
                    f"(CUSUM={cusum_score:.2f}, onset Q{onset_idx} -> Q{qa_index})"
                )
                if note not in heuristics.trigger_reasons:
                    heuristics = heuristics.model_copy(
                        update={
                            "trigger_reasons": [*heuristics.trigger_reasons, note],
                            "trigger_llm_judge": True,
                        }
                    )

        # Stage 2: 3-Tier Priority-Reserved LLM Judge Gating
        should_invoke_llm = self._should_invoke_llm(
            buf=buf,
            heuristics=heuristics,
            injection_hits=injection_hits,
            has_hard_vision=has_hard_vision,
            regime_shift=regime_shift,
            is_calibration_turn=is_calibration_turn,
        )

        llm_verdict = None
        if should_invoke_llm:
            metrics.inc("llm_judge_invocations")
            buf.llm_attempts += 1
            try:
                async with self._llm_semaphore:
                    llm_verdict = await self.llm_router.evaluate_turn(
                        question_text=question_text,
                        answer_text=full_answer,
                        job_title=job_title,
                        baseline=baseline,
                        heuristics=heuristics,
                    )
                # Charge session budget only when a live verdict is actually produced
                if llm_verdict is not None:
                    buf.llm_calls_used += 1
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

        # Idempotent UPSERT + de-duplicated session re-aggregation
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
                "cusum_score": cusum_score,
                "regime_shift_detected": regime_shift,
                "llm_budget_remaining": max(0, self.max_llm_turns_per_session - buf.llm_calls_used),
            },
        )

    def _should_invoke_llm(
        self,
        buf: _SessionBuffer,
        heuristics: Any,
        injection_hits: List[str],
        has_hard_vision: bool,
        regime_shift: bool,
        is_calibration_turn: bool,
    ) -> bool:
        """
        3-tier budget policy preventing early borderline turns from starving late high-risk turns:
          - Tier 0 (Security override): spoken prompt injection always escalates.
          - Tier 1 (Critical): hard vision, risk >= 60, >= 3 tells, or CUSUM shift can use reserved slots.
          - Tier 2 (Standard): normal pre-filter triggers only draw from the unreserved general pool.
        """
        # Safety cap on failed attempts so prolonged API outages never hammer providers indefinitely
        max_attempts_guard = self.max_llm_turns_per_session + 4
        if injection_hits:
            return buf.llm_attempts < (max_attempts_guard + 2)

        if buf.llm_calls_used >= self.max_llm_turns_per_session or buf.llm_attempts >= max_attempts_guard:
            return False

        is_critical = (
            has_hard_vision
            or heuristics.deterministic_risk >= 60.0
            or len(heuristics.ai_tell_matches) >= 3
            or regime_shift
        )
        if is_critical:
            return True

        general_pool_limit = max(1, self.max_llm_turns_per_session - self.reserved_high_priority_slots)
        if is_calibration_turn:
            return False
        return bool(heuristics.trigger_llm_judge) and buf.llm_calls_used < general_pool_limit

    @staticmethod
    def _refine_baseline_with_clean_turn(
        baseline: CandidateBaseline,
        clean_text: str,
        duration_ms: int,
        word_dicts: Optional[List[Dict[str, int]]],
        ttfs_ms: float,
    ) -> CandidateBaseline:
        """Sample-size-weighted Empirical Bayes refinement on clean follow-up turns."""
        fresh = compute_candidate_baseline(
            session_id=baseline.session_id,
            q0_text=clean_text,
            duration_ms=duration_ms,
            words=word_dicts,
            ttfs_ms=ttfs_ms,
        )
        w_old = max(float(baseline.sample_word_count), 12.0)
        w_new = min(float(fresh.sample_word_count), 60.0)
        w_tot = w_old + w_new

        return baseline.model_copy(
            update={
                "baseline_wpm": round((w_old * baseline.baseline_wpm + w_new * fresh.baseline_wpm) / w_tot, 1),
                "mad_wpm": round((w_old * baseline.mad_wpm + w_new * fresh.mad_wpm) / w_tot, 1),
                "baseline_disfluency_rate": round(
                    (w_old * baseline.baseline_disfluency_rate + w_new * fresh.baseline_disfluency_rate) / w_tot, 2
                ),
                "baseline_pause_ratio": round(
                    (w_old * baseline.baseline_pause_ratio + w_new * fresh.baseline_pause_ratio) / w_tot, 3
                ),
                "baseline_ttfs_ms": round(
                    (w_old * baseline.baseline_ttfs_ms + w_new * fresh.baseline_ttfs_ms) / w_tot, 1
                ),
                "sample_word_count": int(min(150, baseline.sample_word_count + fresh.sample_word_count)),
            }
        )

    @staticmethod
    def _attach_vision_baseline_variance(
        session_id: str,
        baseline: CandidateBaseline,
    ) -> CandidateBaseline:
        try:
            summary = get_vision_service().get_session_summary(session_id)
            var = float(summary.get("baseline_gaze_variance") or 0.0)
            if var > 0.0:
                return baseline.model_copy(update={"baseline_gaze_variance": round(var, 4)})
        except Exception:
            pass
        return baseline

    @staticmethod
    def _merge_live_vision_events(
        session_id: str,
        start_ms: int,
        end_ms: int,
        db_events: List[VisionAnomalyEvent],
    ) -> List[VisionAnomalyEvent]:
        merged: List[VisionAnomalyEvent] = list(db_events)
        seen_keys = {
            (ev.timestamp_ms // 1_000, getattr(ev.anomaly_type, "value", str(ev.anomaly_type)).upper())
            for ev in db_events
        }
        try:
            live_events = get_vision_service().get_events_in_window(session_id, start_ms, end_ms)
            for svc_ev in live_events:
                a_type = _normalize_anomaly_type(svc_ev.anomaly_type)
                if a_type is None:
                    continue
                dedupe_key = (svc_ev.timestamp_ms // 1_000, a_type.value.upper())
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                merged.append(
                    VisionAnomalyEvent(
                        event_id=svc_ev.event_id,
                        session_id=svc_ev.session_id,
                        timestamp_ms=svc_ev.timestamp_ms,
                        anomaly_type=a_type,
                        duration_ms=svc_ev.duration_ms,
                        confidence=svc_ev.confidence,
                    )
                )
        except Exception as exc:
            logger.debug("In-memory vision window merge skipped for %s: %s", session_id, exc)

        merged.sort(key=lambda e: e.timestamp_ms)
        return merged

    @staticmethod
    def _join_answer_chunks(chunks: List[UtteranceEvent]) -> str:
        """Joins chronologically sorted chunks while trimming exact 2-6 word boundary echoes."""
        parts: List[str] = []
        for utt in chunks:
            text = (utt.text or "").strip()
            if not text:
                continue
            if not parts:
                parts.append(text)
                continue
            prev_tokens = parts[-1].split()
            curr_tokens = text.split()
            max_k = min(len(prev_tokens), len(curr_tokens), 6)
            overlap = 0
            for k in range(max_k, 1, -1):
                p_slice = [t.lower().strip(".,!?;:") for t in prev_tokens[-k:]]
                c_slice = [t.lower().strip(".,!?;:") for t in curr_tokens[:k]]
                if p_slice == c_slice:
                    overlap = k
                    break
            parts.append(" ".join(curr_tokens[overlap:]))
        return " ".join(p for p in parts if p).strip()

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
        anomaly_type = _normalize_anomaly_type(row.anomaly_type)
        if anomaly_type is None:
            logger.warning("Skipping unrecognized VisionAnomalyType '%s' on row %s.", row.anomaly_type, row.id)
            return None

        return VisionAnomalyEvent(
            event_id=row.id,
            session_id=row.session_id,
            timestamp_ms=row.timestamp_ms,
            anomaly_type=anomaly_type,
            duration_ms=row.duration_ms,
            confidence=row.confidence,
            pitch_deg=getattr(row, "pitch_deg", 0.0) or 0.0,
            yaw_deg=getattr(row, "yaw_deg", 0.0) or 0.0,
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
    global _active_scoring_worker
    worker = ScoringWorker()
    _active_scoring_worker = worker
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
        try:
            await worker.aclose()
        finally:
            if _active_scoring_worker is worker:
                _active_scoring_worker = None
        logger.info("ScoringWorker '%s' shut down cleanly.", consumer_id)