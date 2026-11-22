import asyncio
import logging
from typing import Optional, Dict, List
from sqlalchemy import select

from app.schemas.contracts import (
    UtteranceEvent,
    SpeakerRole,
    CandidateBaseline,
    QAPairEvaluation,
    VisionAnomalyEvent,
    VisionAnomalyType,
)
from app.db.redis_bus import redis_bus
from app.db.postgres import (
    async_session_factory,
    SessionModel,
    QAEvaluationModel,
    VisionEventModel,
    IntegrityReportModel,
)
from app.services.heuristics import (
    compute_candidate_baseline,
    compute_deterministic_heuristics,
)
from app.adapters.llm.base import FallbackLLMRouter
from app.services.aggregator import (
    calculate_qa_combined_score,
    generate_integrity_report,
)
from app.telemetry import metrics

logger = logging.getLogger("oculus.worker.scoring")


class ScoringWorker:
    def __init__(self):
        self.llm_router = FallbackLLMRouter()
        # Per-session state: {session_id: {"active_q": "", "q_end_ms": 0, "ans_chunks": [], "qa_index": 0}}
        self._buffers: Dict[str, Dict] = {}

    def _get_buffer(self, session_id: str) -> Dict:
        if session_id not in self._buffers:
            self._buffers[session_id] = {
                "active_q": "",
                "q_end_ms": 0,
                "ans_chunks": [],
                "ans_start_ms": 0,
                "ans_end_ms": 0,
                "qa_index": 0,
            }
        return self._buffers[session_id]

    async def handle_utterance(self, utt: UtteranceEvent) -> None:
        buf = self._get_buffer(utt.session_id)

        if utt.speaker == SpeakerRole.INTERVIEWER:
            # If candidate previously answered, finalize that Q&A pair before opening new question
            if buf["ans_chunks"]:
                await self._evaluate_and_flush_qa(utt.session_id)

            buf["active_q"] = f"{buf['active_q']} {utt.text}".strip()
            buf["q_end_ms"] = utt.end_ms

        elif utt.speaker == SpeakerRole.CANDIDATE:
            if not buf["ans_chunks"]:
                buf["ans_start_ms"] = utt.start_ms
            buf["ans_end_ms"] = utt.end_ms
            buf["ans_chunks"].append(utt)

    async def _evaluate_and_flush_qa(self, session_id: str) -> None:
        buf = self._get_buffer(session_id)
        if not buf["ans_chunks"]:
            return

        full_answer = " ".join(u.text for u in buf["ans_chunks"]).strip()
        ans_start_ms = buf["ans_start_ms"]
        ans_end_ms = buf["ans_end_ms"]
        q_end_ms = buf["q_end_ms"]
        question_text = buf["active_q"] or "General Question"
        qa_index = buf["qa_index"]
        buf["qa_index"] += 1
        buf["ans_chunks"] = []
        buf["active_q"] = ""

        # Fetch session metadata & baseline from DB
        baseline = None
        job_title = "Software Engineer"
        candidate_name = "Candidate"
        async with async_session_factory() as db:
            session_obj = await db.get(SessionModel, session_id)
            if session_obj:
                job_title = session_obj.job_title
                candidate_name = session_obj.candidate_name
                if session_obj.baseline_json:
                    baseline = CandidateBaseline.model_validate(session_obj.baseline_json)

        # Stage 0: Calibrate baseline on Question 0
        if qa_index == 0:
            baseline = compute_candidate_baseline(
                session_id=session_id,
                q0_text=full_answer,
                duration_ms=max(ans_end_ms - ans_start_ms, 1000),
            )
            async with async_session_factory() as db:
                session_obj = await db.get(SessionModel, session_id)
                if session_obj:
                    session_obj.baseline_json = baseline.model_dump()
                    await db.commit()

        # Query concurrent vision events using Pre-Speech Lookback Window [q_end_ms - 500ms, ans_end_ms]
        lookback_start = max(q_end_ms - 500, 0)
        concurrent_vision: List[VisionAnomalyEvent] = []
        async with async_session_factory() as db:
            stmt = (
                select(VisionEventModel)
                .where(VisionEventModel.session_id == session_id)
                .where(VisionEventModel.timestamp_ms >= lookback_start)
                .where(VisionEventModel.timestamp_ms <= ans_end_ms)
            )
            res = await db.execute(stmt)
            for row in res.scalars().all():
                concurrent_vision.append(
                    VisionAnomalyEvent(
                        event_id=row.id,
                        session_id=row.session_id,
                        timestamp_ms=row.timestamp_ms,
                        anomaly_type=VisionAnomalyType(row.anomaly_type),
                        duration_ms=row.duration_ms,
                        confidence=row.confidence,
                        evidence_frame_key=row.evidence_frame_key,
                        vlm_verification_note=row.vlm_verification_note,
                    )
                )

        # Stage 1: Deterministic Heuristics
        ttfs_latency_ms = max(ans_start_ms - q_end_ms, 0) if q_end_ms > 0 else 0
        heuristics = compute_deterministic_heuristics(
            answer_text=full_answer,
            duration_ms=max(ans_end_ms - ans_start_ms, 1000),
            baseline=baseline,
            ttfs_latency_ms=ttfs_latency_ms,
            concurrent_vision_events=concurrent_vision,
        )

        # Stage 2: Selective LLM Judge
        llm_verdict = None
        if heuristics.trigger_llm_judge:
            metrics.inc("llm_judge_invocations")
            llm_verdict = await self.llm_router.evaluate_turn(
                question_text=question_text,
                answer_text=full_answer,
                job_title=job_title,
                baseline=baseline,
                heuristics=heuristics,
            )
        else:
            metrics.inc("prefilter_skips")

        # Per-turn combined score
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

        # Persist QAEvaluation & update overall report in Postgres
        async with async_session_factory() as db:
            qa_model = QAEvaluationModel(
                id=f"qa_{session_id}_{qa_index}",
                session_id=session_id,
                qa_index=qa_index,
                combined_risk_score=combined_score,
                payload_json=qa_eval.model_dump(),
            )
            db.add(qa_model)

            # Query all QAs to re-aggregate report
            all_qas_stmt = select(QAEvaluationModel).where(QAEvaluationModel.session_id == session_id)
            all_qas_res = await db.execute(all_qas_stmt)
            evaluations = [QAPairEvaluation.model_validate(r.payload_json) for r in all_qas_res.scalars().all()]
            evaluations.append(qa_eval)

            # Query all vision events
            all_v_stmt = select(VisionEventModel).where(VisionEventModel.session_id == session_id)
            all_v_res = await db.execute(all_v_stmt)
            all_vision = [
                VisionAnomalyEvent(
                    event_id=r.id,
                    session_id=r.session_id,
                    timestamp_ms=r.timestamp_ms,
                    anomaly_type=VisionAnomalyType(r.anomaly_type),
                    duration_ms=r.duration_ms,
                    confidence=r.confidence,
                    evidence_frame_key=r.evidence_frame_key,
                    vlm_verification_note=r.vlm_verification_note,
                )
                for r in all_v_res.scalars().all()
            ]

            report = generate_integrity_report(
                session_id=session_id,
                candidate_name=candidate_name,
                job_title=job_title,
                qa_evaluations=evaluations,
                vision_timeline=all_vision,
                baseline=baseline,
            )

            rep_model = await db.get(IntegrityReportModel, session_id)
            if not rep_model:
                rep_model = IntegrityReportModel(
                    session_id=session_id,
                    overall_risk_score=report.overall_risk_score,
                    risk_band=report.risk_band.value,
                    report_json=report.model_dump(),
                )
                db.add(rep_model)
            else:
                rep_model.overall_risk_score = report.overall_risk_score
                rep_model.risk_band = report.risk_band.value
                rep_model.report_json = report.model_dump()

            await db.commit()

        # Broadcast live event to WebSocket
        await redis_bus.broadcast_ui(
            session_id=session_id,
            event_type="QA_EVALUATED",
            data={
                "qa_evaluation": qa_eval.model_dump(),
                "overall_risk_score": report.overall_risk_score,
                "risk_band": report.risk_band.value,
            },
        )


async def run_scoring_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    worker = ScoringWorker()
    logger.info("ScoringWorker started consuming from 'stream:utterances'")
    await redis_bus.consume_stream(
        stream="stream:utterances",
        group="scoring_workers",
        consumer="scoring_1",
        model_cls=UtteranceEvent,
        handler=worker.handle_utterance,
        stop_event=stop_event,
    )
