import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import (
    get_db,
    SessionModel,
    UtteranceModel,
    VisionEventModel,
    QAEvaluationModel,
    IntegrityReportModel,
)
from app.schemas.contracts import (
    IngestionMode,
    SessionStatus,
    IntegrityReport,
    RiskBand,
    SpeakerRole,
    UtteranceEvent,
    WordTiming,
)
from app.services.vocab_extractor import extract_custom_vocab
from app.adapters.ingestion.attendee import AttendeeBotAdapter
from app.db.storage import get_storage_adapter
from app.db.redis_bus import redis_bus
from app.services.aggregator import generate_integrity_report
from app.schemas.contracts import CandidateBaseline, QAPairEvaluation, VisionAnomalyEvent, VisionAnomalyType

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
attendee_adapter = AttendeeBotAdapter()
storage = get_storage_adapter()


class CreateSessionRequest(BaseModel):
    candidate_name: str
    interviewer_name: Optional[str] = None
    job_title: str
    job_description: Optional[str] = ""
    ingestion_mode: IngestionMode = IngestionMode.DEMO_FILE
    meeting_url: Optional[str] = None
    bot_display_name: str = "OCULUS Sentinel"


class SwapSpeakerRequest(BaseModel):
    speaker_role_map: Dict[str, str]  # e.g. {"S1": "candidate", "S2": "interviewer"}


@router.post("")
async def create_session(req: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Extract technical domain vocabulary
    custom_vocab = extract_custom_vocab(req.job_title, req.job_description or "")

    initial_status = SessionStatus.CREATED
    bot_external_id = None

    # If meeting bot requested, dispatch Attendee bot
    if req.ingestion_mode == IngestionMode.ATTENDEE_BOT and req.meeting_url:
        dispatch_res = await attendee_adapter.start_session(
            session_id=session_id,
            meeting_url=req.meeting_url,
            bot_name=req.bot_display_name,
        )
        bot_external_id = dispatch_res.get("bot_id")
        initial_status = SessionStatus.WAITING_IN_LOBBY

    db_session = SessionModel(
        id=session_id,
        candidate_name=req.candidate_name,
        interviewer_name=req.interviewer_name,
        job_title=req.job_title,
        job_description=req.job_description,
        custom_vocab=custom_vocab,
        ingestion_mode=req.ingestion_mode.value,
        status=initial_status.value,
        bot_external_id=bot_external_id,
        speaker_role_map={"S1": "interviewer", "S2": "candidate"},
    )
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    return {
        "session_id": session_id,
        "status": db_session.status,
        "custom_vocab": custom_vocab,
        "ingestion_mode": db_session.ingestion_mode,
        "bot_external_id": bot_external_id,
        "ws_url": f"/ws/sessions/{session_id}",
    }


@router.get("")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    stmt = select(SessionModel).order_by(SessionModel.created_at.desc())
    res = await db.execute(stmt)
    sessions = res.scalars().all()
    results = []
    for s in sessions:
        rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == s.id)
        rep_res = await db.execute(rep_stmt)
        rep = rep_res.scalars().first()
        results.append({
            "id": s.id,
            "candidate_name": s.candidate_name,
            "interviewer_name": s.interviewer_name,
            "job_title": s.job_title,
            "status": s.status,
            "ingestion_mode": s.ingestion_mode,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "overall_risk_score": rep.overall_risk_score if rep else None,
            "risk_band": rep.risk_band if rep else None,
        })
    return {"sessions": results}


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
    rep = (await db.execute(rep_stmt)).scalars().first()

    utt_stmt = (
        select(UtteranceModel)
        .where(UtteranceModel.session_id == session_id)
        .order_by(UtteranceModel.start_ms.asc())
    )
    utt_rows = (await db.execute(utt_stmt)).scalars().all()
    utterances = [
        {
            "utterance_id": u.id,
            "session_id": u.session_id,
            "speaker": u.speaker,
            "speaker_label": u.speaker_label,
            "text": u.text,
            "start_ms": u.start_ms,
            "end_ms": u.end_ms,
            "words": u.words_json or [],
        }
        for u in utt_rows
    ]

    qa_stmt = (
        select(QAEvaluationModel)
        .where(QAEvaluationModel.session_id == session_id)
        .order_by(QAEvaluationModel.qa_index.asc())
    )
    qa_rows = (await db.execute(qa_stmt)).scalars().all()
    qa_evaluations = [r.payload_json for r in qa_rows if isinstance(r.payload_json, dict)]

    vis_stmt = (
        select(VisionEventModel)
        .where(VisionEventModel.session_id == session_id)
        .order_by(VisionEventModel.timestamp_ms.asc())
    )
    vis_rows = (await db.execute(vis_stmt)).scalars().all()
    vision_events = []
    for r in vis_rows:
        item = {
            "event_id": r.id,
            "session_id": r.session_id,
            "timestamp_ms": r.timestamp_ms,
            "anomaly_type": r.anomaly_type,
            "duration_ms": r.duration_ms,
            "confidence": r.confidence,
            "evidence_frame_key": r.evidence_frame_key,
            "vlm_verification_note": r.vlm_verification_note,
        }
        if r.evidence_frame_key:
            item["evidence_url"] = await storage.get_presigned_url(r.evidence_frame_key)
        vision_events.append(item)

    return {
        "id": s.id,
        "candidate_name": s.candidate_name,
        "interviewer_name": s.interviewer_name,
        "job_title": s.job_title,
        "job_description": s.job_description,
        "status": s.status,
        "ingestion_mode": s.ingestion_mode,
        "custom_vocab": s.custom_vocab,
        "speaker_role_map": s.speaker_role_map,
        "baseline": s.baseline_json,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "utterances": utterances,
        "qa_evaluations": qa_evaluations,
        "vision_events": vision_events,
        "report": rep.report_json if rep else None,
    }


def _safe_speaker_role_str(raw: str) -> str:
    try:
        return SpeakerRole(str(raw).strip().lower()).value
    except ValueError:
        return SpeakerRole.UNKNOWN.value


async def _rebuild_integrity_report(
    session_id: str,
    s: SessionModel,
    db: AsyncSession,
) -> IntegrityReportModel:
    all_qas_stmt = (
        select(QAEvaluationModel)
        .where(QAEvaluationModel.session_id == session_id)
        .order_by(QAEvaluationModel.qa_index.asc())
    )
    all_qas_res = await db.execute(all_qas_stmt)
    evaluations = [QAPairEvaluation.model_validate(r.payload_json) for r in all_qas_res.scalars().all()]

    all_v_stmt = (
        select(VisionEventModel)
        .where(VisionEventModel.session_id == session_id)
        .order_by(VisionEventModel.timestamp_ms.asc())
    )
    all_v_res = await db.execute(all_v_stmt)
    all_vision: List[VisionAnomalyEvent] = []
    for r in all_v_res.scalars().all():
        try:
            atype = VisionAnomalyType(r.anomaly_type)
        except ValueError:
            continue
        all_vision.append(
            VisionAnomalyEvent(
                event_id=r.id,
                session_id=r.session_id,
                timestamp_ms=r.timestamp_ms,
                anomaly_type=atype,
                duration_ms=r.duration_ms,
                confidence=r.confidence,
                evidence_frame_key=r.evidence_frame_key,
                vlm_verification_note=r.vlm_verification_note,
            )
        )

    baseline = CandidateBaseline.model_validate(s.baseline_json) if s.baseline_json else None
    final_report = generate_integrity_report(
        session_id=session_id,
        candidate_name=s.candidate_name,
        job_title=s.job_title,
        qa_evaluations=evaluations,
        vision_timeline=all_vision,
        baseline=baseline,
    )
    payload = final_report.model_dump()

    rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
    rep = (await db.execute(rep_stmt)).scalars().first()
    if rep is not None:
        rep.overall_risk_score = final_report.overall_risk_score
        rep.risk_band = final_report.risk_band.value
        rep.report_json = payload
    else:
        rep = IntegrityReportModel(
            session_id=session_id,
            overall_risk_score=final_report.overall_risk_score,
            risk_band=final_report.risk_band.value,
            report_json=payload,
        )
        db.add(rep)
    await db.commit()
    return rep


@router.patch("/{session_id}/speakers")
async def swap_speakers(session_id: str, req: SwapSpeakerRequest, db: AsyncSession = Depends(get_db)):
    from app.workers.stt_worker import get_active_stt_worker
    from app.workers.scoring_worker import ScoringWorker, get_active_scoring_worker

    s = await db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    normalized_map = {str(k): _safe_speaker_role_str(v) for k, v in req.speaker_role_map.items()}
    s.speaker_role_map = normalized_map

    # Update any open buffered utterance in the active stitcher
    stt_w = get_active_stt_worker()
    if stt_w is not None:
        open_utt = stt_w.stitcher.get_open_utterance(session_id)
        if open_utt is not None and open_utt.speaker_label in normalized_map:
            open_utt.speaker = SpeakerRole(normalized_map[open_utt.speaker_label])

    # Retroactively remap persisted utterances in DB
    utt_stmt = (
        select(UtteranceModel)
        .where(UtteranceModel.session_id == session_id)
        .order_by(UtteranceModel.start_ms.asc())
    )
    utt_rows = (await db.execute(utt_stmt)).scalars().all()
    roles_changed = False
    for row in utt_rows:
        mapped = normalized_map.get(row.speaker_label)
        if mapped and row.speaker != mapped:
            row.speaker = mapped
            roles_changed = True

    if roles_changed and utt_rows:
        await db.execute(delete(QAEvaluationModel).where(QAEvaluationModel.session_id == session_id))
        await db.execute(delete(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id))
        s.baseline_json = None
        await db.commit()

        replay_events = [
            UtteranceEvent(
                session_id=row.session_id,
                utterance_id=row.id,
                speaker=SpeakerRole(_safe_speaker_role_str(row.speaker)),
                speaker_label=row.speaker_label,
                text=row.text,
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                words=[WordTiming.model_validate(w) for w in (row.words_json or [])],
            )
            for row in utt_rows
        ]
        scoring_w = get_active_scoring_worker()
        ephemeral = scoring_w is None
        worker = scoring_w or ScoringWorker()
        try:
            worker.reset_session(session_id)
            for ev in replay_events:
                await worker.handle_utterance(ev)
            await worker.flush_session(session_id)
        finally:
            if ephemeral:
                await worker.llm_router.aclose()
    else:
        await db.commit()

    await redis_bus.broadcast_ui(
        session_id=session_id,
        event_type="SPEAKER_MAP_UPDATED",
        data={"speaker_role_map": normalized_map, "roles_changed": roles_changed},
    )
    return {"session_id": session_id, "speaker_role_map": s.speaker_role_map, "roles_changed": roles_changed}


@router.post("/{session_id}/complete")
async def complete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    from app.workers.stt_worker import get_active_stt_worker
    from app.workers.scoring_worker import get_active_scoring_worker

    s = await db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    if s.ingestion_mode == IngestionMode.ATTENDEE_BOT.value and s.bot_external_id:
        await attendee_adapter.stop_session(session_id, bot_id=s.bot_external_id)

    # Flush any unclosed speaker turn and candidate answer before generating final report
    stt_w = get_active_stt_worker()
    flushed_utt = None
    if stt_w is not None:
        flushed_utt = await stt_w.flush_session(session_id)

    scoring_w = get_active_scoring_worker()
    if scoring_w is not None:
        if flushed_utt is not None:
            await scoring_w.handle_utterance(flushed_utt)
        await scoring_w.flush_session(session_id)

    # Refresh SessionModel in case flush calibrated Q0 baseline
    await db.refresh(s)
    s.status = SessionStatus.COMPLETED.value
    await db.commit()

    rep = await _rebuild_integrity_report(session_id, s, db)

    await redis_bus.broadcast_ui(
        session_id=session_id,
        event_type="STATUS_UPDATE",
        data={"status": SessionStatus.COMPLETED.value, "overall_risk_score": rep.overall_risk_score, "risk_band": rep.risk_band},
    )

    return {
        "session_id": session_id,
        "status": s.status,
        "overall_risk_score": rep.overall_risk_score,
        "risk_band": rep.risk_band,
        "report": rep.report_json,
    }


@router.get("/{session_id}/report")
async def get_session_report(session_id: str, db: AsyncSession = Depends(get_db)):
    rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
    rep_res = await db.execute(rep_stmt)
    rep = rep_res.scalars().first()
    if not rep:
        raise HTTPException(status_code=404, detail="Integrity report not yet generated for this session")

    report_data = dict(rep.report_json)
    # Refresh presigned URLs for any evidence keyframes
    for anomaly in report_data.get("vision_timeline", []):
        key = anomaly.get("evidence_frame_key")
        if key:
            anomaly["evidence_url"] = await storage.get_presigned_url(key)

    return report_data
