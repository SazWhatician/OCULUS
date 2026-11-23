import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
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
    rep_res = await db.execute(rep_stmt)
    rep = rep_res.scalars().first()

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
        "report": rep.report_json if rep else None,
    }


@router.patch("/{session_id}/speakers")
async def swap_speakers(session_id: str, req: SwapSpeakerRequest, db: AsyncSession = Depends(get_db)):
    s = await db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    s.speaker_role_map = req.speaker_role_map
    await db.commit()
    return {"session_id": session_id, "speaker_role_map": s.speaker_role_map}


@router.post("/{session_id}/complete")
async def complete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    s.status = SessionStatus.COMPLETED.value
    await db.commit()

    # Re-aggregate report if not present or needs finalization
    rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
    rep_res = await db.execute(rep_stmt)
    rep = rep_res.scalars().first()

    if not rep:
        # Generate final report from stored evaluations
        all_qas_stmt = select(QAEvaluationModel).where(QAEvaluationModel.session_id == session_id)
        all_qas_res = await db.execute(all_qas_stmt)
        evaluations = [QAPairEvaluation.model_validate(r.payload_json) for r in all_qas_res.scalars().all()]

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

        baseline = CandidateBaseline.model_validate(s.baseline_json) if s.baseline_json else None
        final_report = generate_integrity_report(
            session_id=session_id,
            candidate_name=s.candidate_name,
            job_title=s.job_title,
            qa_evaluations=evaluations,
            vision_timeline=all_vision,
            baseline=baseline,
        )
        rep = IntegrityReportModel(
            session_id=session_id,
            overall_risk_score=final_report.overall_risk_score,
            risk_band=final_report.risk_band.value,
            report_json=final_report.model_dump(),
        )
        db.add(rep)
        await db.commit()

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
