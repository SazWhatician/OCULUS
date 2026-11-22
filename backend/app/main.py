import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select

from app.config import settings
from app.db.postgres import init_db, async_session_factory, SessionModel, IntegrityReportModel
from app.db.redis_bus import redis_bus
from app.telemetry import metrics, log_structured
from app.api import sessions, webhooks, ws
from app.workers.stt_worker import run_stt_worker
from app.workers.vision_worker import run_vision_worker
from app.workers.scoring_worker import run_scoring_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("oculus.main")

worker_tasks = []
stop_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info(f"Starting OCULUS Sentinel Backend (Mode: {settings.DEPLOY_MODE})...")
    await init_db()
    await redis_bus.connect()

    # In single_node mode, supervise all 3 worker pools inside the single container process
    if settings.DEPLOY_MODE == "single_node":
        logger.info("Spawning in-process background worker mesh (STT, Vision, Scoring)...")
        worker_tasks.append(asyncio.create_task(run_stt_worker(stop_event)))
        worker_tasks.append(asyncio.create_task(run_vision_worker(stop_event)))
        worker_tasks.append(asyncio.create_task(run_scoring_worker(stop_event)))

    yield

    # Shutdown sequence
    logger.info("Shutting down OCULUS Sentinel Backend...")
    stop_event.set()
    for t in worker_tasks:
        t.cancel()
    await redis_bus.close()


app = FastAPI(
    title="OCULUS Sentinel API",
    description="Event-Driven Multimodal Interview Integrity Sentinel with Zero-Cost Resilience",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(sessions.router)
app.include_router(webhooks.router)
app.include_router(ws.router)


@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "service": "OCULUS Sentinel",
        "deploy_mode": settings.DEPLOY_MODE,
        "stt_provider": settings.STT_PROVIDER,
        "llm_primary_provider": settings.LLM_PRIMARY_PROVIDER,
        "storage_provider": settings.STORAGE_PROVIDER,
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.get("/api/v1/media/{key:path}")
async def serve_media(key: str):
    """Serves locally stored keyframes and audio chunks."""
    safe_key = key.replace("\\", "/").strip("/").replace("..", "_")
    full_path = os.path.join(settings.LOCAL_STORAGE_DIR, safe_key)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(full_path)


# Backward Compatibility Endpoints for existing frontend routes
@app.get("/api/reports")
async def compat_list_reports():
    """Maps to /api/v1/sessions for existing frontend table."""
    async with async_session_factory() as db:
        stmt = select(SessionModel).order_by(SessionModel.created_at.desc())
        sessions_res = (await db.execute(stmt)).scalars().all()
        results = []
        for s in sessions_res:
            rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == s.id)
            rep = (await db.execute(rep_stmt)).scalars().first()
            results.append({
                "id": s.id,
                "candidate": s.candidate_name,
                "role": s.job_title,
                "interviewer": s.interviewer_name or "Interviewer",
                "platform": s.ingestion_mode,
                "date": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "",
                "duration": "Completed" if s.status == "COMPLETED" else "Live",
                "language": "en-IN",
                "status": s.status,
                "statusLabel": s.status.replace("_", " ").title(),
                "statusBadge": "apple-badge bg-emerald-500/10 text-emerald-600 border border-emerald-500/20",
                "coverage": "100%",
                "observationsCount": len(rep.report_json.get("qa_evaluations", [])) if rep else 0,
                "reviewRecommended": rep.risk_band == "ELEVATED" if rep else False,
            })
        return results


@app.get("/api/reports/{session_id}")
async def compat_get_report(session_id: str):
    """Maps to /api/v1/sessions/{id}/report for existing frontend report viewer."""
    async with async_session_factory() as db:
        rep_stmt = select(IntegrityReportModel).where(IntegrityReportModel.session_id == session_id)
        rep = (await db.execute(rep_stmt)).scalars().first()
        if rep:
            return rep.report_json

    # Fallback to demo report if session_zoom_8902 requested
    if "session_zoom_8902" in session_id:
        demo_path = os.path.join(os.path.dirname(__file__), "..", "data", "reports", "session_zoom_8902.json")
        if os.path.exists(demo_path):
            import json
            with open(demo_path, "r") as f:
                return json.load(f).get("report", {})

    raise HTTPException(status_code=404, detail="Report not found")
