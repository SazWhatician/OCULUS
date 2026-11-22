import os
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.postgres import get_db, SessionModel
from app.adapters.ingestion.attendee import AttendeeBotAdapter
from app.adapters.ingestion.demo_replay import DemoFileAdapter
from app.services.vocab_extractor import extract_custom_vocab

router = APIRouter(prefix="/api/v1", tags=["webhooks_and_demo"])
attendee_adapter = AttendeeBotAdapter()
demo_adapter = DemoFileAdapter()


@router.post("/webhooks/attendee")
async def attendee_webhook(request: Request):
    signature = request.headers.get("X-Attendee-Signature", "")
    raw_body = await request.body()

    if not attendee_adapter.verify_webhook_signature(signature, raw_body):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    await attendee_adapter.handle_webhook_event(payload)
    return {"status": "received"}


@router.post("/demo/upload")
async def upload_demo_file(
    file: UploadFile = File(...),
    candidate_name: str = Form("Demo Candidate"),
    job_title: str = Form("Software Engineer"),
    job_description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    session_id = f"sess_demo_{uuid.uuid4().hex[:8]}"

    # Save uploaded file locally
    upload_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_ext = os.path.splitext(file.filename)[1] or ".mp4"
    dest_path = os.path.join(upload_dir, f"{session_id}{file_ext}")

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    custom_vocab = extract_custom_vocab(job_title, job_description)

    # Create session record
    db_session = SessionModel(
        id=session_id,
        candidate_name=candidate_name,
        interviewer_name="Interviewer",
        job_title=job_title,
        job_description=job_description,
        custom_vocab=custom_vocab,
        ingestion_mode="demo_file",
        status="PROCESSING",
        speaker_role_map={"S1": "interviewer", "S2": "candidate"},
    )
    db.add(db_session)
    await db.commit()

    # Trigger demo replay
    await demo_adapter.start_session(session_id=session_id, file_path=dest_path)

    return {
        "session_id": session_id,
        "status": "PROCESSING",
        "custom_vocab": custom_vocab,
        "ws_url": f"/ws/sessions/{session_id}",
    }
