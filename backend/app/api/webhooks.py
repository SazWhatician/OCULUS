import asyncio
import os
import uuid
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

MAX_DEMO_UPLOAD_BYTES = 150 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post("/webhooks/attendee")
async def attendee_webhook(request: Request):
    signature = (
        request.headers.get("X-Webhook-Signature")
        or request.headers.get("X-Attendee-Signature")
        or request.headers.get("Attendee-Signature")
        or ""
    )
    raw_body = await request.body()

    if not attendee_adapter.verify_webhook_signature(signature, raw_body):
        allow_dev = bool(
            settings.APP_ENV == "development"
            and getattr(settings, "ATTENDEE_WEBHOOK_ALLOW_INSECURE_DEV", False)
            and not getattr(settings, "ATTENDEE_WEBHOOK_SECRET", "")
        )
        if not allow_dev:
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

    # Stream uploaded file safely in 1 MB chunks with a hard size guard
    upload_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(file.filename or "recording.mp4")
    file_ext = os.path.splitext(safe_name)[1].lower() or ".mp4"
    dest_path = os.path.join(upload_dir, f"{session_id}{file_ext}")

    total_bytes = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_DEMO_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded recording exceeds 150 MB limit.")
                await asyncio.to_thread(f.write, chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    finally:
        await file.close()

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
