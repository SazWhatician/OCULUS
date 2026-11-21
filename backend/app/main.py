import os
import re
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from typing import Dict, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.speech.sarvam import (
    SarvamSTTService,
    SarvamTTSService,
    SarvamSpeechAgent,
    SUPPORTED_LANGUAGES,
    SUPPORTED_VOICES
)
from app.speech.enhancer import TranscriptEnhancer, TECHNICAL_TERMS_MAP
from app.services.proctoring_service import get_proctoring_service
from app.adapters.zoom import ZoomAdapter
from app.adapters.teams import TeamsAdapter
from app.adapters.google_meet import GoogleMeetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verimeet-backend")

app = FastAPI(
    title="VeriMeet AI Sentinel Backend",
    description="Proctoring & Speech Analysis Agent Backend with Sarvam AI Saaras v3 & Transcript Enhancement (English & Hindi)",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_service = SarvamSTTService()
tts_service = SarvamTTSService()
speech_agent = SarvamSpeechAgent()
enhancer = TranscriptEnhancer()

zoom_adapter = ZoomAdapter()
teams_adapter = TeamsAdapter()
gmeet_adapter = GoogleMeetAdapter()


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into speech")
    target_language_code: str = Field("hi-IN", description="Language code: 'hi-IN' (Hindi) or 'en-IN' (English)")
    speaker_voice: str = Field("meera", description="Voice identifier (e.g. meera, pavitra, mahesh)")
    pace: float = Field(1.0, description="Speech rate multiplier")
    model: str = Field("saaras:v2", description="Sarvam AI speech model")


class EnhanceTextRequest(BaseModel):
    text: str
    clean_fillers: bool = True


class DispatchBotRequest(BaseModel):
    meeting_url: str
    candidate_name: str
    interviewer_name: str
    role_title: Optional[str] = "Software Engineer"
    bot_display_name: str = "VeriMeet Proctor Bot"
    platform: str = "zoom"
    language_code: str = "en-IN"  # Default English (India)
    voice_name: str = "meera"


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "VeriMeet AI Sentinel Backend",
        "sarvam_ai_stt_model": stt_service.model,
        "supported_languages": ["en-IN", "hi-IN"],
        "transcription_enhancer": "active",
        "version": "1.1.0"
    }


@app.get("/api/speech/config")
def get_speech_config():
    return {
        "status": "active",
        "engine": f"Sarvam AI STT ({stt_service.model}) / TTS (saaras:v2)",
        "stt_model": stt_service.model,
        "tts_model": "saaras:v2",
        "api_key_configured": bool(os.getenv("SARVAM_API_KEY")),
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_voices": SUPPORTED_VOICES,
        "domain_terms_count": len(TECHNICAL_TERMS_MAP)
    }


@app.post("/api/speech/stt")
async def process_stt(
    file: UploadFile = File(...),
    language_code: str = Form("en-IN"),
    model: Optional[str] = Form(None),
    clean_fillers: bool = Form(True)
):
    try:
        content = await file.read()
        result = await stt_service.transcribe_audio_bytes(
            audio_bytes=content,
            language_code=language_code,
            model=model or stt_service.model,
            clean_fillers=clean_fillers
        )
        return result
    except Exception as e:
        logger.error(f"STT Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/speech/enhance")
async def process_text_enhancement(req: EnhanceTextRequest):
    """Enhances raw speech text with entity normalization, disfluency removal, and confidence scoring."""
    enhanced = enhancer.enhance_text(req.text, clean_fillers=req.clean_fillers)
    conf_info = enhancer.annotate_confidence(enhanced)
    return {
        "original_text": req.text,
        "enhanced_text": enhanced,
        "clean_fillers": req.clean_fillers,
        "confidence_info": conf_info
    }


from app.services.vision_proctoring import get_vision_service, VisionTelemetryFrame

@app.post("/api/proctoring/vision")
async def ingest_vision_telemetry(frame: VisionTelemetryFrame):
    """Ingests real-time video, gaze tracking, head pose, and multi-face telemetry frames."""
    try:
        vision_svc = get_vision_service()
        anomaly_event = vision_svc.ingest_frame(frame)
        summary = vision_svc.get_session_summary(frame.session_id)
        return {
            "status": "received",
            "session_id": frame.session_id,
            "anomaly_detected": bool(anomaly_event),
            "anomaly_event": anomaly_event.model_dump() if anomaly_event else None,
            "vision_summary": summary
        }
    except Exception as e:
        logger.error(f"Vision telemetry error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/dispatch")
async def dispatch_bot(req: DispatchBotRequest):
    """Dispatches a bot to join a Zoom, Microsoft Teams, or Google Meet call via Attendee API."""
    try:
        meeting_url = req.meeting_url.strip()
        if not meeting_url:
            raise HTTPException(status_code=400, detail="Meeting URL is required")

        url_lower = meeting_url.lower()
        if "meet.google.com" in url_lower or "google.com" in url_lower:
            adapter = gmeet_adapter
            platform_name = "google_meet"
        elif "teams.microsoft.com" in url_lower or "teams.live.com" in url_lower:
            adapter = teams_adapter
            platform_name = "teams"
        else:
            adapter = zoom_adapter
            platform_name = "zoom"

        session_id = f"sess_{int(datetime.utcnow().timestamp())}"
        dispatch_res = await adapter.join_meeting(
            meeting_url=meeting_url,
            bot_name=req.bot_display_name,
            session_id=session_id
        )

        bot_id = dispatch_res.get("bot_id") or session_id

        session_data = {
            "id": bot_id,
            "session_id": session_id,
            "candidate": req.candidate_name,
            "interviewer": req.interviewer_name,
            "role": req.role_title,
            "platform": platform_name,
            "meeting_url": meeting_url,
            "language": req.language_code,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": "00:00",
            "status": "JOINING",
            "statusLabel": "Bot Joining Call...",
            "statusBadge": "apple-badge bg-amber-500/10 text-amber-600 border border-amber-500/20",
            "coverage": "0%",
            "observationsCount": 0,
            "reviewRecommended": False,
            "dispatch": dispatch_res
        }

        # Persist session JSON file
        session_file = os.path.join(SESSIONS_DIR, f"{bot_id}.json")
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        return {
            "status": "dispatched",
            "bot_id": bot_id,
            "session_id": bot_id,
            "dispatch": dispatch_res,
            "session": session_data
        }
    except Exception as e:
        logger.error(f"Dispatch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/bot/{bot_id}/transcript")
async def get_bot_live_transcript(bot_id: str):
    """Fetches live raw transcript lines from Attendee API for a bot."""
    try:
        from app.adapters.attendee_client import get_transcript
        transcript = await get_transcript(bot_id)
        return {"bot_id": bot_id, "transcript": transcript}
    except Exception as e:
        logger.error(f"Error fetching live transcript for bot {bot_id}: {e}")
        return {"bot_id": bot_id, "transcript": []}


@app.get("/api/reports")
async def list_reports():
    """Lists all sessions and their proctoring status for the reports list page."""
    try:
        sessions = []
        if os.path.exists(SESSIONS_DIR):
            for fname in os.listdir(SESSIONS_DIR):
                if fname.endswith(".json"):
                    fpath = os.path.join(SESSIONS_DIR, fname)
                    with open(fpath, "r") as f:
                        sdata = json.load(f)
                        sessions.append(sdata)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        return {"sessions": []}


def normalize_transcript_text(text: str, clean_fillers: bool = True) -> str:
    if not text:
        return text
    return enhancer.enhance_text(text, clean_fillers=clean_fillers)


def parse_attendee_transcript(raw_transcript: list, candidate_name: str = "", interviewer_name: str = "") -> list:
    """
    Parses raw Attendee transcript blocks into clean Question-Answer pairs with timestamps.
    Applies temporal utterance stitching and Sarvam domain entity normalization.
    """
    if not raw_transcript or not isinstance(raw_transcript, list):
        return []

    # First apply utterance stitching to merge rapid sentence chunks
    stitched_blocks = enhancer.stitch_utterances(raw_transcript, max_gap_ms=2500, clean_fillers=True)

    cand_parts = [p.lower() for p in candidate_name.strip().split() if p]
    int_parts = [p.lower() for p in interviewer_name.strip().split() if p]

    def is_candidate_speaker(speaker: str) -> bool:
        sp_lower = speaker.lower()
        if cand_parts and any(p in sp_lower for p in cand_parts):
            return True
        if int_parts and any(p in sp_lower for p in int_parts):
            return False
        return "candidate" in sp_lower or "applicant" in sp_lower

    qa_pairs = []
    current_q_text = []
    current_q_speaker = ""
    current_a_text = []
    current_a_speaker = ""
    start_ts_ms = None

    for block in stitched_blocks:
        speaker = block.get("speaker", "Participant").strip()
        text = block.get("text", "").strip()
        ts_ms = block.get("timestamp_ms", 0)

        if not text:
            continue

        if is_candidate_speaker(speaker):
            if not current_a_speaker:
                current_a_speaker = speaker
            if start_ts_ms is None:
                start_ts_ms = ts_ms
            current_a_text.append(text)
        else:
            if current_a_text:
                qa_pairs.append({
                    "category": "technical",
                    "question": " ".join(current_q_text) if current_q_text else f"Question from {current_q_speaker or 'Interviewer'}",
                    "answer": " ".join(current_a_text),
                    "timestamp_ms": start_ts_ms or 0,
                    "interviewer_speaker": current_q_speaker or speaker,
                    "candidate_speaker": current_a_speaker or "Candidate"
                })
                current_q_text = [text]
                current_q_speaker = speaker
                current_a_text = []
                current_a_speaker = ""
                start_ts_ms = None
            else:
                current_q_text.append(text)
                if not current_q_speaker:
                    current_q_speaker = speaker

    if current_a_text or current_q_text:
        qa_pairs.append({
            "category": "technical",
            "question": " ".join(current_q_text) if current_q_text else "General Question",
            "answer": " ".join(current_a_text) if current_a_text else "No response recorded.",
            "timestamp_ms": start_ts_ms or 0,
            "interviewer_speaker": current_q_speaker or "Interviewer",
            "candidate_speaker": current_a_speaker or "Candidate"
        })

    return qa_pairs

DEMO_MOCK_TRANSCRIPT = {
    "qa_pairs": [
        {
            "category": "introduction",
            "question": "Can you tell me a bit about yourself and your background?",
            "answer": "Yeah, sure! So, um, I have been working as a full-stack software engineer for about four years now. I started with frontend React development at VSSUT and then moved into building microservice backends with Python FastAPI and PostgreSQL."
        },
        {
            "category": "technical",
            "question": "How would you optimize a slow database query in production?",
            "answer": "It is important to note that optimizing a database query requires a multifaceted approach. Firstly, one must delve into the execution plan using the EXPLAIN command to analyze slow index scans. Secondly, there are several indexing strategies such as composite indexing. Furthermore, in today's fast-paced world, caching layers like Redis play a pivotal role. To summarize, a comprehensive understanding of database architecture is key."
        },
        {
            "category": "technical",
            "question": "Could you describe a specific experience where you scaled a distributed queue system?",
            "answer": "Certainly! In my previous role, I configured Kafka message streaming to handle peak user loads. As we can see, utilizing asynchronous background workers prevents bottlenecking and ensures system resilience. Additionally, it is worth noting that monitoring queue latency with Prometheus is essential."
        }
    ]
}

@app.get("/api/reports/{session_id}")
async def get_report(session_id: str):
    """
    Generate or fetch a proctoring report for a given session/bot.
    Fetches real Attendee transcript if available; falls back to DEMO_MOCK_TRANSCRIPT.
    """
    try:
        report_file = os.path.join(REPORTS_DIR, f"{session_id}.json")
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                data = json.load(f)
            report_data = data.get("report", {})
            report_data["transcript"] = data.get("transcript", {})
            return report_data
            
        from app.adapters.attendee_client import get_transcript
        raw_transcript = await get_transcript(session_id)
        
        cand_name = ""
        int_name = ""
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                sdata = json.load(f)
                cand_name = sdata.get("candidate", "")
                int_name = sdata.get("interviewer", "")

        qa_pairs = parse_attendee_transcript(raw_transcript, candidate_name=cand_name, interviewer_name=int_name)
        if not qa_pairs:
            qa_pairs = DEMO_MOCK_TRANSCRIPT["qa_pairs"]
        
        transcript_data = {"qa_pairs": qa_pairs}

        proctoring_service = get_proctoring_service()
        report = await proctoring_service.score_session(
            transcript=transcript_data,
            job_title="Software Engineer"
        )
        report_dict = report.model_dump(mode="json")
        
        # Save generated report to disk for persistence
        with open(report_file, "w") as f:
            json.dump({"report": report_dict, "transcript": transcript_data}, f)
            
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_data = json.load(f)
            session_data["status"] = "REPORT_READY"
            session_data["statusLabel"] = "Report Ready"
            session_data["statusBadge"] = "apple-badge bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
            session_data["duration"] = "Completed"
            session_data["coverage"] = "100%"
            session_data["observationsCount"] = len(report_dict.get("per_answer", []))
            session_data["reviewRecommended"] = report_dict.get("risk_band") == "high"
            if session_data["reviewRecommended"]:
                session_data["status"] = "NEEDS_REVIEW"
                session_data["statusLabel"] = "Needs Review"
                session_data["statusBadge"] = "apple-badge bg-rose-500/10 text-rose-600 border border-rose-500/20"
            with open(session_file, "w") as f:
                json.dump(session_data, f)
                
        report_dict["transcript"] = transcript_data
        return report_dict
    except Exception as e:
        logger.error(f"Error generating proctoring report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/webhooks/attendee")
async def attendee_webhook(payload: Dict[str, Any]):
    bot_id = payload.get("id") or payload.get("bot_id")
    state = payload.get("state")
    transcription_state = payload.get("transcription_state")
    
    logger.info(f"Received Attendee webhook event for bot {bot_id}: state={state}, transcript={transcription_state}")
    
    if state == "ended" and transcription_state == "complete":
        from app.adapters.attendee_client import get_transcript
        raw_transcript = await get_transcript(bot_id)
        
        cand_name = ""
        int_name = ""
        session_file = os.path.join(SESSIONS_DIR, f"{bot_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                sdata = json.load(f)
                cand_name = sdata.get("candidate", "")
                int_name = sdata.get("interviewer", "")

        qa_pairs = parse_attendee_transcript(raw_transcript, candidate_name=cand_name, interviewer_name=int_name)
            
        if qa_pairs:
            proctoring_service = get_proctoring_service()
            report = await proctoring_service.score_session(
                transcript={"qa_pairs": qa_pairs},
                job_title="Software Engineer"
            )
            report_dict = report.model_dump(mode="json")
            with open(os.path.join(REPORTS_DIR, f"{bot_id}.json"), "w") as f:
                json.dump({"report": report_dict, "transcript": {"qa_pairs": qa_pairs}}, f)
                
            if os.path.exists(session_file):
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                session_data["status"] = "REPORT_READY"
                session_data["statusLabel"] = "Report Ready"
                session_data["statusBadge"] = "apple-badge bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                session_data["duration"] = "Completed"
                session_data["coverage"] = "100%"
                session_data["observationsCount"] = len(report_dict.get("per_answer", []))
                session_data["reviewRecommended"] = report_dict.get("risk_band") == "high"
                if session_data["reviewRecommended"]:
                    session_data["status"] = "NEEDS_REVIEW"
                    session_data["statusLabel"] = "Needs Review"
                    session_data["statusBadge"] = "apple-badge bg-rose-500/10 text-rose-600 border border-rose-500/20"
                with open(session_file, "w") as f:
                    json.dump(session_data, f)
                    
            logger.info(f"Generated proctoring report for bot {bot_id}: Risk={report.risk_band}")
        else:
            logger.warning(f"No transcript data found for bot {bot_id}")
        
    return {"status": "received"}

