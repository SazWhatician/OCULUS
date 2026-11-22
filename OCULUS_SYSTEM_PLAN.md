# OCULUS: Architecture, Workflow, Tech Stack & Engineering Risks

**Working Title:** OCULUS (AI Interview Integrity Sentinel & Multimodal Proctoring Platform)  
**Repository:** [github.com/SazWhatician/OCULUS](https://github.com/SazWhatician/OCULUS)  
**Status:** Architecture Specification & Development Roadmap

---

## 1. Executive Summary & Objective

**OCULUS** is an autonomous interview integrity surveillance and evaluation platform designed to assess whether a candidate in an online technical or behavioral interview is using hidden generative AI assistance (such as ChatGPT, Claude, live speech coaches, or teleprompter setups).

Unlike intrusive desktop monitoring software that requires root installation on candidate machines, OCULUS operates primarily through an **in-meeting bot plane** (joining Zoom, Google Meet, or Microsoft Teams as a participant), recording and processing audio/video streams to generate an objective, timestamped **Interview Integrity Report**.

---

## 2. End-to-End System Workflow

```
[HR / Recruiter Dashboard]
           │
           │ 1. Dispatch Request (Meeting URL, Candidate Name, Job Title)
           ▼
[FastAPI Sentinel Backend]
           │
           │ 2. Join Meeting via Adapter
           ▼
[Meeting Bot Gateway (Attendee / Recall)]
           │
           ├── Joins Google Meet / Zoom / MS Teams
           ├── Streams Audio (per participant or mixed)
           ├── Streams Video (candidate camera tile)
           └── Emits Roster & Status Events
           │
           ▼
[Live Ingestion & Processing Pipeline]
 ┌─────────────────────────┴─────────────────────────┐
 │                                                   │
 ▼ Audio Stream                                      ▼ Video Stream
[Speech-to-Text Pipeline]                   [Vision Proctoring Engine]
 - Whisper / STT Engine                     - OpenCV / MediaPipe / YOLO
 - Word timestamps & Diarization            - Gaze deviation tracking (Iris)
 - Transcript Normalizer & Filler cleaner   - Object detection (phone in lap)
 - Entity Stitcher (technical terms)        - Multi-face / off-screen prompt flags
 │                                                   │
 └─────────────────────────┬─────────────────────────┘
                           │
                           ▼
              [Multimodal Proctoring Service]
                           │
       ┌───────────────────┴───────────────────┐
       ▼ Stage 1                               ▼ Stage 2
[Deterministic Heuristics (0ms)]      [LLM Judge Reasoning (Groq / Gemini)]
 - Disfluency deficit (lack of fillers) - Formulaic answer structuring
 - AI Tell regex match ("delve into")  - Question-mirroring detection
 - Cadence & WPM anomalies             - Semantic depth vs recitation audit
 - Vision anomaly correlation          - Candidate vs job level calibration
       └───────────────────┬───────────────────┘
                           │
                           ▼
             [Report Generation & Storage]
 - Overall Integrity & Risk Score (0-100%)
 - Per-Answer Breakdown with verbatim quotes & evidence
 - Visual timeline of gaze/device flags
 - JSON report persistence (data/reports/)
                           │
                           ▼
           [Next.js 16 Web Dashboard]
 - Live Session Monitor (Real-time flags & speech stream)
 - Executive Integrity Summary (Low / Medium / High Risk)
 - Downloadable PDF / JSON Export
```

### Step-by-Step Flow

1. **Session Dispatch:** Recruiter enters meeting link, candidate name, interviewer name, and role into the Next.js frontend (`/sessions/new`).
2. **Bot Injection:** Backend adapter calls Attendee API to spawn a headless meeting participant that joins the room with a configurable display name (`VeriMeet Proctor Bot`).
3. **Stream Capture:**
   - **Audio:** Ingests raw audio, separates speakers, transcribes text with timestamps.
   - **Video:** Periodically samples candidate video tile at 1–5 FPS for anomaly detection.
4. **Real-Time Normalization:** Transcript is cleaned, stitched, and mapped against technical domain terms (e.g., distinguishing "Kafka" from "Caffa").
5. **Two-Stage Authenticity Scoring:**
   - **Deterministic Stage (Code):** Evaluates speech disfluency rate, self-reference density, and known LLM phrase tells.
   - **Reasoning Stage (LLM):** Prompts a high-speed reasoning model to judge whether complex technical answers were constructed live or read off an AI generation buffer.
6. **Reporting:** Generates an advisory report displaying visual proof points, quote snippets, and confidence bands.

---

## 3. Technology Stack

### Frontend Application
* **Framework:** Next.js 16.3 (React 19, App Router)
* **Styling:** Tailwind CSS v4, custom Apple-inspired design system (`globals.css`)
* **State & Interaction:** React Hooks, Server/Client component boundary, Theme Switcher (Dark/Light mode)
* **Data Fetching:** Native fetch with server-side proxying to FastAPI backend

### Backend Architecture
* **Runtime:** Python 3.11+
* **Web Framework:** FastAPI (Asynchronous REST API, CORS middleware, Webhook listeners)
* **Data Validation:** Pydantic v2 (Strict typing, JSON schema validation)
* **Process Orchestration:** Docker & Docker Compose (`python:3.11-slim`, Uvicorn ASGI)

### Speech & Vision Engines
* **Speech-to-Text (STT):** Groq Whisper Large v3 / Faster-Whisper / Speechmatics (replacing legacy Sarvam integration)
* **Text Processing:** Custom Python regex normalizer (`TranscriptEnhancer`), temporal utterance stitcher
* **Computer Vision:** OpenCV / MediaPipe for facial landmark and iris gaze tracking; YOLOv8 for secondary electronic device (phone/tablet) detection

### Reasoning & LLM Judge Layer
* **LLM Orchestration:** `langchain_openai` / LangChain Core
* **Reasoning Models:** Groq Cloud (`deepseek-r1-distill-llama-70b`, `llama-3.3-70b-versatile`) or Google AI Studio (`gemini-2.0-flash`)

### External Infrastructure
* **Meeting Bot Infrastructure:** Attendee API (`https://app.attendee.dev` or self-hosted Attendee cluster) / Recall.ai adapter fallback

---

## 4. Engineering Challenges & Critical Risks

### Risk 1: The "Meeting-Bot Blind Spot" (Hardware Level Defeat)
* **The Problem:** An in-meeting bot only sees what Zoom/Meet transmits. A candidate with a second smartphone propped directly below the webcam or on their lap can read ChatGPT prompts without touching their computer.
* **Mitigation:**
  1. OCULUS implements gaze angle deviation: prolonged downward glances (>3 seconds) during complex question delivery are flagged.
  2. The final report explicitly includes a legal disclaimer that meeting-plane proctoring provides behavioral inference, not operating-system telemetry.

### Risk 2: Meeting Platform Anti-Bot Defenses & Ingestion Latency
* **The Problem:**
  - Google Meet often requires manual host admission ("Someone wants to join this call").
  - Zoom and Teams frequently roll out Cloudflare bot challenges, CAPTCHAs, or admin tenant restrictions blocking third-party bots.
* **Mitigation:**
  - Build retry logic and clear webhook status alerts (`WAITING_IN_LOBBY`, `HOST_DENIED`).
  - Keep adapters modular so the team can switch from Attendee to Recall.ai or a client-side WebRTC extension without touching core proctoring code.

### Risk 3: False Positive Traps (ESL Candidates & Neurodiversity)
* **The Problem:**
  - Non-native English speakers often speak slowly, pause unnaturally, or repeat memorized phrasing.
  - Neurodivergent candidates (e.g., ADHD, Autism) frequently avoid camera eye contact or fidget.
  - Rigid algorithms can unfairly penalize honest candidates.
* **Mitigation:**
  - Scoring is **never binary** (no automatic "Cheating Detected" banner).
  - Use relative **baseline normalization**: calculate metrics against the candidate's own introductory talk (Question 0: "Tell me about yourself") rather than an arbitrary universal constant.
  - Reports are explicitly advisory and highlight observations for human interviewers to review.

### Risk 4: Audio Diarization & Crosstalk Confusion
* **The Problem:** When an interviewer interrupts or both parties speak simultaneously, single-channel audio mixes candidate answers with interviewer questions, distorting cadence and disfluency calculations.
* **Mitigation:**
  - Prioritize per-participant audio track recording via the meeting bot gateway over mixed room audio.
  - Implement temporal gap stitching (2,500ms threshold) to prevent sentence fragmenting.

### Risk 5: LLM Judge Hallucination & Cost/Rate Limits
* **The Problem:** Sending full 1-hour interview transcripts to proprietary LLMs on every turn causes API timeouts, rate-limit crashes on free tiers, and expensive token bills.
* **Mitigation:**
  - **Heuristic Pre-Filtering:** Only send Q&A pairs to the LLM judge if deterministic heuristics detect anomaly indicators (e.g., 3+ AI tell phrases or <1% disfluency on a 100+ word answer).
  - Structured output schemas (`model_dump_json`) enforce strict JSON parsing, avoiding parsing crashes.

---

## 5. Development Roadmap & Priorities

1. **Phase 1 (Current):** Modularization, sanitization of API secrets, Git repository setup, and clean project architecture.
2. **Phase 2:** Implement Groq Whisper Large v3 STT adapter to replace the legacy Sarvam speech service.
3. **Phase 3:** Integrate DeepSeek-R1 / Gemini 2.0 Flash into `proctoring_service.py` with structured Pydantic response formatting.
4. **Phase 4:** Live WebSocket streaming from Attendee to frontend for real-time risk gauges during the interview.
5. **Phase 5:** Exportable executive PDF reports (using WeasyPrint / React PDF).
