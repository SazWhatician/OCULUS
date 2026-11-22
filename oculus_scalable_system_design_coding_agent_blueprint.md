# OCULUS: Scalable System Design & Coding-Agent Implementation Blueprint

**Project:** OCULUS — AI Interview Integrity Sentinel & Multimodal Proctoring Platform  
**Repository:** `github.com/SazWhatician/OCULUS`  
**Target Cost:** **$0** for unlimited demos | **$0.25–$0.60 per 1-hour interview** in production  
**Architecture Style:** Stateless API Gateway + Redis Streams Event-Driven Workers + Pluggable Provider Adapters

---

## 1. Executive Summary & Design Goals

**OCULUS** evaluates whether a candidate in an online technical or behavioral interview is relying on hidden generative AI assistance (e.g., ChatGPT, Claude, live speech coaches, or off-screen teleprompters) without installing intrusive desktop spyware.

To make OCULUS **both free to demo and horizontally scalable in production**, this architecture enforces four core principles:

1. **Zero Local State:** Replace local `data/reports/` JSON storage with Postgres + S3-compatible object storage (Cloudflare R2 / local dev adapter) so services scale independently.
2. **Zero GPU Servers:** Replace continuous YOLOv8 inference with CPU-only MediaPipe facial/iris landmarking at 1–2 FPS, escalating to a Vision-Language Model (Gemini Flash) **only** on confirmed 3+ second anomalies.
3. **Zero-Cost Demo Modes:** Decouple meeting ingestion via an `IngestionAdapter` so you can run the full pipeline on uploaded MP4/WAV files or live browser tab capture (WebRTC) for **$0** without consuming paid meeting-bot minutes.
4. **Fairness & Cost Control by Design:** Calibrate speech metrics against each candidate's own Question 0 baseline to prevent ESL/neurodivergent false positives, and use deterministic heuristics to filter out 70–80% of normal answers before invoking the LLM Judge.

---

## 2. Zero-Cost Demo & Scalable Production Tech Stack

| Layer | Free Demo Stack ($0) | Scalable Production Path | Why It Scales & Saves Cost |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Vercel Hobby** (Next.js 16, React 19, Tailwind v4) | Vercel Pro / Cloudflare Pages | Edge-cached SSR/CSR dashboard; zero infrastructure overhead. |
| **Meeting Ingestion** | **Mode A:** Local Upload & Browser WebRTC Tab Capture (`$0`)<br>**Mode B:** **Attendee.dev** (5 hrs free trial) or Self-Hosted Attendee (Docker) | Self-Hosted **Attendee** cluster on Hetzner/AWS Spot (~$0.05/hr) or Hosted Attendee ($0.35–$0.50/hr) | Pluggable `IngestionAdapter` enables infinite $0 demos and instant failover between Attendee, Recall.ai, or browser capture. |
| **Speech-to-Text (STT)** | **Primary (Mixed Audio):** **Speechmatics** ($100 free credit, no card)<br>**Secondary (Per-track):** **Groq Whisper v3 Turbo** (Free: 8 hrs audio/day) | **Hybrid Router:** Groq Whisper for isolated speaker tracks; Speechmatics Melia 1 ($0.12/hr batch) or Linden 1 ($0.15/hr RT) for mixed audio | Eliminates custom diarization maintenance and avoids GPU Whisper hosting costs. |
| **Vision Proctoring** | **MediaPipe Face Landmarker** (CPU-only, 1–2 FPS) + **Gemini Flash** (Free Tier) for flagged frames | CPU Worker Pool + Gemini Flash (~$0.002/interview for 10–15 anomaly frames) | Eliminates GPU server bills while still detecting phones/second screens when gaze anomalies occur. |
| **LLM Judge** | **Groq Free Tier** (`qwen/qwen3-32b` or `llama-3.3-70b-versatile`) + **Google AI Studio Gemini Flash** fallback | Groq Paid / Gemini Flash + Heuristic Pre-Filter (blocks 70–80% of normal turns) | Keeps LLM token cost to fractions of a cent per interview with automatic 429 failover. |
| **Event Bus & Pub/Sub** | **Upstash Redis** (Free tier) or local Redis container | Upstash Redis / Managed Valkey | Redis Streams decouple ingestion from processing; Redis Pub/Sub streams live telemetry to WebSockets. |
| **Database** | **Neon** or **Supabase** Postgres (Free tier) | Managed Postgres with connection pooling | Supports multi-tenant dashboards, session history, and indexed evidence timelines. |
| **Blob Storage** | **Local Disk Adapter** (dev) / **Cloudflare R2** (10 GB free, $0 egress) | Cloudflare R2 / AWS S3 | Stores audio chunks and confirmed evidence keyframe JPEGs cheaply. |
| **Backend Hosting** | **Render / Koyeb / Fly.io** Free Tier (or Oracle Cloud Always Free ARM VM) | Fly.io Machines / AWS ECS Fargate autoscaling on Redis Stream lag | API stays lightweight; workers scale independently or run merged in `single_node` dev mode. |
| **PDF Export** | Client-side **`@react-pdf/renderer`** or `@media print` in Next.js | Same (client-side rendering) | Removes heavy WeasyPrint/Chromium dependencies from backend containers (<256 MB RAM). |

---

## 3. Speechmatics & Hybrid STT Strategy

Speechmatics solves two of OCULUS's toughest engineering bottlenecks—**speaker diarization crosstalk** and **technical domain transcription errors**—while providing **$100 in free credits (no credit card required)**, covering **250 to 800+ hours of free demo interviews**.

### Key Advantages of Speechmatics in OCULUS
1. **Native Speaker Diarization + Word-Level Timestamps:** Groq's Whisper endpoint does not perform diarization. Whenever OCULUS ingests mixed room audio (uploaded MP4/WAV files, browser tab capture, or single-track meeting recordings), Speechmatics separates `S1` (Interviewer) and `S2` (Candidate) with exact word timings.
2. **Custom Technical Dictionary (`additional_vocab`):** Replaces brittle regex maps (e.g., mapping `"Caffa"` to `"Kafka"`). At session creation, extract tech stack keywords from the job title/description and pass them directly to Speechmatics' custom dictionary config.
3. **ESL & Code-Switching Accuracy:** Speechmatics **Melia 1** ($0.12/hr batch) supports 55+ languages and mid-sentence code-switching, preventing false-positive disfluency/cadence flags caused by misheard accents.

### Hybrid STT Routing Policy (`STTRouter`)
* **Route to Speechmatics** when:
  * `AudioChunkEvent.is_mixed_audio == True` (requires speaker diarization), OR
  * Running high-precision post-interview batch audits (`Melia 1`), OR
  * Running live real-time WebSocket demos (`Linden 1`, up to 2 concurrent free streams).
* **Route to Groq `whisper-large-v3-turbo`** when:
  * `AudioChunkEvent.is_mixed_audio == False` (Attendee delivers isolated per-participant audio tracks, so diarization is unnecessary), OR
  * `SPEECHMATICS_API_KEY` is not configured (automatic zero-cost fallback using Groq's 28,800 audio seconds/day free tier).

---

## 4. End-to-End Event-Driven System Architecture

Never run STT, MediaPipe, FFmpeg, or LLM inference inside FastAPI HTTP/WebSocket handlers. Every component communicates exclusively through **Redis Streams** and **Redis Pub/Sub**.

```text
                       ┌──────────────────────────────────────────┐
                       │     INGESTION ADAPTERS (Pluggable)       │
                       │  1. AttendeeBotAdapter (Meet/Zoom/Teams) │
                       │  2. WebRTCTabAdapter (Free Live Demo)    │
                       │  3. DemoFileAdapter (Free MP4/WAV Demo)  │
                       └────────────────────┬─────────────────────┘
                                            │ HTTP Webhook / Upload / WS Chunk
                                            ▼
┌────────────────────────────┐   ┌──────────────────────────────────────────┐
│  Next.js 16 Dashboard      │◄──┤  FastAPI Gateway (Stateless, Async)      │
│  - Session Dispatch        │WS │  - Auth & Pydantic validation            │
│  - Live Risk Gauge         │   │  - Stores raw chunks in R2/Local         │
│  - Advisory Report & PDF   │   │  - Publishes to Redis Streams            │
└────────────────────────────┘   │  - Relays Redis Pub/Sub -> WebSocket     │
                                 └──────────┬───────────────────┬───────────┘
                                            │                   │
                       Redis Stream:        ▼                   ▼       Redis Stream:
                     `stream:audio`                             `stream:video`
                            │                                           │
                            ▼                                           ▼
              ┌──────────────────────────┐                ┌──────────────────────────┐
              │   STT Worker Pool        │                │   Vision Worker Pool     │
              │ - Speechmatics RT/Batch  │                │ - MediaPipe FaceMesh CPU │
              │ - Groq Whisper Fallback  │                │ - Pitch / Yaw / Iris     │
              │ - Temporal Stitcher      │                │ - Debouncer (>= 3000ms)  │
              └─────────────┬────────────┘                │ - Conditional VLM Check  │
                            │                             └─────────────┬────────────┘
                            │ `stream:utterances`                       │ `stream:vision_events`
                            └─────────────────────┬─────────────────────┘
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │   Scoring & Judge Worker    │
                                   │ 0. Q0 Baseline Calibrator   │
                                   │ 1. Deterministic Heuristics │
                                   │ 2. Pre-Filter Gate          │
                                   │ 3. LLM Judge (Groq->Gemini) │
                                   └──────────────┬──────────────┘
                                                  │
                              ┌───────────────────┴───────────────────┐
                              ▼                                       ▼
                   ┌─────────────────────┐                 ┌─────────────────────┐
                   │ Postgres (Neon)     │                 │ Redis Pub/Sub       │
                   │ - sessions          │                 │ `channel:session_id`│
                   │ - utterances        │                 │ -> Live Dashboard   │
                   │ - vision_events     │                 └─────────────────────┘
                   │ - qa_evaluations    │
                   │ - integrity_reports │
                   └─────────────────────┘
```

---

## 5. Strict Repository Structure

Implement modules following this exact directory tree so imports remain consistent across services:

```text
oculus/
├── docker-compose.yml              # 1-command local demo (Redis, Postgres, API, Workers)
├── .env.example                    # Provider feature flags & API keys
├── backend/
│   ├── pyproject.toml              # Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Redis, MediaPipe
│   ├── tests/
│   │   ├── test_heuristics.py      # Unit tests for Q0 baseline & pre-filter gate
│   │   ├── test_stitcher.py        # Unit tests for 2500ms utterance merging
│   │   └── test_contracts.py       # Schema serialization tests
│   └── app/
│       ├── main.py                 # FastAPI app, CORS, Lifespan (starts combined workers if single_node)
│       ├── config.py               # Pydantic BaseSettings
│       ├── schemas/
│       │   └── contracts.py        # SINGLE SOURCE OF TRUTH (all shared Pydantic v2 models)
│       ├── db/
│       │   ├── postgres.py         # Async SQLAlchemy engine, session factory, ORM tables
│       │   ├── redis_bus.py        # Redis Streams producer/consumer + Pub/Sub helper
│       │   └── storage.py          # StorageAdapter ABC + LocalDiskAdapter + CloudflareR2Adapter
│       ├── adapters/
│       │   ├── ingestion/
│       │   │   ├── base.py         # IngestionAdapter ABC
│       │   │   ├── attendee.py     # Attendee.dev bot spawn & webhook parser
│       │   │   └── demo_replay.py  # FFmpeg MP4/WAV splitter & WebRTC chunk handler
│       │   ├── stt/
│       │   │   ├── base.py         # STTAdapter ABC + STTRouter
│       │   │   ├── speechmatics.py # Speechmatics diarized STT + custom vocab
│       │   │   └── groq_whisper.py # Groq whisper-large-v3-turbo adapter
│       │   └── llm/
│       │       ├── base.py         # LLMJudgeAdapter ABC + FallbackRouter
│       │       ├── groq_judge.py   # Groq structured JSON judge
│       │       └── gemini_judge.py # Gemini Flash text judge + keyframe VLM verifier
│       ├── services/
│       │   ├── vocab_extractor.py  # Extracts technical domain terms from job_title/description
│       │   ├── stitcher.py         # Temporal utterance stitcher (2500ms merge window)
│       │   ├── heuristics.py       # Stage 0 Q0 calibrator + Stage 1 deterministic scorer
│       │   └── aggregator.py       # Computes combined risk & generates IntegrityReport
│       ├── workers/
│       │   ├── runner.py           # Entrypoint to run individual workers or combined single_node mode
│       │   ├── stt_worker.py       # Consumes stream:audio -> emits stream:utterances
│       │   ├── vision_worker.py    # Consumes stream:video -> emits stream:vision_events
│       │   └── scoring_worker.py   # Consumes utterances + vision -> Postgres + Pub/Sub
│       └── api/
│           ├── sessions.py         # Session CRUD + Integrity Report retrieval
│           ├── webhooks.py         # Attendee webhook + Demo file upload endpoint
│           └── ws.py               # Live dashboard WS + Browser WebRTC ingestion WS
└── frontend/                       # Next.js 16 (App Router, React 19, Tailwind v4)
    └── src/
        ├── app/sessions/new/page.tsx         # Dispatch bot OR upload demo file OR start tab capture
        ├── app/sessions/[id]/live/page.tsx   # Real-time transcript, gaze flags & rolling risk gauge
        └── app/sessions/[id]/report/page.tsx # Executive Integrity Summary + Client-side PDF export
```

---

## 6. Step-by-Step Coding-Agent Implementation Blueprint

Build and test in the exact order below (`Step 1` -> `Step 7`). Never pass untyped `dict` objects between services.

### Step 1: Shared Data Contracts (`backend/app/schemas/contracts.py`)

**Agent Objective:** Implement this file first. All API endpoints, Redis Stream messages, workers, and LLM responses must validate against these exact Pydantic v2 models.

```python
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class IngestionMode(str, Enum):
    ATTENDEE_BOT = "attendee_bot"
    BROWSER_WEBRTC = "browser_webrtc"
    DEMO_FILE = "demo_file"


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    WAITING_IN_LOBBY = "WAITING_IN_LOBBY"
    IN_CALL = "IN_CALL"
    HOST_DENIED = "HOST_DENIED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SpeakerRole(str, Enum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class AudioChunkEvent(BaseModel):
    session_id: str
    chunk_index: int
    timestamp_ms: int
    speaker_hint: SpeakerRole = SpeakerRole.UNKNOWN  # Populated for per-track bot audio
    storage_key: str                                 # Path in R2 or local tmp_storage
    is_mixed_audio: bool = True


class WordTiming(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


class UtteranceEvent(BaseModel):
    session_id: str
    utterance_id: str
    speaker: SpeakerRole
    speaker_label: str                               # e.g. "S1", "S2" from Speechmatics
    text: str
    start_ms: int
    end_ms: int
    words: list[WordTiming] = Field(default_factory=list)


class VisionFrameEvent(BaseModel):
    session_id: str
    timestamp_ms: int
    storage_key: str                                 # 640x360 JPEG sampled at 1-2 FPS


class VisionAnomalyEvent(BaseModel):
    session_id: str
    timestamp_ms: int
    anomaly_type: Literal[
        "GAZE_DEVIATION",
        "HEAD_PITCH_DOWN",
        "MULTI_FACE",
        "DEVICE_DETECTED",
    ]
    duration_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frame_key: Optional[str] = None         # Persisted only on confirmed anomaly
    vlm_verification_note: Optional[str] = None      # Populated if Gemini Flash VLM ran


class CandidateBaseline(BaseModel):
    """Calculated from Question 0 ('Tell me about yourself') to prevent ESL/neurodivergent bias."""
    session_id: str
    baseline_wpm: float
    baseline_disfluency_rate: float                  # Fillers per 100 words
    baseline_pause_ratio: float                      # Fraction of duration spent in pauses > 600ms
    baseline_gaze_variance: float = 0.0


class HeuristicScores(BaseModel):
    word_count: int
    wpm: float
    wpm_zscore_vs_baseline: float
    disfluency_rate: float
    disfluency_delta_vs_baseline: float              # Positive = sharper drop in fillers vs Q0
    ai_tell_matches: list[str]
    first_person_ratio: float
    concurrent_vision_flags: int
    deterministic_risk: float = Field(ge=0.0, le=100.0)
    trigger_llm_judge: bool                          # Pre-filter gate for Stage 2


class LLMJudgeVerdict(BaseModel):
    answer_id: str
    suspicion_score: float = Field(ge=0.0, le=100.0)
    confidence_band: Literal["LOW", "MEDIUM", "HIGH"]
    reasoning_signals: list[str]                     # Specific behavioral/linguistic observations
    recitation_vs_live_assessment: str
    advisory_follow_up_question: str                 # Probe interviewer can ask to verify depth


class QAPairEvaluation(BaseModel):
    qa_index: int
    question_text: str
    answer_text: str
    start_ms: int
    end_ms: int
    heuristics: HeuristicScores
    llm_verdict: Optional[LLMJudgeVerdict] = None
    combined_risk_score: float = Field(ge=0.0, le=100.0)


class IntegrityReport(BaseModel):
    session_id: str
    candidate_name: str
    job_title: str
    overall_risk_score: float = Field(ge=0.0, le=100.0)
    risk_band: Literal["LOW", "MODERATE", "ELEVATED"]
    disclaimer: str = (
        "Advisory behavioral inference derived from meeting-plane audio and video signals. "
        "Not operating-system telemetry and not definitive proof of misconduct."
    )
    baseline: Optional[CandidateBaseline] = None
    qa_evaluations: list[QAPairEvaluation]
    vision_timeline: list[VisionAnomalyEvent]
```

---

### Step 2: Infrastructure Layer (`backend/app/db/`)

**Agent Objective:** Build environment-agnostic wrappers for Redis Streams, Postgres, and Object Storage.

#### 2.1 `redis_bus.py`
* **Stream Keys:**
  * `stream:audio` (`AudioChunkEvent`)
  * `stream:video` (`VisionFrameEvent`)
  * `stream:utterances` (`UtteranceEvent`)
  * `stream:vision_events` (`VisionAnomalyEvent`)
* **Methods:**
  * `publish_stream(stream: str, event: BaseModel) -> str`: Calls `XADD` with `{"payload": event.model_dump_json()}` and `maxlen=5000, approximate=True` to keep free-tier Upstash Redis memory bounded.
  * `ensure_group(stream: str, group: str)`: Creates consumer group with `mkstream=True`, ignoring `BUSYGROUP`.
  * `consume_loop(stream: str, group: str, consumer: str, model_cls, handler)`: Reads via `XREADGROUP(block=2000, count=5)`, deserializes `model_cls.model_validate_json(payload)`, invokes `await handler(event)`, and calls `XACK` only on success.
  * `broadcast_ui(session_id: str, event_type: str, data: dict)`: Publishes JSON to `channel:session:{session_id}`.

#### 2.2 `postgres.py`
* Use `sqlalchemy.ext.asyncio` with `asyncpg` (supports Neon, Supabase, and local Docker Postgres).
* **Tables:**
  * `sessions`: `id (PK)`, `candidate_name`, `interviewer_name`, `job_title`, `custom_vocab (JSONB)`, `ingestion_mode`, `status`, `bot_external_id`, `baseline_json (JSONB)`, `created_at`.
  * `utterances`: `id (PK)`, `session_id (FK)`, `speaker`, `speaker_label`, `text`, `start_ms`, `end_ms`, `words_json (JSONB)`.
  * `vision_events`: `id (PK)`, `session_id (FK)`, `timestamp_ms`, `anomaly_type`, `duration_ms`, `confidence`, `evidence_frame_key`, `vlm_verification_note`.
  * `qa_evaluations`: `id (PK)`, `session_id (FK)`, `qa_index`, `payload_json (JSONB)` (stores `QAPairEvaluation`).
  * `integrity_reports`: `session_id (PK, FK)`, `overall_risk_score`, `risk_band`, `report_json (JSONB)`, `updated_at`.

#### 2.3 `storage.py`
* Define `StorageAdapter` ABC with async methods `put_bytes(key: str, data: bytes, content_type: str)`, `get_bytes(key: str) -> bytes`, `delete(key: str)`, and `get_presigned_url(key: str) -> str`.
* Implement `LocalDiskStorageAdapter` (writes to `./tmp_storage/`) when `STORAGE_PROVIDER=local`.
* Implement `CloudflareR2StorageAdapter` (using `aioboto3` or `boto3` wrapped in `asyncio.to_thread`) when `STORAGE_PROVIDER=r2`.

---

### Step 3: Pluggable Ingestion Adapters (`backend/app/adapters/ingestion/`)

**Agent Objective:** Support live bots AND zero-cost demo modes feeding identical events into `stream:audio` and `stream:video`.

1. **`attendee.py` (`AttendeeBotAdapter`):**
   * `spawn_bot(session_id, meeting_url)`: Calls `POST {ATTENDEE_BASE_URL}/api/v1/bots` with display name `OCULUS Sentinel` and webhook target `{PUBLIC_API_URL}/api/v1/webhooks/attendee`.
   * Webhook handler verifies signature, updates `sessions.status` (`WAITING_IN_LOBBY`, `IN_CALL`, `HOST_DENIED`, `COMPLETED`), broadcasts status via `redis_bus.broadcast_ui()`, stores media chunks via `StorageAdapter`, and publishes `AudioChunkEvent` / `VisionFrameEvent`.
2. **`demo_replay.py` (`DemoFileAdapter` — $0 Demo Mode):**
   * Accepts an uploaded `.mp4`, `.webm`, or `.wav` mock interview via `POST /api/v1/demo/upload`.
   * Runs `ffmpeg` in a background task (`asyncio.create_subprocess_exec`):
     * Extracts 16 kHz mono WAV audio, chunks into 15-second segments (or passes full audio for fast batch mode), uploads to storage, and emits `AudioChunkEvent(is_mixed_audio=True)`.
     * If video is present, samples frames at **1 FPS**, scales to `640x360` JPEG, and emits `VisionFrameEvent`s.
3. **Browser WebRTC Tab Capture (`api/ws.py` `/ws/ingest/{session_id}` — $0 Live Demo Mode):**
   * Frontend calls `navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })` to capture a live Google Meet / Zoom browser tab.
   * Streams 5-second `MediaRecorder` WebM audio blobs and 1 FPS `<canvas>` JPEGs over WebSocket to FastAPI, which stores and publishes them directly to `stream:audio` and `stream:video`.

---

### Step 4: Hybrid STT Worker & Temporal Stitcher (`backend/app/adapters/stt/` & `workers/stt_worker.py`)

**Agent Objective:** Transcribe audio with accurate domain vocabulary and speaker labels, then merge fragmented sentences.

```python
# backend/app/adapters/stt/base.py
from abc import ABC, abstractmethod
from app.schemas.contracts import UtteranceEvent


class STTAdapter(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        session_id: str,
        offset_ms: int,
        custom_vocab: list[str],
        diarize: bool = True,
    ) -> list[UtteranceEvent]:
        ...
```

* **`speechmatics.py` (`SpeechmaticsAdapter`):**
  * Uses `httpx.AsyncClient` against Speechmatics Batch API (`Melia 1`) or WebSocket RT API (`Linden 1`).
  * Passes:
    ```json
    {
      "type": "transcription",
      "transcription_config": {
        "language": "en",
        "diarization": "speaker",
        "operating_point": "enhanced",
        "additional_vocab": [{"content": "Kafka"}, {"content": "Kubernetes"}]
      }
    }
    ```
  * Groups contiguous words sharing the same `speaker` tag (`S1`, `S2`) into `UtteranceEvent` objects with shifted `start_ms = offset_ms + int(word_start * 1000)`.
* **`groq_whisper.py` (`GroqWhisperAdapter`):**
  * Used when `event.is_mixed_audio is False` OR when `STT_PROVIDER=groq`.
  * Calls `https://api.groq.com/openai/v1/audio/transcriptions` with `model="whisper-large-v3-turbo"`, `response_format="verbose_json"`, `timestamp_granularities=["word"]`, and `prompt=", ".join(custom_vocab)`.
  * Assigns `speaker = event.speaker_hint`.
* **`services/stitcher.py` (`UtteranceStitcher`):**
  * Stores the latest open utterance per `session_id` in Redis Hash `stitch:{session_id}`.
  * **Merge Rule:** If `new_utt.speaker_label == prev_utt.speaker_label` AND `(new_utt.start_ms - prev_utt.end_ms) <= 2500`, concatenate `text`, extend `words`, and update `end_ms`.
  * **Flush Rule:** When the speaker changes or gap exceeds `2500ms`, persist `prev_utt` to Postgres, publish to `stream:utterances`, and broadcast live transcript text to `channel:session:{session_id}`.
* **Speaker Role Assignment:** Default the speaker who asks the opening question (`"?"` or earliest short prompt) as `INTERVIEWER` and the responder as `CANDIDATE`; expose `PATCH /api/v1/sessions/{id}/speakers` so a recruiter can swap roles in 1 click if needed.

---

### Step 5: Zero-GPU Vision Proctoring Worker (`backend/app/workers/vision_worker.py`)

**Agent Objective:** Detect prolonged gaze deviations, downward head pitch, multiple faces, and hidden devices using CPU-only MediaPipe + gated VLM calls.

1. Consume `VisionFrameEvent` from `stream:video`. Fetch JPEG bytes from `StorageAdapter` and decode with `cv2.imdecode`.
2. Run **MediaPipe Face Landmarker** (`num_faces=3`, `output_facial_transformation_matrixes=True`, CPU delegate, ~10–15ms/frame).
3. Compute per-frame features:
   * **Multi-face check:** `len(face_landmarks) >= 2` -> emit `MULTI_FACE`.
   * **Head Pose (Euler Angles):** Decompose the 4x4 facial transformation matrix via `cv2.RQDecomp3x3` to extract `pitch` (looking down at lap/phone) and `yaw` (looking sideways at a second monitor).
   * **Iris Gaze Offset:** Measure normalized horizontal/vertical ratio of iris center landmarks (`468`, `473`) relative to eye corner landmarks (`33`, `133`, `362`, `263`).
4. **Stateful Anomaly Debouncer (Redis Hash `vision_state:{session_id}`):**
   * Ignore brief glances (< 3 seconds).
   * Thresholds: `abs(yaw) > 22 deg` or `pitch < -18 deg` or `abs(iris_offset) > 0.35`.
   * When continuous anomaly duration reaches `>= 3000ms`, emit a `VisionAnomalyEvent` (`HEAD_PITCH_DOWN` or `GAZE_DEVIATION`).
5. **Conditional Gemini Flash VLM Verification (Zero-GPU Device Detection):**
   * Trigger **only** when a `>= 3000ms` anomaly fires AND last VLM call for this session was `> 30 seconds` ago.
   * Send the single JPEG frame to `gemini_judge.verify_frame(jpeg_bytes)` requesting strict JSON:
     ```json
     {
       "phone_or_tablet_visible": false,
       "secondary_screen_reflection": false,
       "note": "Candidate looking downward toward desk area."
     }
     ```
   * If `phone_or_tablet_visible == True`, emit `DEVICE_DETECTED` and retain `evidence_frame_key` in storage.
   * **Storage Hygiene:** Delete all non-anomalous frames from storage immediately after MediaPipe runs to keep R2/disk usage near zero.

---

### Step 6: Two-Stage Scoring & Baseline Calibration Worker (`backend/app/services/heuristics.py` & `workers/scoring_worker.py`)

**Agent Objective:** Pair interviewer questions with candidate answers, calibrate against Q0, run 0ms heuristics, and selectively invoke the LLM Judge.

#### 6.1 Question-Answer Pairing
* Group contiguous `INTERVIEWER` utterances into `question_text`, followed by contiguous `CANDIDATE` utterances into `answer_text` (`qa_index = 0, 1, 2, ...`).
* Skip trivial candidate acknowledgments (`word_count < 15`, e.g., `"Yes, I can hear you"`).

#### 6.2 Stage 0 — Candidate Baseline Calibration (`qa_index == 0`)
* On the first substantive candidate answer (`word_count >= 40`, typically `"Tell me about yourself"`), compute and persist `CandidateBaseline`:
  * `baseline_wpm = word_count / max((end_ms - start_ms) / 60000.0, 0.05)`
  * `baseline_disfluency_rate = (filler_count / word_count) * 100.0`
    * Filler set: `{"um", "uh", "erm", "like", "you know", "basically", "actually", "sort of", "kind of", "i mean"}`
  * `baseline_pause_ratio = sum(gap_ms for gap_ms > 600) / max(end_ms - start_ms, 1)`

#### 6.3 Stage 1 — Deterministic Heuristics (`0ms`, Pure Python)
For every subsequent Q&A pair (`qa_index >= 1`), compute:
1. **Disfluency Drop vs Baseline:** `disfluency_delta = baseline.baseline_disfluency_rate - curr_disfluency_rate`. Only penalize a *drop* relative to the candidate's own baseline (protects naturally articulate speakers and ESL speakers alike).
2. **Cadence / Pacing Z-Score:** Compare current WPM against `baseline_wpm` (assuming standard deviation floor of `20 WPM`) and flag robotic inter-word gap uniformity.
3. **LLM Tell Phrase Bank:** Case-insensitive regex matches against high-signal spoken LLM phrasing:
   * `r"\bdelve into\b"`, `r"\bit'?s (crucial|important) to note\b"`, `r"\bin summary\b"`, `r"\bmultifaceted\b"`, `r"\bcomprehensive approach\b"`, `r"\bfirstly\b.*\bsecondly\b"`, `r"\bleverage\b.*\bsynergy\b"`, `r"\bin conclusion\b"`.
4. **First-Person Density:** Count `I / me / my / we / our` divided by `word_count`. Textbook/LLM recitations on behavioral/experience questions typically exhibit near-zero self-reference (`< 0.015`).
5. **Vision Correlation:** Count `vision_events` stored in Postgres with `timestamp_ms` inside `[question.start_ms, answer.end_ms]`.
6. **Deterministic Risk Formula (0–100):**
   * Weighted sum clamped to `[0, 100]`:
     * `+25` if `disfluency_delta >= 2.5` and `word_count >= 80`
     * `+15 * min(len(ai_tell_matches), 3)`
     * `+20` if `abs(wpm_zscore) >= 2.0`
     * `+15 * min(concurrent_vision_flags, 2)`
     * `+10` if `first_person_ratio < 0.015` and `word_count >= 80`
7. **Pre-Filter Gate (`trigger_llm_judge`):**
   * Set `True` **ONLY IF** `word_count >= 50` AND any of:
     * `deterministic_risk >= 45.0`
     * `len(ai_tell_matches) >= 2`
     * `concurrent_vision_flags >= 1`
     * `disfluency_delta_vs_baseline >= 2.5`

#### 6.4 Stage 2 — Structured LLM Judge with Automatic Failover (`backend/app/adapters/llm/`)
* **Prompt Payload (Small & Cheap):** Send **only** the single Q&A pair, `job_title`, `CandidateBaseline`, and `HeuristicScores` summary—never the full transcript.
* **System Instruction:** Instruct the model to act as a senior technical interviewer assessing whether the answer shows live cognitive construction (self-corrections, concrete trade-offs, personal war stories) vs. verbatim AI recitation (over-polished listicles, question mirroring, generic abstractions). Remind the model not to penalize non-native English grammar or accents.
* **Structured Output:** Enforce strict JSON schema matching `LLMJudgeVerdict`.
* **Fallback Chain (`FallbackLLMRouter`):**
  1. Call **Groq** (`qwen/qwen3-32b` or `llama-3.3-70b-versatile`) with `tenacity` retry (max 2 retries on network blip).
  2. On HTTP `429` (rate limit) or `5xx`, immediately fail over to **Google AI Studio Gemini Flash**.
  3. On total failure, return `None` and fall back gracefully to Stage 1 score.

#### 6.5 Score Aggregation (`services/aggregator.py`)
* **Per-Answer Combined Risk:**
  * If `llm_verdict` is present: `combined_risk_score = 0.35 * heuristics.deterministic_risk + 0.65 * llm_verdict.suspicion_score`
  * If Stage 2 was skipped by the gate: `combined_risk_score = heuristics.deterministic_risk`
* **Overall Session Score:** Weighted average of `combined_risk_score` across evaluated Q&A pairs (weighted by `word_count`) plus a capped vision bonus (`+15` max for confirmed `DEVICE_DETECTED` or `MULTI_FACE`).
* **Risk Band Mapping (Advisory Only — Never Binary):**
  * `0.0 – 34.9`: `LOW`
  * `35.0 – 64.9`: `MODERATE`
  * `65.0 – 100.0`: `ELEVATED`

---

### Step 7: API Gateway, WebSockets & Next.js 16 Dashboard (`backend/app/api/` & `frontend/`)

#### REST & WebSocket Contract

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/sessions` | `POST` | Creates session, extracts `custom_vocab` from `job_title`/JD, dispatches Attendee bot (if `attendee_bot`) or returns ready session ID for demo modes. |
| `/api/v1/sessions/{id}` | `GET` | Returns current session metadata, status, and rolling metrics. |
| `/api/v1/sessions/{id}/speakers` | `PATCH` | Allows recruiter to swap `S1`/`S2` speaker-to-role mapping if diarization labels inverted. |
| `/api/v1/demo/upload` | `POST` | Accepts multipart MP4/WAV for free local replay through the full Redis worker pipeline. |
| `/api/v1/webhooks/attendee` | `POST` | Handles Attendee bot lifecycle (`WAITING_IN_LOBBY`, `HOST_DENIED`, `IN_CALL`, `ENDED`) and media callbacks. |
| `/ws/ingest/{id}` | `WS` | Accepts live browser WebRTC audio blobs & 1 FPS video frames for $0 live demos. |
| `/ws/sessions/{id}` | `WS` | Subscribes to `channel:session:{id}` and pushes live transcript utterances, gaze flags, and updated risk scores to Next.js. |
| `/api/v1/sessions/{id}/report` | `GET` | Returns full `IntegrityReport` JSON with presigned URLs for any evidence keyframes. |

#### Frontend Views (`frontend/src/app/`)
1. **`/sessions/new`:** Three clean mode tabs—**Demo File Upload ($0)**, **Live Browser Tab Capture ($0)**, and **Meeting Bot Dispatch (Attendee)**—plus Candidate Name, Job Title, and optional Job Description (used to seed Speechmatics `custom_vocab`).
2. **`/sessions/[id]/live`:** Connects to `/ws/sessions/{id}` to display connection/lobby status, live diarized transcript stream, vision anomaly badges, rolling risk gauge, and suggested **Advisory Follow-Up Questions** generated by the LLM Judge in real time.
3. **`/sessions/[id]/report`:** Renders the advisory disclaimer banner, `LOW / MODERATE / ELEVATED` band, Q0 baseline comparison card, per-answer expandable evidence cards (with verbatim quotes + heuristic/LLM signals), visual gaze/device timeline, and an **Export PDF** button powered by `@react-pdf/renderer` (zero backend load).

---

## 7. Environment Configuration & Deployment Modes

### `.env.example`

```env
# --- Deployment Mode ---
# single_node: FastAPI runs all 3 async workers in-process via asyncio.gather (fits 1 free container!)
# distributed: API and workers run as separate horizontally-scalable containers
DEPLOY_MODE=single_node

# --- Provider Switches ---
INGESTION_PROVIDER=demo_file            # demo_file | browser_webrtc | attendee_bot
STT_PROVIDER=hybrid                     # hybrid | speechmatics | groq
LLM_PRIMARY_PROVIDER=groq               # groq | gemini
STORAGE_PROVIDER=local                  # local | r2

# --- Database & Event Bus ---
DATABASE_URL=postgresql+asyncpg://oculus:oculus@postgres:5432/oculus
REDIS_URL=redis://redis:6379/0

# --- API Keys (Leave unused ones blank for $0 local demo) ---
SPEECHMATICS_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
ATTENDEE_API_KEY=
ATTENDEE_BASE_URL=https://app.attendee.dev
PUBLIC_API_URL=http://localhost:8000

# --- Cloudflare R2 (Production Storage) ---
R2_ENDPOINT_URL=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=oculus-media
```

### Dual-Mode Scaling Strategy (`DEPLOY_MODE`)
* **Free Hosting (`DEPLOY_MODE=single_node`):** Inside `app/main.py` FastAPI lifespan, spawn `stt_worker`, `vision_worker`, and `scoring_worker` as background `asyncio.Task`s (with CPU-bound MediaPipe calls dispatched via `asyncio.to_thread`). This runs the entire backend on **a single free-tier container** (Render/Koyeb/Fly.io) alongside free Upstash Redis and free Neon Postgres.
* **Production Scale (`DEPLOY_MODE=distributed`):** Set `DEPLOY_MODE=distributed` and deploy `uvicorn app.main:app` independently from `python -m app.workers.runner --worker stt|vision|scoring`. Autoscale worker replicas based on Redis Stream consumer group lag (`XPENDING`).

---

## 8. Engineering Risk Mitigations Summary

| Risk | Architectural Mitigation Built Into OCULUS |
| :--- | :--- |
| **Hardware-Level Blind Spot** (phone below webcam) | MediaPipe head-pitch/iris debouncer (`>= 3000ms`) + conditional Gemini Flash VLM check + mandatory legal/advisory disclaimer in `IntegrityReport`. |
| **Meeting Platform Anti-Bot Defenses** | Webhook status alerts (`WAITING_IN_LOBBY`, `HOST_DENIED`), exponential retry, and swappable `IngestionAdapter` (Attendee / Recall.ai / Browser WebRTC extension). |
| **ESL & Neurodivergent False Positives** | Mandatory Q0 `CandidateBaseline` normalization, non-binary advisory bands (`LOW`/`MODERATE`/`ELEVATED`), and Speechmatics multilingual/accent accuracy. |
| **Audio Crosstalk & Fragmentation** | Speechmatics native diarization for mixed audio, Groq isolated per-track routing when available, and 2,500ms `UtteranceStitcher`. |
| **LLM Rate Limits & Token Bills** | Stage 1 pre-filter gate skips 70–80% of normal turns; single-turn prompt payloads; `tenacity` backoff + Groq -> Gemini Flash automatic failover. |

---

## 9. Coding-Agent Guardrails & Verification Checklist

When implementing this specification, follow these strict rules:

1. **Strict Build Order:** `contracts.py` -> `config.py` + `db/` -> `services/heuristics.py` + `services/stitcher.py` -> unit tests (`pytest`) -> `adapters/` -> `workers/` -> `api/` -> `frontend/`.
2. **Non-Blocking Async Rule:** Never call `requests`, blocking `boto3`, or synchronous `ffmpeg` on the FastAPI event loop. Use `httpx.AsyncClient`, `asyncio.create_subprocess_exec`, and `asyncio.to_thread` for CPU-bound MediaPipe/OpenCV calls.
3. **Graceful Degradation:** If `GEMINI_API_KEY` is missing, skip VLM frame verification without crashing. If `SPEECHMATICS_API_KEY` is missing, route automatically to `GroqWhisperAdapter`.
4. **No Hardcoded Secrets:** All credentials load strictly from `app/config.py` (`pydantic-settings`).
5. **Definition of Done for Demo:** Running `docker compose up --build`, opening `/sessions/new`, uploading a 2-minute sample `.mp4` interview, watching live transcript/flags stream over WebSocket, and exporting a structured PDF report—all at **$0 infrastructure cost**.