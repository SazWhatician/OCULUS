# OCULUS: Principal-Level System Design & Code-Free Coding-Agent Blueprint

**System:** OCULUS — Event-Driven Multimodal Interview Integrity Sentinel  
**Operating Cost:** **$0.00** (Free-Tier Cloud + Unlimited Local/WebRTC Demos) | Scales linearly to production with zero code rewrites  
**Design Caliber:** Production Distributed Systems (Control/Data Plane isolation, Idempotent Stream Processing, Circuit Breakers, Quota-Aware Routing, Robust Statistical Calibration)  
**Purpose of Document:** Pure-text, zero-code architectural specification. Every module, symbol, formula, state machine, and edge case is dictated explicitly so a low-level coding agent can implement the entire platform cleanly without ambiguity.

---

## 1. Architectural Thesis & Zero-Cost Engineering Matrix

Most naive proctoring prototypes fail on three fronts: they run blocking AI calls inside HTTP handlers, burn GPU servers on continuous object detection, and produce biased false positives on non-native English or neurodivergent candidates. **OCULUS** eliminates all three flaws while staying strictly inside **$0 free tiers**.

### 1.1 The $0 Free-Tier Resource Architecture

| Subsystem | Zero-Cost Provider | Free-Tier Ceiling | Architectural Trick That Keeps Cost at $0 |
| :--- | :--- | :--- | :--- |
| **Web Dashboard** | **Vercel Hobby** (Next.js 16) | Unlimited static/SSR demos | Moves PDF report rendering entirely to the browser (`@react-pdf/renderer`), eliminating backend Chromium/WeasyPrint RAM overhead. |
| **Backend & Workers** | **Oracle Cloud Always Free ARM** (or **Render / Fly.io** Free Tier) | 1 container (512 MB RAM) or Always-Free VM | Dual-mode runtime (`single_node` vs. `distributed`): runs API + all 3 async workers in one supervised Python process on free hosting, or splits into independent containers at scale. |
| **Event Bus & Cache** | **Upstash Redis** (or Local Redis) | Bounded monthly command & memory caps | Uses **blocking stream reads** (`BLOCK 5000`) instead of tight polling loops, reducing idle Redis commands by $> 99\%$, plus approximate stream trimming (`MAXLEN`). |
| **Primary Database** | **Neon** or **Supabase** Postgres | 500 MB storage, serverless scale-to-zero | Stores only structured JSONB metadata, transcripts, and scores—never raw audio or video blobs. |
| **Media Object Store** | **Cloudflare R2** (or Local Disk Dev Adapter) | 10 GB storage, **$0 egress fees** | **Ephemeral Media Lifecycle:** deletes non-anomalous video frames within 15 milliseconds of MediaPipe processing and purges raw audio after session finalization. |
| **Meeting Ingestion** | **Browser WebRTC Tab Capture** + **MP4/WAV Replay** + **Attendee.dev** Trial | Infinite local/tab demos + 5 hrs Attendee free | Pluggable `IngestionAdapter` feeds identical Redis Stream events regardless of whether media comes from a paid meeting bot, a live Chrome Meet tab, or an uploaded recording. |
| **Speech-to-Text** | **Speechmatics** ($100 free credit, no card) + **Groq Whisper v3 Turbo** | ~650–800 hrs Speechmatics + **8 hrs/day** Groq | **Hybrid Cost Router:** Groq handles isolated per-speaker tracks at $0/day; Speechmatics handles mixed-audio diarization and job-specific technical dictionaries. |
| **Vision Proctoring** | **MediaPipe Face Landmarker (CPU)** + **Gemini Flash** (Free Tier) | Unlimited local CPU + free VLM tier | **3-Tier Vision Funnel:** skips static frames (Tier 0), runs 12ms CPU head-pose/iris math (Tier 1), and escalates to Gemini Flash VLM **only** on sustained $\ge 3\text{ s}$ anomalies (capped at 12 calls/session). |
| **Reasoning Judge** | **Groq Free Tier** (`qwen/qwen3-32b` / `llama-3.3-70b`) + **Gemini Flash** | Free RPM/TPD quotas on both | **Deterministic Pre-Filter Gate** eliminates 70–80% of normal answers before any LLM call; sends single Q&A turns instead of full transcripts; auto-fails over on HTTP 429. |

---

## 2. Distributed Systems Topology & Data-Flow Guarantees

### 2.1 Control Plane vs. Data Plane Separation
* **Control Plane (FastAPI Gateway):** Handles session CRUD, recruiter authentication, Attendee webhooks, presigned URL generation, and WebSocket fanout. It **never** executes FFmpeg, MediaPipe, STT, or LLM inference. Its strict p99 response budget is $< 50\text{ ms}$.
* **Data Plane (Async Worker Mesh):** Three decoupled consumer pools communicate strictly via Redis Streams:
  1. `stream:audio` consumed by **STT Worker** $\rightarrow$ emits to `stream:utterances`
  2. `stream:video` consumed by **Vision Worker** $\rightarrow$ emits to `stream:vision_events`
  3. `stream:utterances` + `stream:vision_events` consumed by **Scoring Worker** $\rightarrow$ writes to **Postgres** and fans out live updates via Redis Pub/Sub (`channel:session:{session_id}`).

### 2.2 Delivery Semantics, Idempotency & Crash Recovery
Free-tier containers restart unpredictably. OCULUS guarantees **at-least-once delivery with effectively-once state updates**:
* **Deterministic Idempotency Keys:** Every ingested media unit carries a deterministic key: `idem:{session_id}:{stream}:{sequence_index}`.
* **Deduplication Guard:** Before writing to Postgres or invoking paid/rate-limited APIs, every worker executes an atomic Redis `SET key 1 NX EX 86400` (backed by Postgres `UNIQUE` constraints with `ON CONFLICT DO NOTHING`).
* **Self-Healing Orphan Recovery:** On startup and every 60 seconds, each worker runs `XAUTOCLAIM` on its consumer group for pending messages idle $> 60\text{ s}$ (indicating a crashed container) and reprocesses them cleanly.
* **Dead-Letter Queue (DLQ):** If a message fails processing 3 consecutive times (tracked via Redis delivery count), the worker acknowledges it from the main stream and moves the payload plus stack trace metadata to `stream:dlq` so one poison pill never stalls an interview.

### 2.3 Asymmetric Backpressure & Load Shedding
Audio and video have fundamentally different durability requirements:
* **Audio (`stream:audio`) is Lossless:** Never drop audio chunks. Cap stream length at `MAXLEN ~ 5000`. If STT workers lag, audio queues safely in R2/Redis and catches up asynchronously.
* **Video (`stream:video`) is Lossy/Droppable:** Cap stream length at `MAXLEN ~ 300`. If a frame's `timestamp_ms` is older than $10\text{ s}$ behind wall-clock time during a live session, the Vision Worker immediately discards the frame, deletes it from storage, and jumps to the freshest frame to preserve real-time responsiveness.

### 2.4 Dual-Mode Execution Engine (`DEPLOY_MODE`)
* **`single_node` Mode (Free Hosting):** FastAPI's lifespan manager launches all three workers as supervised `asyncio.Task` coroutines in the same container, wrapping CPU-bound OpenCV/MediaPipe calls in a bounded `ThreadPoolExecutor(max_workers=2)` so the HTTP/WebSocket event loop never stalls.
* **`distributed` Mode (Production Scale):** FastAPI runs only the API gateway. Separate worker containers execute the worker runner module targeted at a single queue (`stt`, `vision`, or `scoring`), scaling horizontally based on Redis `XPENDING` lag.

---

## 3. Zero-Cost Resilience & Quota Engineering Patterns

### 3.1 Three-State Circuit Breaker (`CircuitBreaker`)
Wrap every external provider adapter (`Speechmatics`, `GroqWhisper`, `GroqLLM`, `GeminiFlash`, `Attendee`) in a shared Redis-backed state machine:
* **`CLOSED` (Healthy):** Requests pass normally. Track rolling failures over a 60-second window. If failure rate hits 5 consecutive `429`/`5xx`/timeouts, transition to `OPEN`.
* **`OPEN` (Tripped):** Fail fast locally in $< 1\text{ ms}$ without making network calls for a cooldown period of $30\text{ s}$ (or the HTTP `Retry-After` header duration). Immediately invoke the next fallback provider in the chain.
* **`HALF_OPEN` (Probing):** After cooldown expires, allow 1 probe request. Success transitions back to `CLOSED`; failure resets to `OPEN` with exponential backoff (capped at $300\text{ s}$).

### 3.2 Free-Tier Token-Bucket Quota Manager (`QuotaGuard`)
Maintain per-provider Redis token buckets using atomic Lua scripts (or pipeline `INCRBY` + `EXPIRE`) so demo traffic never triggers account bans:
* **Groq Whisper Bucket:** Enforces max 18 requests/min and $6{,}800$ audio-seconds/hour (leaving safety headroom below Groq's 20 RPM / $7{,}200$ sec/hr free limits).
* **Gemini Flash VLM Bucket:** Enforces max 10 requests/min globally AND a strict per-session ceiling of **12 VLM frame verifications per interview** with a **30-second per-session cooldown**.
* **Fallback Cascade:**
  * *STT Mixed Audio:* Speechmatics $\rightarrow$ (if key missing or circuit open) Groq Whisper with heuristic turn-change labeling.
  * *STT Isolated Tracks:* Groq Whisper $\rightarrow$ (if quota exhausted) Speechmatics.
  * *LLM Judge:* Groq (`qwen/qwen3-32b` or `llama-3.3-70b-versatile`) $\rightarrow$ Gemini Flash $\rightarrow$ Stage 1 Deterministic Fallback (never crashes the pipeline).
  * *Vision VLM Check:* Gemini Flash $\rightarrow$ (if exhausted/missing) MediaPipe-only gaze/head-pose flag with `vlm_verification_note = "VLM skipped (quota/offline)"`.

---

## 4. Canonical Data Contracts & Storage Blueprint

The coding agent must enforce strict Pydantic v2 typing across every boundary. Untyped dictionaries are forbidden between modules.

### 4.1 Core Pydantic v2 Contracts (`backend/app/schemas/contracts.py`)
Define the following models with strict field types and validators:

1. **Enums:**
   * `IngestionMode`: `attendee_bot`, `browser_webrtc`, `demo_file`
   * `SessionStatus`: `CREATED`, `WAITING_IN_LOBBY`, `IN_CALL`, `HOST_DENIED`, `PROCESSING`, `COMPLETED`, `FAILED`
   * `SpeakerRole`: `interviewer`, `candidate`, `unknown`
   * `VisionAnomalyType`: `GAZE_DEVIATION`, `HEAD_PITCH_DOWN`, `MULTI_FACE`, `DEVICE_DETECTED`
   * `RiskBand`: `LOW`, `MODERATE`, `ELEVATED`
   * `ConfidenceBand`: `LOW`, `MEDIUM`, `HIGH`
2. **`AudioChunkEvent`:** `idempotency_key` (str), `session_id` (str), `chunk_index` (int), `timestamp_ms` (int), `duration_ms` (int), `speaker_hint` (`SpeakerRole`), `storage_key` (str), `is_mixed_audio` (bool).
3. **`WordTiming`:** `word` (str), `start_ms` (int), `end_ms` (int), `confidence` (float, $[0, 1]$).
4. **`UtteranceEvent`:** `utterance_id` (str), `session_id` (str), `speaker` (`SpeakerRole`), `speaker_label` (str, e.g., `S1`/`S2`), `text` (str), `start_ms` (int), `end_ms` (int), `words` (list of `WordTiming`).
5. **`VisionFrameEvent`:** `idempotency_key` (str), `session_id` (str), `frame_index` (int), `timestamp_ms` (int), `ingested_at_ms` (int), `storage_key` (str), `is_live` (bool).
6. **`VisionAnomalyEvent`:** `event_id` (str), `session_id` (str), `timestamp_ms` (int), `anomaly_type` (`VisionAnomalyType`), `duration_ms` (int), `confidence` (float), `pitch_deg` (optional float), `yaw_deg` (optional float), `evidence_frame_key` (optional str), `vlm_verification_note` (optional str).
7. **`CandidateBaseline`:** `session_id` (str), `median_wpm` (float), `mad_wpm` (float), `baseline_disfluency_rate` (float), `baseline_pause_ratio` (float), `baseline_ttfs_ms` (float, median time-to-first-substantive-word), `sample_word_count` (int).
8. **`HeuristicScores`:** `word_count` (int), `wpm` (float), `wpm_robust_zscore` (float), `disfluency_rate` (float), `disfluency_drop` (float), `ttfs_latency_ms` (int), `cadence_uniformity_cv` (float, coefficient of variation of inter-word gaps), `ai_tell_matches` (list of str), `first_person_ratio` (float), `concurrent_vision_flags` (int), `deterministic_risk` (float, $[0, 100]$), `trigger_llm_judge` (bool), `trigger_reasons` (list of str).
9. **`LLMJudgeVerdict`:** `answer_id` (str), `provider_used` (str), `suspicion_score` (float, $[0, 100]$), `confidence_band` (`ConfidenceBand`), `reasoning_signals` (list of str), `verbatim_evidence_quotes` (list of str), `recitation_vs_live_assessment` (str), `advisory_follow_up_question` (str).
10. **`QAPairEvaluation`:** `qa_index` (int), `question_text` (str), `answer_text` (str), `question_end_ms` (int), `answer_start_ms` (int), `answer_end_ms` (int), `heuristics` (`HeuristicScores`), `llm_verdict` (optional `LLMJudgeVerdict`), `combined_risk_score` (float, $[0, 100]$).
11. **`IntegrityReport`:** `session_id` (str), `candidate_name` (str), `job_title` (str), `overall_risk_score` (float, $[0, 100]$), `risk_band` (`RiskBand`), `advisory_disclaimer` (str, mandatory legal notice), `baseline` (optional `CandidateBaseline`), `qa_evaluations` (list of `QAPairEvaluation`), `vision_timeline` (list of `VisionAnomalyEvent`), `generated_at_iso` (str).

### 4.2 Relational Schema (`backend/app/db/postgres.py`)
Implement via `SQLAlchemy 2.0` async ORM + `asyncpg` connection pool (`pool_size=5`, `max_overflow=5` to fit Neon free tier):
* **`sessions`:** `id` (UUID PK), `candidate_name`, `interviewer_name`, `job_title`, `job_description`, `custom_vocab` (JSONB), `ingestion_mode`, `status`, `bot_external_id` (indexed), `speaker_role_map` (JSONB, e.g., `{"S1": "interviewer", "S2": "candidate"}`), `baseline_json` (JSONB), `created_at`, `completed_at`.
* **`utterances`:** `id` (UUID PK), `session_id` (FK, indexed), `sequence_key` (UNIQUE with `session_id`), `speaker`, `speaker_label`, `text`, `start_ms` (indexed), `end_ms`, `words_json` (JSONB).
* **`vision_events`:** `id` (UUID PK), `session_id` (FK, indexed), `timestamp_ms` (indexed), `anomaly_type`, `duration_ms`, `confidence`, `evidence_frame_key`, `vlm_verification_note`.
* **`qa_evaluations`:** `id` (UUID PK), `session_id` (FK), `qa_index` (UNIQUE with `session_id`), `combined_risk_score`, `payload_json` (JSONB).
* **`integrity_reports`:** `session_id` (UUID PK/FK), `overall_risk_score`, `risk_band`, `report_json` (JSONB), `updated_at`.

### 4.3 Redis Keyspace Architecture
Every Redis key must carry an explicit TTL (24 hours for active session state) to prevent memory creep on free Upstash tiers:
* **Streams:** `stream:audio`, `stream:video`, `stream:utterances`, `stream:vision_events`, `stream:dlq`
* **Pub/Sub:** `channel:session:{session_id}` (JSON envelope `{event_type, timestamp_ms, data}`)
* **Stateful Buffers (TTL 24h):**
  * `stitch:{session_id}` (JSON of currently open utterance)
  * `vision:state:{session_id}` (Hash: `anomaly_start_ms`, `last_normal_ms`, `active_type`, `vlm_calls_count`, `last_vlm_ts_ms`)
  * `qa:buffer:{session_id}` (JSON of pending interviewer question + accumulated candidate answer turns)
  * `cb:{provider_name}` (Hash: `state`, `failures`, `opened_at_ms`)
  * `idem:{idempotency_key}` (String `"1"`, TTL 24h)

---

## 5. Module-by-Module Coding-Agent Implementation Blueprint

Implement each module strictly in the order listed below. Every module's responsibilities, interfaces, and exact mathematical/logical rules are specified below.

### Module 1: Configuration & Structured Observability (`config.py`, `telemetry.py`)
* **`Settings` (`pydantic-settings`):** Loads all env flags (`DEPLOY_MODE`, `INGESTION_PROVIDER`, `STT_PROVIDER`, `LLM_PRIMARY_PROVIDER`, `STORAGE_PROVIDER`), provider API keys (all optional strings defaulting to empty so the app boots with zero keys in offline/mock mode), and numeric thresholds (gaze angle limits, stitch gap, pre-filter score cutoff).
* **`telemetry.py`:** Configures JSON structured logging binding `session_id`, `worker_name`, `stage`, and `latency_ms`. Exposes an in-memory rolling metrics registry served at `GET /api/v1/metrics` reporting active sessions, Redis stream `XPENDING` lag, circuit-breaker states, and free-tier quota utilization percentages.

### Module 2: Event Bus & Ephemeral Storage (`db/redis_bus.py`, `db/storage.py`)
* **`RedisBus`:**
  * `publish_stream(stream, model, max_len)`: Serializes model via `model_dump_json()`, calls `XADD` with `approximate=True`.
  * `consume_stream(stream, group, consumer, model_cls, handler, dlq_max_retries=3)`: Creates group with `mkstream=True`. Loops using `XREADGROUP(..., block=5000, count=5)` (critical for Upstash free-tier command conservation). Every 60s calls `XAUTOCLAIM` for orphaned messages. Validates JSON, checks idempotency key, invokes `handler`, calls `XACK`. On exception, tracks retry count; if retries $\ge 3$, moves to `stream:dlq` and `XACK`s.
  * `broadcast_ui(session_id, event_type, payload)`: Publishes compact JSON to `channel:session:{session_id}`.
* **`StorageAdapter` ABC + Implementations:**
  * Methods: `put_bytes(key, data, content_type)`, `get_bytes(key)`, `delete(key)`, `get_presigned_url(key, expires_sec=3600)`.
  * `LocalDiskStorageAdapter`: Stores files under `./tmp_storage/`, serves presigned URLs via a local signed FastAPI static route `/api/v1/media/{key}`.
  * `CloudflareR2StorageAdapter`: S3-compatible client using `asyncio.to_thread` around `boto3` configured with `R2_ENDPOINT_URL`.

### Module 3: Pluggable Ingestion Plane (`adapters/ingestion/`)
All three adapters normalize incoming media into identical `AudioChunkEvent` and `VisionFrameEvent` messages:
* **`AttendeeBotAdapter` (`attendee.py`):**
  * `dispatch_bot(session_id, meeting_url, bot_name)`: Sends async HTTP POST to Attendee API with webhook URL and per-participant audio track request enabled. Stores returned `bot_external_id`.
  * `parse_webhook(headers, raw_body)`: Verifies HMAC-SHA256 signature. Maps Attendee lifecycle states (`joining`, `in_waiting_room`, `in_call_recording`, `call_ended`, `rejected`) to `SessionStatus`, updates Postgres, and broadcasts status to WebSocket UI. On media delivery callbacks, writes audio/frame bytes to `StorageAdapter` and pushes stream events (`is_mixed_audio=False` if participant ID is provided, else `True`).
* **`DemoFileAdapter` (`demo_replay.py` — $0 Demo Engine):**
  * Triggered by `POST /api/v1/demo/upload` (accepts `.mp4`, `.webm`, `.wav`, `.mp3`, max 150 MB).
  * Spawns async `ffmpeg` subprocesses (`asyncio.create_subprocess_exec`) in a background task:
    1. Extracts 16 kHz mono 16-bit WAV audio, segments into 15-second chunks (or 1 chunk if duration $< 90\text{ s}$ for rapid Speechmatics batch processing), writes to storage, and publishes `AudioChunkEvent(is_mixed_audio=True)`.
    2. If a video track exists, samples at 1 FPS resized to `640x360` JPEG (quality 75), writes to storage, and publishes `VisionFrameEvent(is_live=False)`.
* **`WebRTCIngestHandler` (`api/ws.py` `/ws/ingest/{session_id}` — $0 Live Demo Engine):**
  * Accepts binary or base64 framed WebSocket messages tagged `{"kind": "audio", "seq": n, "ts_ms": t}` (5-second WebM/Opus blobs from browser `MediaRecorder`) or `{"kind": "frame", "seq": n, "ts_ms": t}` (1 FPS canvas JPEGs). Transcodes WebM audio chunk to 16 kHz WAV via stdin/stdout piped `ffmpeg` subprocess, persists to storage, and emits to Redis Streams.

### Module 4: Domain Vocabulary Extractor & Hybrid STT Worker (`services/vocab_extractor.py`, `adapters/stt/`, `workers/stt_worker.py`)
* **`vocab_extractor.py`:** Given `job_title` and optional `job_description`, extracts up to 50 unique technical terms (capitalized acronyms, CamelCase/PascalCase tokens, hyphenated frameworks, plus a curated lookup table for software/data/cloud engineering terms like `Kafka`, `Kubernetes`, `gRPC`, `Redis`, `PyTorch`, `FastAPI`) and persists them in `sessions.custom_vocab`.
* **`STTRouter` (`adapters/stt/base.py`):**
  * Inspects `event.is_mixed_audio`, `Settings.STT_PROVIDER`, and `CircuitBreaker` / `QuotaGuard` states.
  * Routes mixed audio to **`SpeechmaticsAdapter`** (if API key configured and circuit closed); routes isolated per-participant tracks (or mixed fallback) to **`GroqWhisperAdapter`**.
* **`SpeechmaticsAdapter` (`speechmatics.py`):**
  * Submits audio chunk to Speechmatics Batch API (`operating_point="enhanced"`, `diarization="speaker"`, `additional_vocab` populated from `session.custom_vocab`). Uses async polling with exponential backoff (0.5s $\rightarrow$ 2s) or Speechmatics WebSocket RT client (`Linden 1`) for live sessions.
  * Aggregates contiguous word tokens with identical speaker labels (`S1`, `S2`) into `UtteranceEvent` list, adding `event.timestamp_ms` offset to every `WordTiming`.
* **`GroqWhisperAdapter` (`groq_whisper.py`):**
  * Checks `QuotaGuard` before calling Groq `/openai/v1/audio/transcriptions` with `model="whisper-large-v3-turbo"`, `response_format="verbose_json"`, `timestamp_granularities=["word"]`, and `prompt` seeded with comma-separated `custom_vocab`.
  * Maps word timestamps directly to `UtteranceEvent` using `event.speaker_hint`.
* **`UtteranceStitcher` (`services/stitcher.py`):**
  * Reads pending utterance from Redis `stitch:{session_id}`.
  * **Merge Condition:** If `incoming.speaker_label == pending.speaker_label` AND $(t_{\text{start, new}} - t_{\text{end, prev}}) \le 2500\text{ ms}$, concatenate text with single space, append `words`, and update `end_ms`.
  * **Flush Condition:** On speaker label change, gap $> 2500\text{ ms}$, or `END_OF_SESSION` sentinel: resolve `SpeakerRole` using `session.speaker_role_map` (auto-assign first speaker asking a question ending in `?` or first short prompt as `interviewer`, second speaker as `candidate`), write to Postgres `utterances`, publish to `stream:utterances`, and broadcast live caption via `broadcast_ui`.

### Module 5: 3-Tier Zero-GPU Vision Proctoring Worker (`workers/vision_worker.py`)
Processes `VisionFrameEvent` using a strict 3-tier compute funnel so average CPU cost stays under 15ms/frame:

* **Pre-Check (Backpressure Guard):** If `event.is_live` is True and $(\text{now\_ms} - \text{event.ingested\_at\_ms}) > 10000$, drop frame, delete from storage, and `XACK`.
* **Tier 0 — Static/Blur Gate ($< 2\text{ ms}$):** Decode JPEG via OpenCV. Compute grayscale Laplacian variance (reject completely occluded/black frames with variance $< 10$). Optionally skip full landmarking if mean absolute pixel difference from previous frame is $< 1.5$ (frozen camera).
* **Tier 1 — CPU MediaPipe Landmarker & Pose Geometry ($\approx 12\text{ ms}$):**
  * Run MediaPipe Face Landmarker (`num_faces=3`, `output_facial_transformation_matrixes=True`, CPU delegate) inside `asyncio.to_thread`.
  * **Multi-Face Rule:** If `len(faces) >= 2`, emit `MULTI_FACE` immediately and keep frame as evidence.
  * **3D Head Pose Extraction:** Decompose the primary face's $4 \times 4$ transformation matrix via `cv2.RQDecomp3x3` into Euler angles $(\text{pitch}, \text{yaw}, \text{roll})$.
  * **Normalized Iris Gaze Ratio:** Compute left and right iris center positions (landmarks `468`, `473`) normalized horizontally between inner/outer eye corners (`33`/`133` and `362`/`263`) to yield $g_x \in [-1, 1]$ (where $0$ is centered).
  * **Frame Anomaly Predicate:** A frame is flagged abnormal if:
    $$|\text{yaw}| > 22^\circ \quad \lor \quad \text{pitch} < -18^\circ \quad \lor \quad |g_x| > 0.35$$
  * **Dual-Threshold Hysteresis Debouncer (in Redis `vision:state:{session_id}`):**
    * Prevents noisy flickering from single blinks or brief thinking glances.
    * **Enter Anomaly State:** Only emit `HEAD_PITCH_DOWN` (if $\text{pitch} < -18^\circ$) or `GAZE_DEVIATION` once continuous abnormal duration reaches $\ge 3000\text{ ms}$.
    * **Exit Anomaly State:** Reset timer only after $\ge 1000\text{ ms}$ of continuous normal gaze.
* **Tier 2 — Budget-Capped VLM Verification (Gemini Flash):**
  * Executes **only** when Tier 1 fires a newly confirmed anomaly AND `vlm_calls_count < 12` AND $(\text{now\_ms} - \text{last\_vlm\_ts\_ms}) \ge 30000$ AND `GEMINI_API_KEY` is present and circuit is closed.
  * Sends the single JPEG to `GeminiJudgeAdapter.verify_frame()` demanding strict JSON output: `{phone_or_tablet_visible: bool, offscreen_reading_posture: bool, note: str}`.
  * If `phone_or_tablet_visible` is True, emit an additional `DEVICE_DETECTED` event and retain `evidence_frame_key`.
* **Storage Hygiene Rule:** After processing, **always delete** `event.storage_key` unless it was explicitly promoted to `evidence_frame_key` on a confirmed anomaly.

### Module 6: Adaptive Multimodal Scoring & LLM Judge Worker (`services/heuristics.py`, `adapters/llm/`, `workers/scoring_worker.py`)

#### 6.1 Q&A Turn Accumulator & Pre-Speech Lookback Join
* Consumes `UtteranceEvent` from `stream:utterances`. Buffers interviewer utterances as the active `question_text` and tracks `question_end_ms`.
* Accumulates subsequent candidate utterances until the interviewer speaks again (or session ends). Ignores trivial backchannels (`word_count < 15`, e.g., `"Sure, let me think"`).
* **Multimodal Temporal Join (with Pre-Speech Lookback):** Queries Postgres `vision_events` within the window:
  $$[t_{\text{question\_end}} - 500\text{ ms},\; t_{\text{answer\_end}}]$$
  *Why this matters:* Candidates using live AI tools glance down or sideways *during the 2–4 seconds right after the question ends* while waiting for AI tokens to stream, *before* speaking. Standard `[answer_start, answer_end]` joins miss this critical signal.

#### 6.2 Stage 0 — Robust Candidate Baseline Calibration (`qa_index == 0`)
On the first substantive candidate response (`word_count >= 40`, e.g., `"Tell me about yourself"`), compute and persist `CandidateBaseline` using **robust statistics** (median and Median Absolute Deviation, immune to single outlier pauses):
* Split the Q0 response into 10-second sub-windows to compute per-window WPM values $\{w_1, \dots, w_k\}$:
  $$\tilde{w} = \text{median}(w), \qquad \text{MAD}_w = \text{median}(|w_i - \tilde{w}|)$$
* Compute `baseline_disfluency_rate` as fillers per 100 words using a compiled regex bank (`um`, `uh`, `erm`, `like`, `you know`, `basically`, `actually`, `sort of`, `kind of`, `i mean`).
* Compute `baseline_pause_ratio` (total inter-word gaps $> 600\text{ ms}$ divided by speaking duration) and `baseline_ttfs_ms` (time from `question_end_ms` to first substantive candidate word).

#### 6.3 Stage 1 — Deterministic Heuristics Engine (`0ms`, Pure Python)
For every subsequent Q&A pair (`qa_index >= 1`), compute six orthogonal signals:
1. **Robust WPM Shift:**
   $$Z_{\text{wpm}} = \frac{0.6745 \cdot (\text{WPM}_{\text{curr}} - \tilde{w})}{\max(\text{MAD}_w,\, 18.0)}$$
2. **Relative Disfluency Drop:** $\Delta_{\text{disf}} = \text{baseline\_disfluency\_rate} - \text{curr\_disfluency\_rate}$. (Penalizes sudden disappearance of natural hesitation relative to the candidate's *own* baseline—protecting both fluent speakers and ESL speakers.)
3. **Time-to-First-Substantive-Word (TTFS) Anomaly:** Measures $t_{\text{answer\_start}} - t_{\text{question\_end}}$. Flags suspicious AI-generation buffer stalls when $\text{TTFS} \in [2500\text{ ms}, 7000\text{ ms}]$ paired with an unusually polished, zero-filler opening clause.
4. **Cadence Uniformity (Teleprompter Detector):** Computes the Coefficient of Variation ($\text{CV} = \sigma_{\text{gap}} / \mu_{\text{gap}}$) of inter-word pauses across the answer. Natural spontaneous speech is bursty ($\text{CV} > 0.8$); reading generated prose off a screen produces abnormally flat, metronomic pacing ($\text{CV} < 0.45$).
5. **Spoken LLM Tell Bank:** Case-insensitive regex matches for high-signal LLM verbal tics (`delve into`, `it's crucial to note`, `multifaceted`, `comprehensive approach`, `in summary`, `firstly ... secondly ... thirdly`, `foster synergy`, `in conclusion`).
6. **Self-Reference Density:** Ratio of first-person pronouns (`I`, `me`, `my`, `we`, `our`) to `word_count`. Near-zero self-reference ($< 0.015$) on long responses ($\ge 80$ words) signals textbook recitation.

**Deterministic Risk Score ($R_{\text{det}} \in [0, 100]$):**
Sum weighted components and clamp to $[0, 100]$:
* $+25$ if $\Delta_{\text{disf}} \ge 2.5$ and `word_count >= 80`
* $+15 \times \min(\text{len(ai\_tell\_matches)},\, 3)$
* $+15$ if $|Z_{\text{wpm}}| \ge 2.2$
* $+15$ if $\text{CV} < 0.45$ and `word_count >= 60`
* $+15 \times \min(\text{concurrent\_vision\_flags},\, 2)$
* $+10$ if $\text{TTFS} \ge 2500\text{ ms}$ and $\text{curr\_disfluency\_rate} < 0.5$
* $+10$ if `first_person_ratio < 0.015` and `word_count >= 80`

**Cost-Saving Pre-Filter Gate (`trigger_llm_judge`):**
Set `True` **only if** `word_count >= 50` AND at least one trigger fires:
* $R_{\text{det}} \ge 45.0$, OR
* `len(ai_tell_matches) >= 2`, OR
* `concurrent_vision_flags >= 1`, OR
* $\Delta_{\text{disf}} \ge 2.5$.

#### 6.4 Stage 2 — Structured LLM Judge & Failover Router (`adapters/llm/`)
* **Minimal Token Prompt:** Pass **only** the single Q&A pair, `job_title`, `CandidateBaseline`, and `HeuristicScores` (approx. 400–800 input tokens vs. 15,000+ for full transcripts).
* **Prompting Guardrails:** Instruct the judge to separate *live cognitive construction* (self-corrections, domain war stories, concrete trade-offs, informal phrasing) from *verbatim AI recitation* (question mirroring, symmetric 3-point lists, textbook definitions devoid of personal experience). Explicitly forbid penalizing grammar errors, non-native phrasing, or slow pacing. Require exact `verbatim_evidence_quotes` copied from `answer_text` and a tailored `advisory_follow_up_question` the interviewer can ask immediately to probe true depth.
* **Strict JSON Enforcement:** Use provider-native structured JSON output validated against `LLMJudgeVerdict`.
* **Failover Execution:** Try `GroqJudgeAdapter` (`qwen/qwen3-32b` or `llama-3.3-70b-versatile`) wrapped in `CircuitBreaker` $\rightarrow$ fail over to `GeminiJudgeAdapter` (`gemini-2.0-flash`) $\rightarrow$ fail gracefully to `None`.

#### 6.5 Score Aggregation (`services/aggregator.py`)
* **Per-Answer Combined Score:**
  $$R_{\text{qa}} = \begin{cases} 0.35 \cdot R_{\text{det}} + 0.65 \cdot R_{\text{llm}} & \text{if Stage 2 ran} \\ R_{\text{det}} & \text{if Stage 2 skipped or unavailable} \end{cases}$$
* **Overall Session Score:** Word-count-weighted average of $R_{\text{qa}}$ across all evaluated turns (`qa_index >= 1`), plus a capped vision integrity adjustment ($+15$ max if confirmed `DEVICE_DETECTED` or `MULTI_FACE` occurred), clamped to $[0, 100]$.
* **Advisory Risk Bands (Non-Binary):**
  * $[0, 35) \rightarrow$ `LOW`
  * $[35, 65) \rightarrow$ `MODERATE`
  * $[65, 100] \rightarrow$ `ELEVATED` (Never display punitive labels like "Cheater Detected".)

### Module 7: Stateless API Gateway & WebSockets (`api/sessions.py`, `api/webhooks.py`, `api/ws.py`)
Implement strict async endpoints with Pydantic request/response schemas:
* `POST /api/v1/sessions`: Creates session in Postgres, runs `vocab_extractor`, dispatches Attendee bot if `ingestion_mode == attendee_bot`, returns session object + WebSocket/upload URLs.
* `GET /api/v1/sessions`: Paginated list of sessions with current status and risk band.
* `GET /api/v1/sessions/{id}`: Full live state snapshot (for page reloads/reconnections).
* `PATCH /api/v1/sessions/{id}/speakers`: Swaps `speaker_role_map` (`S1` $\leftrightarrow$ `S2`) and triggers fast re-scoring of stored utterances without re-running STT.
* `POST /api/v1/demo/upload`: Multipart stream upload for zero-cost demo replay; validates MIME type and size, kicks off background `DemoFileAdapter`.
* `POST /api/v1/sessions/{id}/finalize`: Flushes open stitcher/QA buffers, generates final `IntegrityReport`, triggers asynchronous deletion of raw audio chunks from R2/disk.
* `POST /api/v1/webhooks/attendee`: HMAC-verified webhook receiver.
* `WS /ws/ingest/{id}`: Live browser WebRTC ingestion endpoint.
* `WS /ws/sessions/{id}`: Subscribes to Redis `channel:session:{id}`. On connect, sends initial hydration snapshot from Postgres so refreshes never lose history; then streams incremental events (`STATUS_UPDATE`, `UTTERANCE`, `VISION_ANOMALY`, `QA_EVALUATED`, `REPORT_READY`) with 25-second ping/pong heartbeats.
* `GET /api/v1/sessions/{id}/report`: Returns `IntegrityReport` with fresh presigned URLs for any `evidence_frame_key`s.
* `GET /healthz` & `GET /api/v1/metrics`: Liveness/readiness probes (verifies Postgres + Redis connectivity) and observability metrics.

---

## 6. Frontend Architecture & Zero-Cost UX Contract (`frontend/`)

Built with **Next.js 16 (App Router, React 19)** and **Tailwind CSS v4**, deployed free on Vercel Hobby.

1. **Session Launcher (`/sessions/new`):**
   * Three clean mode tabs: **Demo Recording Upload ($0)**, **Live Browser Tab Capture ($0)**, and **Meeting Bot Dispatch (Attendee)**.
   * Inputs: Candidate Name, Interviewer Name, Job Title, Optional Job Description (auto-populates Speechmatics technical dictionary).
2. **Live Sentinel Console (`/sessions/[id]/live`):**
   * **Browser Tab Capture Engine (when in `browser_webrtc` mode):** Uses `navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })`. Runs a `MediaRecorder` emitting 5-second audio blobs and an offscreen `<canvas>` drawing 1 FPS `640x360` JPEGs, both streamed over `WS /ws/ingest/{id}`.
   * **Live Telemetry Panels (driven by `WS /ws/sessions/{id}` with auto-reconnect + exponential backoff):**
     * Session status pill (`WAITING_IN_LOBBY`, `IN_CALL`, `HOST_DENIED`) + 1-click **Swap Speaker Roles** button.
     * Rolling **Advisory Risk Gauge** (`LOW` / `MODERATE` / `ELEVATED`) + Q0 Baseline calibration badge.
     * Live diarized transcript with inline vision anomaly chips (`Gaze Deviation 4.2s`, `Head Pitch Down 3.8s`).
     * **Interviewer Co-Pilot Card:** Surfaces the LLM Judge's generated `advisory_follow_up_question` in real time so the human interviewer can probe suspiciously polished answers naturally during the call.
3. **Executive Integrity Report (`/sessions/[id]/report`):**
   * Prominent **Advisory Disclaimer Banner** explaining behavioral inference boundaries.
   * Q0 Baseline vs. Interview Metrics comparison grid (WPM, Disfluency Delta, TTFS Latency, Vision Events).
   * Expandable Q&A Evidence Cards showing verbatim candidate quotes, matched LLM tells, heuristic breakdown, and LLM reasoning signals.
   * Interactive Visual Anomaly Timeline with click-to-inspect evidence keyframes (only confirmed anomalies).
   * **Client-Side PDF Export:** Implemented via `@react-pdf/renderer` directly in the browser—zero server CPU, zero backend dependencies.

---

## 7. Security, Privacy-by-Design & Compliance Guardrails

1. **Ephemeral Media Purge:** Non-anomalous video frames live for $< 5\text{ s}$ before deletion. Raw audio chunks are deleted automatically upon `POST /finalize` (plus a 24-hour R2 bucket lifecycle expiration rule as a safety net).
2. **Webhook & Upload Security:** Verify Attendee HMAC signatures in constant time (`hmac.compare_digest`). Enforce strict upload size caps (150 MB) and FFmpeg execution timeouts to prevent DoS on free containers.
3. **Bias & Legal Defensibility:** Never output binary cheating verdicts. Every report preserves the Q0 baseline metrics and raw verbatim quotes so human hiring committees retain full decision authority.
4. **Zero-Secret Repo Policy:** All API keys load via `pydantic-settings`; startup logs mask every credential and warn cleanly which fallback adapters are active.

---

## 8. Strict Coding-Agent Build Sequence & Verification Gates

The coding agent must build and verify **one phase at a time**. Do not advance to the next phase until all verification gates pass.

| Phase | Files to Build | Mandatory Automated Verification Gate (`pytest` / Smoke Check) |
| :--- | :--- | :--- |
| **Phase 1: Core Contracts & Config** | `config.py`, `schemas/contracts.py`, `telemetry.py` | Unit tests verifying JSON round-trip serialization, score clamping $[0, 100]$, and booting cleanly with zero API keys set. |
| **Phase 2: Pure Logic & Math** | `services/vocab_extractor.py`, `services/stitcher.py`, `services/heuristics.py`, `services/aggregator.py` | Unit tests covering: (a) 2,500ms stitch merge vs. split; (b) Q0 MAD baseline calculation; (c) ESL fluent vs. sudden disfluency-drop scenarios; (d) Pre-filter gate skipping normal answers; (e) Pre-speech lookback window join. |
| **Phase 3: Infrastructure & Resilience** | `db/postgres.py`, `db/redis_bus.py`, `db/storage.py`, `CircuitBreaker`, `QuotaGuard` | Integration tests against local Docker Redis/Postgres verifying `XREADGROUP` + `XACK`, idempotency deduplication, DLQ routing after 3 failures, and `CLOSED -> OPEN -> HALF_OPEN` circuit transitions. |
| **Phase 4: Provider Adapters** | `adapters/ingestion/*`, `adapters/stt/*`, `adapters/llm/*` | Mocked `httpx` unit tests verifying Speechmatics custom vocab payload, Groq Whisper routing on isolated tracks, HTTP 429 automatic failover from Groq LLM to Gemini Flash, and Attendee HMAC verification. |
| **Phase 5: Async Workers** | `workers/stt_worker.py`, `workers/vision_worker.py`, `workers/scoring_worker.py`, `workers/runner.py` | Synthetic end-to-end stream test pushing 1 mock WAV + 5 sample JPEGs into Redis Streams and verifying `IntegrityReport` creation in Postgres + cleanup of non-anomalous frames. |
| **Phase 6: API Gateway & Docker** | `api/sessions.py`, `api/webhooks.py`, `api/ws.py`, `main.py`, `docker-compose.yml` | Run `docker compose up --build`; verify `GET /healthz`, upload a 60-second sample `.mp4` via `/api/v1/demo/upload`, observe live WebSocket events, and fetch `/report`. |
| **Phase 7: Next.js 16 Dashboard** | `frontend/src/app/sessions/*` + `@react-pdf/renderer` report view | Full browser demo: upload mock interview OR capture a live tab, verify live transcript/risk updates, swap speaker roles, and download a formatted PDF report at **$0 cost**. |

---

## 9. System Design Interview Defense Cheat Sheet

When presenting **OCULUS** in a senior/staff system design interview, lead with these five high-leverage trade-offs:

1. **"Why Redis Streams + Dual-Mode Runtime instead of Kafka/Celery?"**  
   Redis Streams gives consumer groups, crash recovery (`XAUTOCLAIM`), backpressure (`MAXLEN`), and WebSocket Pub/Sub in a single zero-cost primitive. Combined with `DEPLOY_MODE=single_node|distributed`, the exact same codebase runs on a free 512 MB container for demos and scales horizontally across pods without changing a single line of business logic.
2. **"How do you process video at $0 without GPU servers?"**  
   A **3-Tier Vision Funnel**: frame-diff/blur gating (Tier 0) $\rightarrow$ 12ms CPU MediaPipe pose + iris geometry with dual-threshold hysteresis debouncing (Tier 1) $\rightarrow$ budget-capped Gemini Flash VLM verification only on sustained $\ge 3\text{ s}$ anomalies (Tier 2). This cuts compute cost by $> 99\%$ while catching secondary devices.
3. **"Why hybrid Speechmatics + Groq instead of a single STT engine?"**  
   Diarization and custom domain vocabularies are the biggest failure modes on mixed room audio—Speechmatics handles both natively (and covers hundreds of demo hours on free credit). When Attendee supplies isolated per-speaker tracks, diarization is redundant, so the router shifts traffic to Groq's free Whisper v3 Turbo tier (8 hrs/day free).
4. **"How do you prevent false positives on ESL and neurodivergent candidates?"**  
   Never score against universal constants. Stage 0 calibrates every candidate against their own Q0 introduction using **robust median/MAD statistics** (resistant to outlier pauses), combines a **pre-speech lookback window** with **cadence uniformity**, and outputs strictly advisory non-binary bands plus **live follow-up probe questions** for the human interviewer.
5. **"How do you guarantee resilience on rate-limited free tiers?"**  
   Four layers working together: (a) blocking Redis reads (`BLOCK 5000`) to preserve Upstash command quotas, (b) deterministic Stage 1 pre-filtering cutting LLM volume by 75%, (c) Redis token-bucket `QuotaGuard`, and (d) a 3-state `CircuitBreaker` cascading automatically from Groq $\rightarrow$ Gemini Flash $\rightarrow$ deterministic degradation so an interview never drops.