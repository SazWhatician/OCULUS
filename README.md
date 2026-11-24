<div align="center">

<img src="assets/oculus-brand-banner.svg" alt="OCULUS Brand Banner" width="100%" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.11+-000000.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-000000.svg?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-000000.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-Streams-000000.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-000000.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Groq LLaMA 3.3](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-000000.svg?style=for-the-badge)](https://groq.com)
[![Google Gemini](https://img.shields.io/badge/Fallback-Gemini%202.5%20Flash-000000.svg?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-000000.svg?style=for-the-badge)](LICENSE)

<br/>

**Multimodal Real-Time Forensic Proctoring Engine & Acoustic Cadence Analyzer**

*Detecting teleprompter reads, conversational AI copilots, and synthetic recitation with microsecond telemetry.*

[Architecture](#system-architecture) • [Core Innovations](#core-innovations) • [Detection Matrix](#detection-telemetry-matrix) • [Quickstart](#quickstart--deployment) • [API Contracts](#api--streaming-specifications) • [Frontend UX](#optical-brutalist-interface)

</div>

---

## Executive Overview

**OCULUS** is an industrial-grade, zero-trust multimodal proctoring and forensic reasoning engine designed to expose generative AI cheating during technical interviews and remote assessments.

Conventional proctoring tools rely on crude facial absence or generic object detection—blind to the reality of modern interview fraud:
1. **Audio/Screen AI Copilots:** Real-time generation of technical solutions delivered via hidden earpieces or dual-screen displays.
2. **Teleprompter Recitation:** Mechanical verbal delivery of generated text characterized by abnormal pause uniformity and suppressed natural disfluencies (`um`, `uh`, false starts).
3. **Teleprompter Gaze Saccades:** Micro-horizontal rapid eye sweeps across second monitors synchronized with sudden spikes in verbal technical complexity.
4. **Session Score Dilution:** Exploiting standard average-risk scoring by chattering innocently for 40 minutes after cheating for 5 minutes.

OCULUS resolves these vulnerabilities by fusing **sub-50ms deterministic acoustic heuristics**, **computer vision saccadic tracking**, a **signal-aware gated dual-LLM judge**, and **anti-dilution Bayesian aggregation**.

---

## System Architecture

The OCULUS pipeline executes across four decoupled, resilient computational stages linked by Redis Streams and async workers.

<div align="center">
  <img src="assets/architecture-diagram.svg" alt="OCULUS System Architecture Diagram" width="100%" />
</div>

### Stage 0: 60-Second Cold-Start Baseline Calibration
Cheating detection fails if you measure all candidates against a generic population mean. Fast native speakers speak quickly without cheating; ESL speakers naturally pause more frequently.
* During the first 60 seconds of unscripted conversation ("Tell me about your background"), OCULUS captures the candidate's physiological baseline:
  * Words Per Minute ($\mu_{wpm}, \sigma_{wpm}$)
  * Natural Disfluency Rate ($f_{disfluency}$ per 100 words)
  * Inter-word pause duration distribution and standard deviation ($\sigma_{pause}$)
  * Pitch variability and vocal jitter ($F_0$)
* **Empirical Bayesian Prior Shrinkage:** The cold-start estimate is shrunk toward population priors to eliminate small-sample noise before scoring begins.

### Stage 1: Deterministic Multimodal Forensics (<50ms)
Incoming audio and video frames are processed asynchronously in rolling 10-second windows:
* **Acoustic Prosody & Silence CV:** Calculates the Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) of inter-word silence intervals. Human spontaneous speech exhibits irregular pauses ($CV \in [0.45, 0.85]$). A candidate reading streaming AI output exhibits unnatural temporal regularity ($CV < 0.25$).
* **Disfluency Suppression Index:** Sudden collapse of natural hesitations (`um`, `like`, `you know`) during complex algorithmic explanations triggers deterministic escalation.
* **Teleprompter Saccade Velocity:** Mediapipe iris landmarks compute micro-saccadic horizontal sweeps. High-frequency horizontal panning with zero vertical head movement indicates reading from an off-axis monitor.
* **Deterministic Lexical Tell Detector:** Fast regex screening sweeps for high-probability AI phrases (`delve into`, `pivotal role`, `crucial to note`, `harness the power`, `ever-evolving landscape`).
* **Prompt Injection Quarantine:** Transcribed text is pre-scanned for spoken jailbreaks (`ignore all previous instructions`, `system prompt override`, `act as a proctor bypass`) before entering subsequent layers.

### Stage 2: Signal-Aware Gated Dual-LLM Judge
LLM evaluations are computationally expensive. OCULUS implements an intelligent triage gate:
* **Inconclusive Short-Answer Gate:** Trivial utterances (< 8 words like *"Yes, I can hear you"*) without security or acoustic flags return `None` (inconclusive), preventing false positives and zero-score cache pollution.
* **Escalation Trigger:** An utterance is dispatched to the LLM Judge if and only if:
  1. Spoken prompt injection detected.
  2. High-precision AI tell regex match.
  3. Deterministic heuristic risk exceeds 35.0.
  4. Concurrent vision proctoring flags > 0.
* **Provider Chain & Circuit Breakers:**
  * **Primary:** Groq LLaMA-3.3-70B Versatile (~380 tokens/sec, low latency).
  * **Secondary:** Google Gemini 2.5 Flash via circuit breaker fallback (`cb_groq` trips on timeouts or HTTP 429).
  * **Offline/Dev:** Zero-downtime mock engine simulating calibrated verdicts for local development without live API keys.
* **Zero-Trust Verbatim Quote Verification:** LLM judges can hallucinate evidence. OCULUS cross-references all `verbatim_evidence_quotes` against raw STT transcripts. Any quote not strictly existing in the transcript is discarded; verdicts with invalid evidence undergo automatic confidence-band demotion.

### Stage 3: Anti-Dilution Top-K Session Risk Aggregator
A common exploit in automated proctoring is **score dilution**: a candidate cheats for 2 minutes to solve the primary algorithm, then speaks casually for 30 minutes, lowering the arithmetic mean score below alert thresholds.
* OCULUS employs **Top-K Weighted Pooling**:
  $$\text{FinalScore} = \alpha \cdot \frac{1}{K}\sum_{i=1}^K s_{(i)} + (1 - \alpha) \cdot \frac{1}{N}\sum_{j=1}^N s_j$$
  *(where $s_{(i)}$ are the highest suspicion turns, $K=3$, $\alpha=0.65$)*.
* A single flagrant AI recitation permanently influences the session verdict, resisting dilution attacks.

---

## Detection Telemetry Matrix

| Telemetry Vector | Modality | Detection Metric | Cheating Signature | Normal Spontaneous Speech |
| :--- | :--- | :--- | :--- | :--- |
| **Silence Cadence Uniformity** | Audio / STT | Pause Duration Coefficient of Variation ($CV_{pause}$) | $CV < 0.25$ (Metronomic reading cadence) | $CV \ge 0.50$ (Bursts of words, variable thinking pauses) |
| **Disfluency Density** | Audio / Text | Disfluencies per 100 words ($D_{rate}$) | $D_{rate} \to 0$ on difficult topics | Natural occurrence of `uh`, `um`, self-corrections ($D_{rate} \ge 2.5$) |
| **Gaze Micro-Saccades** | Video / Iris | Horizontal velocity & saccade frequency | Repeated horizontal oscillations at screen edge | Focused eye contact with conversational wandering |
| **Lexical AI Tell Signature** | Text | Multitier structural regex matching | $\ge 2$ structured synthetic phrasing markers | Natural, colloquial, domain-practical phrasing |
| **Prompt Injection Defense** | Text | Spoken jailbreak patterns | Spoken override strings targeting LLM judge | Clean technical domain discourse |
| **Response Latency Disconnect** | Multimodal | Time from prompt to fluent technical monologue | Immediate (< 1.5s) unbroken articulation on hard problems | Cognitive processing delay ($2.0s - 6.0s$), vocalized contemplation |

---

## Technical Stack

```
OCULUS Engine
├── Frontend (Next.js 14 App Router)
│   ├── Typography: Geist Mono & Space Grotesk
│   ├── Styling: TailwindCSS with Optical Brutalist Design System
│   ├── Audio Capture: Web Audio API (16kHz PCM Float32 Processor)
│   ├── Vision: HTML5 Canvas Video Stream + Mediapipe Iris Tracker
│   └── Networking: Native WebSocket Audio/Telemetry Pipeline
│
├── Backend (FastAPI / Python 3.11+)
│   ├── API Gateway: FastAPI Asynchronous Endpoints & WebSockets
│   ├── Event Broker: Redis Streams (Consumer Groups for Scoring Workers)
│   ├── Speech-to-Text: Groq Whisper-Large-v3 / Speechmatics Batch STT
│   ├── Deterministic Engine: NumPy, SciPy (Signal Processing), Regex Toolkit
│   ├── LLM Judicial Layer: Groq LLaMA-3.3-70B + Google Gemini 2.5 Flash
│   ├── Storage & ORM: SQLAlchemy 2.0 Async + PostgreSQL 16 / SQLite
│   └── Resilience: Tenacity Exponential Backoff + PyBreaker Circuit Breaking
```

---

## Optical Brutalist Interface

The OCULUS frontend is constructed on the principles of **Laboratory-Grade Precision Instrumentation**:
- **Monochrome High-Contrast Reticle Visuals:** Stark `#000000` / `#FFFFFF` high-contrast aesthetic with 1px technical borders, hairline division grids, and corner reticle marks.
- **Dynamic Color Inverter:** Dedicated mode switch providing instant polarity inversion (Obsidian Instrument vs. High-Key Laboratory Paper) with persistent state.
- **Micro-Telemetry Gauges:** Live SVG aperture gauges, real-time audio waveform deformation, pause CV meters, and streaming confidence bands.
- **Forensic Audit Ledger:** Drilldown session inspection view with sentence-by-sentence timeline badges, verbatim quotation highlights, and provider attribution badges.

---

## Repository Structure

```
OCULUS/
├── assets/
│   ├── oculus-brand-banner.svg         # Dark/Light dual-polarity brand masthead
│   └── architecture-diagram.svg        # Complete 4-stage pipeline topology
├── backend/
│   ├── app/
│   │   ├── adapters/
│   │   │   ├── llm/
│   │   │   │   ├── base.py             # Router, calibration, regex toolkit & contracts
│   │   │   │   ├── groq_judge.py       # Groq LLaMA-3.3-70B adapter with circuit breaker
│   │   │   │   └── gemini_judge.py     # Gemini 2.5 Flash fallback adapter
│   │   │   └── stt/
│   │   │       ├── base.py             # Speech-to-text contract definition
│   │   │       └── groq_whisper.py     # Fast Whisper audio chunk transcription
│   │   ├── api/                        # Session, audio, and report REST/WS routes
│   │   ├── core/                       # Database engine, session lifecycle, exceptions
│   │   ├── models/                     # SQLAlchemy 2.0 async ORM schemas
│   │   ├── services/
│   │   │   ├── heuristics.py           # Acoustic cadence, disfluency & pause CV analysis
│   │   │   ├── vision_proctoring.py    # Gaze saccades, multi-face & iris tracking
│   │   │   └── proctoring_service.py   # Turn aggregation and Bayesian state update
│   │   └── workers/
│   │       └── scoring_worker.py       # Async Redis stream consumer & batch processor
│   ├── tests/                          # Pytest suite (adapters, heuristics, API, workers)
│   ├── test_env.py                     # Provider & database connectivity verification
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                        # Next.js App Router (pages: /, /sessions, /reports)
│   │   └── components/                 # Brutalist UI components, optics canvas, audio meter
│   ├── tailwind.config.ts
│   └── package.json
├── docker-compose.yml                  # Full-stack orchestration (API, worker, redis, postgres)
└── README.md
```

---

## Quickstart & Deployment

### 1. Clone & Configure Environment

```bash
git clone https://github.com/SazWhatician/OCULUS.git
cd OCULUS
```

Configure backend environment variables:

```bash
cp backend/.env.example backend/.env
```

Ensure `.env` contains your operational keys:
```ini
# LLM Providers
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...

# Optional: Dedicated Speechmatics Streaming Key
SPEECHMATICS_API_KEY=...

# Database & Cache
DATABASE_URL=postgresql+asyncpg://oculus:oculus_password@localhost:5432/oculus_db
REDIS_URL=redis://localhost:6379/0

# App Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```
*(Note: If Postgres or Redis are not running locally, OCULUS automatically falls back to an embedded SQLite database and in-memory simulated streams for zero-friction local testing).*

---

### 2. Docker Compose Deployment (Recommended)

Boot the entire ecosystem with a single command:

```bash
docker compose up --build -d
```

Services exposed:
- **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation & Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redis Event Stream:** `localhost:6379`
- **PostgreSQL 16:** `localhost:5432`

---

### 3. Manual Local Development

#### Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# Run diagnostic pre-flight check
python test_env.py

# Launch FastAPI Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a separate terminal, launch the background scoring worker:
```bash
python -m app.workers.scoring_worker
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## API & Streaming Specifications

### Real-Time Audio WebSocket
`WS /api/v1/sessions/{session_id}/audio-stream`

Accepts raw 16kHz binary PCM audio chunks. Transcribes and evaluates cadence in rolling 10-second segments.

### Real-Time Vision Telemetry
`WS /api/v1/sessions/{session_id}/vision-telemetry`

**Payload:**
```json
{
  "timestamp_ms": 1718000000000,
  "face_count": 1,
  "head_pose": { "pitch": -2.1, "yaw": 14.8, "roll": 0.5 },
  "iris_gaze": { "gaze_x": -0.42, "gaze_y": 0.08 },
  "saccade_event": true
}
```

### Forensic Session Report
`GET /api/v1/reports/{session_id}`

**Response:**
```json
{
  "session_id": "8fa21e64-50a1-432a-89f5-199b1d5c22e4",
  "candidate_name": "Senior Candidate #402",
  "session_duration_seconds": 1840,
  "overall_verdict": "SUSPICIOUS_AI_ASSISTED",
  "composite_suspicion_score": 84.6,
  "top_k_peak_score": 91.2,
  "baseline_metrics": {
    "speaking_wpm": 148.2,
    "disfluency_rate_pct": 3.8,
    "silence_cv": 0.62
  },
  "turn_evaluations": [
    {
      "turn_id": 14,
      "transcript": "It is crucial to note that one must delve into composite indexing when optimizing join constraints.",
      "heuristic_score": 72.0,
      "llm_judge_score": 88.0,
      "provider_used": "groq/llama-3.3-70b-versatile",
      "flagged_tells": ["crucial to note", "delve into"],
      "verbatim_evidence": ["crucial to note", "delve into"],
      "silence_cv": 0.18,
      "saccade_flag": true
    }
  ]
}
```

---

## Test Suite Execution

Run the complete test suite across adapters, heuristics, API contracts, and async workers:

```bash
cd backend
python -m pytest tests/ -v
```

To run individual adapter resilience tests:
```bash
python -m pytest tests/test_adapters.py -v
```

---

## Security & Ethics Statement

* **Privacy by Design:** Raw video frames are processed locally inside client WebAssembly / Mediapipe memory and never persisted to disk. Only numeric landmark coordinates and saccadic frequencies are transmitted.
* **Fairness & Neurodiversity:** Baseline calibration prevents penalizing non-native speakers or neurodivergent candidates with atypical eye contact. Cheating flags require **multimodal concordance** (concurrent acoustic silence uniformity, lexical tells, and prompt anomalies) rather than isolated gaze metrics.
* **Auditability:** Every judgment preserves exact verbatim textual evidence, heuristic telemetry, and raw model reasoning outputs for human review.

---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
