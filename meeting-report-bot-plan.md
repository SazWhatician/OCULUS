# Meeting Report Bot — Engineering Plan

**Working title:** Proctor Bot / Interview Integrity Reporter
**Scope:** A bot that joins a Google Meet or Zoom interview, observes the session, and produces a 2–3 page downloadable integrity report with timestamped observations.

---

## 0. Read this first: the one architectural constraint that shapes everything

**A bot that joins a meeting as a participant cannot see the candidate's screen, processes, or browser tabs.** It only receives what the meeting platform transmits: audio, video tiles, screenshare (if shared), chat, and roster events.

Requirement #5 (detect screen switching / AI tool usage) is therefore **not solvable inside the meeting-bot plane**. It requires a second, candidate-side data source. Everything below is designed around that split. If the product ships without a candidate-side component, requirement #5 must be downgraded from "detection" to "weak behavioral inference," and the report must say so.

A second unavoidable truth: **a phone in the candidate's lap or a second laptop off-camera defeats every telemetry channel.** The only signal against that is gaze and head-pose analysis, which is circumstantial. Design the product to be honest about this rather than to imply coverage it does not have.

---

## 1. System architecture

```
┌─ PLANE A: MEETING INGESTION ────────────────────────────────────┐
│  Bot joins Zoom / Google Meet                                   │
│  → per-participant audio streams                                │
│  → per-participant video tiles                                  │
│  → roster events (join, leave, mute, camera on/off, share)      │
│  → screenshare stream, chat                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─ PLANE B: CANDIDATE ENDPOINT (optional but required for #5) ────┐
│  Web interview client and/or desktop agent                      │
│  → focus/blur, visibility, fullscreen exit, paste events        │
│  → (agent only) foreground window, display count, VM flags      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─ PLANE C: PROCESSING ───────────────────────────────────────────┐
│  ASR + timestamping   │  Vision workers   │  Transcript agent   │
│  Speaker attribution  │  gaze / pose      │  linguistic markers │
│  Overlap detection    │  objects / faces  │  acoustic markers   │
│                       │  liveness         │  intra-session drift│
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─ PLANE D: EVENT STORE + FUSION ─────────────────────────────────┐
│  Normalized event stream on one session clock                   │
│  Evidence artifacts (frames, clips, transcript spans) in S3     │
│  Confidence scoring, deduplication, severity assignment         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─ PLANE E: REPORT ───────────────────────────────────────────────┐
│  Structured JSON → deterministic template → 2–3 page PDF        │
│  Signed download URL, expiring, access-audited                  │
└─────────────────────────────────────────────────────────────────┘
```

**Session clock.** Every event from every plane carries `t_offset_ms` relative to a single session start timestamp, plus wall-clock UTC. Endpoint agent clocks are skew-corrected against the server on connect and re-synced every 60s. Without this, the timeline in the report is worthless.

---

## 2. Plane A — Meeting ingestion and bot lifecycle

**Build vs. buy.** Start with a meeting-bot vendor (Recall.ai and similar provide a unified API across Zoom / Meet / Teams). Move to native SDKs — Zoom Meeting SDK with real-time media streams, Google's Meet Media API — only when per-interview cost or media-access control justifies the maintenance burden. Verify current API capabilities before committing; both platforms change media-access terms regularly, and raw per-participant media access is often gated behind app review.

**The capability that matters most: per-participant audio streams.** If the platform gives separate audio tracks per participant, the overlapping-speech problem largely disappears (see §4). If you only get a mixed downmix, you inherit a diarization problem. Confirm this before locking in a platform path.

**Lifecycle state machine:**

`SCHEDULED → JOINING → WAITING_ROOM → ADMITTED → ANNOUNCING → RECORDING → (DEGRADED) → LEAVING → PROCESSING → REPORT_READY`

- **Announcing** is not optional. The bot posts a chat message and, where supported, uses a display name that identifies it as a recording proctor. Several jurisdictions require all-party notice for recording.
- **Degraded** is a first-class state, not an error. Media drops, a participant's camera dies, screenshare stops. The bot keeps recording what it can and logs a coverage gap.
- Reconnect with exponential backoff. Log every gap: `coverage_gap` events roll up into the report's monitoring-coverage percentage.
- Auto-leave when the host leaves or after N minutes of an empty room. Hard cap on session duration.

---

## 3. Identity resolution — interviewer vs. candidate

Use a precedence ladder. Never let a weak signal override a strong one, and never guess silently.

**Tier 1 — Pre-registration binding (authoritative).**
The ATS or scheduling system creates the interview record with candidate name, email, and a unique join token. The candidate receives a personal magic link; when they join, their meeting participant ID binds to `role=candidate`. Interviewers join via corporate SSO or a recognized email domain and bind to `role=interviewer`. This resolves 90%+ of sessions with zero ambiguity.

**Tier 2 — Meeting metadata.**
Display name, email address and domain, host/co-host flags, join order. Fuzzy-match display name against the expected candidate name. Good for confirmation, unreliable alone — display names are user-editable.

**Tier 3 — Biometric confirmation (opt-in, consent-gated).**
At check-in the candidate presents a government ID and a live selfie. Compute a face embedding (ArcFace/InsightFace) from both, verify match, and store the embedding as the session's enrollment reference. Optionally enroll a voice embedding from a scripted 10-second read.

The real value is **continuous verification**, not the one-time check: sample the candidate's video tile every few seconds and compare against the enrollment embedding. This is what catches a person-swap mid-interview, which a login check cannot.

**Tier 4 — Transcript-derived heuristics (flagging only, never authoritative).**
Question-to-answer ratio, turn-length distribution, interrogative density. Useful for one thing: detecting that Tier 1/2 assignments look wrong (e.g., the "candidate" is asking all the questions). When Tier 4 disagrees with Tier 1, do not reassign — raise an `identity_inconsistency` flag for human review.

**Conflict rule:** if tiers disagree beyond a threshold, mark the session `identity_unresolved` and surface it prominently in the report. An integrity report built on a misattributed speaker is worse than no report.

---

## 4. Caveat: simultaneous speech

**Primary solution — separate at the source.** With per-participant audio tracks, run an independent ASR stream per participant. Attribution is structural, not inferred. Merge into a single timeline by timestamp. Crosstalk becomes two correctly-attributed overlapping segments instead of one garbled one.

**Fallback for mixed audio.** Run diarization (pyannote.audio or equivalent) plus a dedicated overlapped-speech detector. Every segment gets `speaker_id`, `overlap: bool`, and `attribution_confidence`.

**Handling rules regardless of path:**

- **Acoustic bleed.** The candidate's speakers may re-emit the interviewer's voice into the candidate's mic, making it look like the candidate said it. Detect via cross-correlation between streams with a small lag window, and via energy ratio. Suppress bleed segments before attribution.
- **Confidence downgrade rule.** Any integrity flag whose evidence sits inside an overlap window is capped at `confidence ≤ 0.5` and marked `requires_human_review`. This is the single most important rule for avoiding false accusations from garbled audio.
- **Barge-in is normal.** Interruptions, backchannels ("mm-hm," "right"), and simultaneous starts are ordinary conversation. They are not integrity signals and must never be scored as such. Only the *transcript quality degradation* matters, and it is handled by the confidence cap.
- **Report surfacing.** Overlap-heavy stretches appear as transcript-reliability notes, not as findings.

---

## 5. Caveat: camera turned off mid-conversation

Treat this as a **coverage state change**, not as evidence of wrongdoing. Conflating the two is the most common design failure in proctoring tools.

**Detection:** roster events give explicit `video_off` / `video_on`. Corroborate with the media track itself — some clients report video-on while sending a black or frozen frame.

**Classification (report the class, not a verdict):**

| Class | Signals |
|---|---|
| Deliberate toggle | Clean roster event, audio continues uninterrupted, connection quality stable |
| Network degradation | Packet loss / bitrate collapse precedes it, audio also degrades, possible rejoin |
| Client crash / rejoin | Participant leave + rejoin within a short window |
| Spoofed feed | Video "on" but frames are static, looping, or fail liveness checks |

**Liveness checks (run continuously, not just on camera-off):** frame-difference entropy, blink detection, micro-motion in facial landmarks, and periodic novelty checks against earlier frames to catch a looped clip. A virtual-camera feed is a more serious finding than a camera simply being off, because it implies intent.

**Behavior:**
1. Log `camera_off` with start, end, duration, and class.
2. Suspend all vision-derived checks for that window. Do not infer.
3. Optional configurable actions: bot posts a chat reminder, notifies the interviewer, or stays silent — per organization policy.
4. The report states coverage explicitly: *"Visual monitoring unavailable 00:14:22–00:19:40 (5m 18s) — deliberate camera toggle."* No inference, no score penalty by default.
5. Roll all gaps into a single **Monitoring Coverage** percentage on page 1, so the reader knows how much of the session was actually observed.

---

## 6. Vision pipeline — gaze, head pose, objects, multiple faces

**Sampling.** 2–5 fps from the candidate's tile is sufficient and roughly an order of magnitude cheaper than full frame rate. Burst to higher rates for a few seconds around a candidate detection to capture better evidence frames.

**Stages:**

1. **Face detection + landmarks** — MediaPipe Face Mesh or RetinaFace. Cheap, runs first, gates everything else.
2. **Head pose** — solvePnP on 3D landmark correspondences → yaw / pitch / roll.
3. **Gaze** — iris landmarks or a dedicated gaze model (L2CS-Net class). Combine with head pose; neither alone is reliable.
4. **Object detection** — YOLO-family model. COCO already covers phone, book, laptop, TV, person. Fine-tune on a domain set for: phone held near face vs. lying on desk, printed notes, earbuds/earpieces, second monitor, second person.
5. **Face count + verification** — number of distinct faces, and identity check of each against the enrollment embedding.

**Calibration is mandatory.** Spend the first 30 seconds of small talk establishing the candidate's neutral head pose and gaze centroid. Monitor position, chair height, and multi-monitor setups vary enormously; uncalibrated thresholds produce unusable false-positive rates. Store the baseline per session.

**Temporal and contextual gating — this is where accuracy is won or lost:**

- **Sustain thresholds.** A glance is not a finding. Require deviation beyond threshold for a continuous window (e.g., ≥ 3s for gaze, ≥ 5s for an object) before emitting an event.
- **Hysteresis.** Different enter/exit thresholds to prevent event flapping.
- **Speech-state correlation.** Looking away while formulating an answer is normal human behavior. Looking away *while speaking fluently* is a different pattern. Gate gaze events on the candidate's speaking state and on time-since-question.
- **Reading detection.** Fixation on a fixed off-screen region with horizontal saccadic sweeps, correlated with fluent uninterrupted speech, is a meaningfully stronger signal than "looked away." Build this as a compound rule.
- **Compound object rules.** "Phone visible on desk" is near-worthless. "Phone in hand, raised, gaze directed at it, sustained ≥ 4s" is worth reporting. Encode the compound conditions explicitly; do not emit raw detector output.
- **Second face.** Require sustained presence, not a single frame — a family member crossing the room is not an integrity event. Verify whether the *candidate's* face is still the enrolled identity.

**Known failure modes to document and tune against:** low light; strong backlight; glasses and reflections; darker skin tones under poor illumination (several open gaze/face models degrade measurably — measure this explicitly, do not assume parity); virtual backgrounds clipping objects; bookshelves triggering "book"; assistive devices resembling prohibited objects.

---

## 7. Screen switching and AI-tool detection

This is the hardest requirement and the one most likely to be oversold. Four options, in descending strength:

### Option A — Mandatory full-screen share (recommended default)
The candidate shares their entire screen for the session. The bot receives the screenshare track.

- **Signals:** OCR of the browser URL bar and tab titles; a lightweight visual classifier trained on known AI-tool UIs; detection of window switching via frame-difference and layout change; `screenshare_stopped` as its own high-severity event.
- **Strengths:** no install, works in both Zoom and Meet, gives actual content evidence.
- **Limits:** second device invisible; VM or second user account possible; candidate can stop sharing (logged, but the window is then blind); OCR on compressed screenshare is lossy — expect misreads and require corroboration before flagging a specific tool by name.

### Option B — Web-based interview client (low friction, weak-to-moderate signal)
The candidate joins through your web app, which embeds or accompanies the meeting.

- **Signals:** Page Visibility API, `blur`/`focus`, fullscreen exit, paste events, unusual input timing.
- **Strengths:** zero install, minimal privacy footprint.
- **Limits:** tells you they left your tab, never where they went. Browsers deliberately do not expose other tabs. Modern browsers also throttle background timers, so treat gaps as lower-bound estimates. Trivially defeated by a second device.

### Option C — Desktop agent (strongest signal, highest cost)
An Electron or native agent enumerating foreground window titles, running processes, display count, VM/remote-desktop indicators, and clipboard events.

- **Strengths:** genuinely sees screen switching and AI-tool processes.
- **Costs:** install friction; antivirus false-positives; support burden; significant privacy intrusion; refusal rate among candidates; the heaviest legal exposure of any component here. Reserve for high-stakes roles, make it explicitly opt-in with a documented alternative, and scope collection tightly (window titles only, allowlist/denylist matching, no keystroke content, no screenshots without notice).

### Option D — Behavioral inference (no telemetry; last resort)
Answer-latency spikes, sudden fluency shifts, keyboard audio during silences, reading-pattern gaze. All circumstantial. Flag for review only; never present as detection of AI usage.

**Recommendation:** ship **A + B** as the default configuration, offer **C** as an opt-in tier, and always run **D** as corroboration rather than as a primary signal. State the residual blind spot (second device) in the report methodology section, every time.

---

## 8. Transcript-based AI-usage analysis

An analysis agent consumes the streaming transcript over a sliding window (~30s, with 10s overlap) and emits structured findings. **This is the least reliable signal in the system and must be weighted accordingly.**

### 8.1 Signal families

**Lexical / stylistic (weak).** Model-favored vocabulary and connectives ("delve," "moreover," "it is important to note," "in today's fast-paced world," "leverage," "robust framework"), tricolon structures, perfectly parallel clauses, hedged-then-balanced conclusions.

**Structural (moderate).** Orally delivered answers with clean enumerated structure ("There are three key reasons: first… second… third…"), unnatural completeness, textbook-shaped definitions, no self-correction.

**Acoustic / prosodic (strongest — prioritize this).** Reading cadence is far more discriminative than word choice:
- Disfluency rate — near-total absence of "um," "uh," "you know," restarts
- Pause placement at punctuation boundaries rather than at thought boundaries
- Flat pitch variance and metronomic pacing
- Onset latency: long silence after a question, then an immediately fluent, fully-formed answer
- Keyboard audio during answer-formulation silences

**Intra-session drift (strongest transcript-only signal).** Establish a linguistic and prosodic baseline from the candidate's own small talk and self-introduction, then measure deviation on technical answers. This is self-normalizing — it compares the candidate to themselves rather than to a population, which sidesteps most of the demographic bias in absolute-threshold detectors.

### 8.2 Implementation

Ensemble, not a single detector:

1. **Rule/lexicon layer** — versioned buzzword and phrase list, per-window density scoring. Cheap, fast, transparent, auditable.
2. **Statistical layer** — perplexity and burstiness over the window using a small local LM; low variance in sentence-level surprisal is the classic signal.
3. **LLM judge** — strict rubric, structured JSON output (`{finding, evidence_span, start_ms, end_ms, rationale, confidence}`). Forbid free-form verdicts. Require it to quote the exact span it is reacting to; a finding with no span is discarded.
4. **Acoustic layer** — disfluency and prosody features from the audio, aligned to the same windows.

Fuse with a calibrated model. Require **at least two independent families** to agree before emitting a finding above low severity.

### 8.3 Mandatory caveats — build these into the product, not just the docs

Automated AI-text detection is not reliable enough to support an accusation. Published evaluations repeatedly find elevated false-positive rates for **non-native English speakers**, for people who speak formally, and for candidates delivering **rehearsed answers** — which is exactly what interview preparation produces. Accordingly:

- Cap this family's contribution to the overall session score (suggested: ≤ 25%).
- Never let a transcript-only finding reach the highest severity tier on its own.
- Always print the verbatim quoted span next to the finding so a human can judge it.
- Label the section **"Linguistic pattern notes"**, not "AI detection."
- Tune thresholds for **precision over recall**. In hiring, a false accusation is far more costly than a missed instance of cheating. Target precision ≥ 0.9 on high-severity flags and accept whatever recall that yields.

---

## 9. Event schema and score fusion

Every plane writes to one normalized stream:

```json
{
  "session_id": "int_2f9c…",
  "event_id": "evt_8812",
  "t_start_ms": 1382000,
  "t_end_ms": 1418500,
  "wall_clock_utc": "2026-08-01T10:23:02Z",
  "subject": { "participant_id": "p_2", "role": "candidate", "identity_confidence": 0.97 },
  "category": "object_detected",
  "signal": "phone_in_hand_gaze_correlated",
  "source_plane": "vision",
  "confidence": 0.82,
  "severity": "medium",
  "corroborating_events": ["evt_8809"],
  "evidence": { "frames": ["s3://…/f1.jpg"], "clip": "s3://…/c1.mp4" },
  "coverage_ok": true,
  "requires_human_review": true
}
```

**Fusion rules:**
- Category weights are configurable per organization; there is no universal "cheating score."
- Corroboration across planes raises confidence multiplicatively; single-plane findings are capped.
- Any event inside a coverage gap or overlap window is capped and marked for review (§4, §5).
- Deduplicate: one sustained behavior is one event with a duration, never fifty per-frame events.
- The output is **`review_recommended: true/false`**, not a pass/fail or a cheating probability. The system does not adjudicate.

---

## 10. Report specification (2–3 pages, downloadable)

**Page 1 — Session summary**
- Header: candidate, role, interviewer(s), date, duration, platform, session ID
- Monitoring coverage bar: % of session with audio / video / screen observed, with gaps listed
- Observation counts by category, highest severity present
- **Outcome line:** `Human review recommended: Yes / No` — and nothing that reads as a verdict
- Consent record: what was disclosed, what the candidate agreed to, timestamp

**Page 2 — Timeline of observations**
- Horizontal timeline strip showing coverage, speech turns, and flagged windows
- Table: `Timestamp | Duration | Category | Description | Confidence | Evidence`
- Show only events above the configured confidence floor; cap at the top N by severity, with an overflow line ("+7 lower-confidence observations in the full appendix")

**Page 3 — Linguistic notes, methodology, limitations**
- Transcript findings with verbatim quoted spans and timestamps
- Explicit limitations section, printed on every report: second-device blind spot; gaze inference is circumstantial; linguistic analysis is indicative only; known false-positive conditions (lighting, glasses, non-native speech, rehearsed answers)
- Candidate's right to review and appeal, with the contact route
- Data retention and deletion date

**Keeping it to 2–3 pages:** the page budget is a hard constraint, so make it deterministic — fixed sections, capped table rows, overflow to a linked web appendix rather than extra pages. Full evidence (frames, clips, complete transcript) lives behind an authenticated link with its own access log, never inlined.

**Generation:** structured JSON → HTML template → PDF (WeasyPrint or headless Chrome; ReportLab if you want tighter layout control). Serve via expiring signed URL. Log every download.

---

## 11. Suggested stack

| Layer | Choice |
|---|---|
| Meeting ingestion | Recall.ai to start; Zoom Meeting SDK / Google Meet Media API when justified |
| ASR (streaming) | Deepgram or AssemblyAI; self-hosted Whisper large-v3 if data residency demands it |
| Diarization / overlap | pyannote.audio (fallback path only) |
| Face + landmarks | MediaPipe Face Mesh; InsightFace/ArcFace for verification embeddings |
| Head pose / gaze | solvePnP + a dedicated gaze model |
| Object detection | YOLO-family, fine-tuned on a proctoring dataset |
| Screenshare analysis | PaddleOCR + a small UI classifier |
| Transcript agent | Claude or equivalent with enforced structured output, plus rule/statistical layers |
| Queue / workers | Kafka or Redis Streams; Python workers, GPU pool for vision |
| Storage | Postgres for events; S3 with SSE and lifecycle expiry for media |
| Report service | FastAPI + WeasyPrint |

---

## 12. Ethical, legal, and fairness constraints (design requirements, not disclaimers)

These are engineering requirements because they change the architecture — bolt them on later and you will rebuild.

- **Consent gate before the bot records.** Disclose exactly what is monitored, what is stored, and for how long. Explicit opt-in. Offer a documented alternative (live human proctoring or in-person) for candidates who decline; a "consent" with no alternative is not consent.
- **Human in the loop, always.** No automated rejection, no automated scoring that feeds a hiring decision without a human reading the evidence. Employment-related automated decision systems face specific obligations across jurisdictions — the EU AI Act treats employment use as high-risk, NYC Local Law 144 requires bias audits for automated employment decision tools, Illinois regulates AI use in video interviews, and biometric templates (face and voice embeddings) are separately regulated under BIPA in Illinois and comparable laws in Texas and Washington. Get counsel to scope your specific deployment; the architecture must assume these constraints apply.
- **Observations, never verdicts.** The report describes what was detected with confidence and evidence. It does not say the candidate cheated.
- **Accommodations channel.** Neurodivergent candidates may show atypical gaze; some disabilities require assistive devices that resemble prohibited objects; some candidates cannot use a camera. Provide a pre-interview accommodation request that disables or adjusts specific checks, and record that the accommodation was applied so reviewers interpret the report correctly.
- **Measure disparate impact.** Run flag rates disaggregated by relevant cohorts (notably non-native English speakers, and lighting/skin-tone conditions for vision models) before launch and on a recurring schedule. If the false-positive rate is not equal across groups, the tool is not ready.
- **Data minimization and retention.** Default retention 30–90 days, automatic deletion, evidence media encrypted at rest, access-logged. Delete biometric enrollment templates at session end unless there is a documented reason to retain them.
- **Candidate access and appeal.** The candidate can request their report and contest a finding. Build the workflow; do not leave it to email.

---

## 13. Rollout

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | Legal review, consent flow, policy config, threat model | 2 wks |
| 1 | **MVP:** bot joins, records, per-participant transcript, identity binding, basic report | 4–6 wks |
| 2 | Vision pipeline: calibration, gaze/pose, objects, multi-face, liveness | 4–6 wks |
| 3 | Screen telemetry: screenshare classification + web-client focus events | 3–4 wks |
| 4 | Transcript + acoustic analysis, fusion model, threshold tuning | 4 wks |
| 5 | **Shadow mode**, bias audit, appeal workflow, hardening | 4+ wks |

**Phase 5 is not optional.** Run the full system for several weeks producing reports that no decision-maker sees, and measure the false-positive rate against known-clean sessions before anyone acts on output. A proctoring tool that ships without this measurement is shipping an unknown accusation rate.

**Targets to hold yourself to:**
- Precision on high-severity flags ≥ 0.9
- False positives per clean candidate-hour < 0.5
- Monitoring coverage ≥ 95% on healthy sessions
- Report generated within 5 minutes of session end
- Flag rate parity across cohorts within a defined tolerance
- Cost per interview-hour tracked from day one (GPU vision is the dominant line item; frame sampling rate is the main lever)

---

## 14. Open risks

1. **Platform media access.** Zoom and Google both gate raw per-participant media behind app review and changing terms. Validate access early; the whole overlap-handling design depends on it.
2. **Second-device blind spot.** Unfixable within this architecture. Manage by disclosure, not by pretending otherwise.
3. **Adversarial candidates.** Virtual cameras, VMs, screen-share spoofing, prompt injection aimed at the transcript agent (a candidate reading "ignore previous instructions" aloud). Treat all transcript content as untrusted input to the LLM judge.
4. **AI-detection reliability.** The weakest link and the most reputationally dangerous. Keep it capped, corroborated, and clearly labeled.
5. **Candidate trust.** Heavy surveillance affects offer-acceptance rates and employer brand. Worth measuring alongside the integrity metrics.
