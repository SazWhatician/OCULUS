"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { fetchSession, swapSpeakers, completeSession, WS_BASE_URL, SessionDetail } from "@/lib/api";
import { webrtcCapture, CaptureTelemetry } from "@/lib/webrtcCapture";

interface UtteranceItem {
  utterance_id: string;
  speaker: string;
  speaker_label: string;
  text: string;
  start_ms: number;
  end_ms: number;
}

interface VisionEventItem {
  event_id: string;
  anomaly_type: string;
  timestamp_ms: number;
  duration_ms: number;
  confidence: number;
  vlm_verification_note?: string;
  evidence_url?: string;
}

interface VisionTelemetryItem {
  timestamp_ms: number;
  engine: string;
  camera_active: boolean;
  faces_detected: number;
  gaze_direction: string;
  gaze_offset_x: number;
  gaze_offset_y: number;
  head_pose_yaw: number;
  head_pose_pitch: number;
  head_pose_roll: number;
  liveness_score: number;
}

interface QAEvalItem {
  qa_index: number;
  question_text: string;
  answer_text: string;
  combined_risk_score: number;
  heuristics: {
    wpm: number;
    wpm_robust_zscore: number;
    disfluency_rate: number;
    disfluency_drop: number;
    ai_tell_matches: string[];
    first_person_ratio: number;
    deterministic_risk: number;
    trigger_llm_judge: boolean;
  };
  llm_verdict?: {
    suspicion_score: number;
    confidence_band: string;
    reasoning_signals: string[];
    recitation_vs_live_assessment: string;
    advisory_follow_up_question: string;
  } | null;
}

export default function LiveSentinelPage() {
  const router = useRouter();
  const params = useParams();
  const sessionId = Array.isArray(params.id) ? params.id[0] : (params.id as string);

  const [session, setSession] = useState<SessionDetail | null>(null);
  const [status, setStatus] = useState<string>("PROCESSING");
  const [riskScore, setRiskScore] = useState<number>(0);
  const [riskBand, setRiskBand] = useState<"LOW" | "MODERATE" | "ELEVATED">("LOW");
  const [speakerMap, setSpeakerMap] = useState<Record<string, string>>({ S1: "interviewer", S2: "candidate" });
  const [baseline, setBaseline] = useState<any>(null);

  const [utterances, setUtterances] = useState<UtteranceItem[]>([]);
  const [visionEvents, setVisionEvents] = useState<VisionEventItem[]>([]);
  const [visionTelemetry, setVisionTelemetry] = useState<VisionTelemetryItem | null>(null);
  const [qaEvaluations, setQaEvaluations] = useState<QAEvalItem[]>([]);

  const [filterSpeaker, setFilterSpeaker] = useState<"ALL" | "candidate" | "interviewer">("ALL");
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [endingSession, setEndingSession] = useState<boolean>(false);
  const [captureTelemetry, setCaptureTelemetry] = useState<CaptureTelemetry>(() => webrtcCapture.getState());

  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    return webrtcCapture.subscribe((snap) => setCaptureTelemetry(snap));
  }, []);

  useEffect(() => {
    webrtcCapture.attachPreview(previewVideoRef.current);
  }, [captureTelemetry.active, captureTelemetry.sessionId]);

  // Auto-scroll transcript on new utterance
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [utterances]);

  // Initial HTTP hydration
  useEffect(() => {
    async function loadSession() {
      const data = await fetchSession(sessionId);
      if (data) {
        setSession(data);
        setStatus(data.status);
        if (data.speaker_role_map) setSpeakerMap(data.speaker_role_map);
        if (data.baseline) setBaseline(data.baseline);
        if (Array.isArray(data.utterances)) setUtterances(data.utterances);
        if (Array.isArray(data.qa_evaluations)) setQaEvaluations(data.qa_evaluations);
        if (Array.isArray(data.vision_events)) setVisionEvents(data.vision_events);
        if (data.report) {
          setRiskScore(data.report.overall_risk_score || 0);
          setRiskBand(data.report.risk_band || "LOW");
        }
      }
    }
    loadSession();
  }, [sessionId]);

  // WebSocket Live Subscription
  useEffect(() => {
    if (!sessionId) return;

    const wsUrl = `${WS_BASE_URL}/ws/sessions/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsConnected(true);
    };

    ws.onclose = () => {
      setWsConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const eventType = String(msg.event_type || "").toUpperCase();
        const data = msg.data ?? msg.payload ?? msg;

        if (eventType === "PING") {
          return;
        } else if (eventType === "HYDRATE_STATE") {
          if (data?.status) setStatus(data.status);
          if (Array.isArray(data?.utterances)) setUtterances(data.utterances);
          if (Array.isArray(data?.qa_evaluations)) setQaEvaluations(data.qa_evaluations);
          if (Array.isArray(data?.vision_events)) setVisionEvents(data.vision_events);
          if (data?.baseline) setBaseline(data.baseline);
          if (typeof data?.overall_risk_score === "number") setRiskScore(data.overall_risk_score);
          if (data?.risk_band) setRiskBand(data.risk_band);
        } else if (eventType === "UTTERANCE" && data) {
          setUtterances((prev) => {
            const exists = prev.some((u) => u.utterance_id === data.utterance_id);
            return exists
              ? prev.map((u) => (u.utterance_id === data.utterance_id ? data : u))
              : [...prev, data];
          });
        } else if (eventType === "QA_EVALUATED" && data) {
          const qaItem = data.qa_evaluation ?? data;
          if (qaItem && typeof qaItem.qa_index === "number") {
            setQaEvaluations((prev) => {
              const exists = prev.some((q) => q.qa_index === qaItem.qa_index);
              return exists
                ? prev.map((q) => (q.qa_index === qaItem.qa_index ? qaItem : q))
                : [...prev, qaItem];
            });
          }
          if (typeof data.overall_risk_score === "number") setRiskScore(data.overall_risk_score);
          if (data.risk_band) setRiskBand(data.risk_band);
          if (data.baseline_calibrated) {
            fetchSession(sessionId).then((fresh) => {
              if (fresh?.baseline) setBaseline(fresh.baseline);
            });
          }
        } else if (eventType === "RISK_UPDATE" && data) {
          setRiskScore(data.overall_risk_score || 0);
          setRiskBand(data.risk_band || "LOW");
        } else if (eventType === "BASELINE_CALIBRATED" && data) {
          setBaseline(data);
        } else if (eventType === "VISION_TELEMETRY" && data) {
          setVisionTelemetry(data);
        } else if (eventType === "VISION_ANOMALY" && data) {
          setVisionEvents((prev) => {
            const exists = prev.some((v) => v.event_id === data.event_id);
            return exists ? prev : [...prev, data];
          });
        } else if (eventType === "SPEAKER_MAP_UPDATED") {
          if (data?.speaker_role_map) setSpeakerMap(data.speaker_role_map);
          fetchSession(sessionId).then((fresh) => {
            if (!fresh) return;
            if (Array.isArray(fresh.utterances)) setUtterances(fresh.utterances);
            if (Array.isArray(fresh.qa_evaluations)) setQaEvaluations(fresh.qa_evaluations);
            if (fresh.baseline) setBaseline(fresh.baseline);
            if (fresh.report) {
              setRiskScore(fresh.report.overall_risk_score || 0);
              setRiskBand(fresh.report.risk_band || "LOW");
            }
          });
        } else if (eventType === "STATUS_UPDATE" || eventType === "STATUS_CHANGE") {
          if (data?.status) setStatus(data.status);
          if (typeof data?.overall_risk_score === "number") setRiskScore(data.overall_risk_score);
          if (data?.risk_band) setRiskBand(data.risk_band);
        }
      } catch (err) {
        console.error("WS Parse Error:", err);
      }
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  // Speaker Swap Action
  const handleSwapSpeakers = async () => {
    try {
      const nextMap = {
        S1: speakerMap["S1"] === "candidate" ? "interviewer" : "candidate",
        S2: speakerMap["S2"] === "candidate" ? "interviewer" : "candidate",
      };
      const res = await swapSpeakers(sessionId, nextMap);
      const updated = res.speaker_role_map || nextMap;
      setSpeakerMap(updated);
      setUtterances((prev) =>
        prev.map((u) => {
          const newRole = updated[u.speaker_label] || u.speaker;
          return { ...u, speaker: newRole };
        })
      );
    } catch (err) {
      alert("Failed to swap speaker roles.");
    }
  };

  // Complete Session Action
  const handleCompleteSession = async () => {
    if (!confirm("Are you sure you want to conclude monitoring and compile the final Audit Report?")) return;
    setEndingSession(true);
    try {
      if (captureTelemetry.active && captureTelemetry.sessionId === sessionId) {
        await webrtcCapture.stop();
        await new Promise((r) => setTimeout(r, 150));
      }
      await completeSession(sessionId);
      setStatus("COMPLETED");
      router.push(`/reports/${sessionId}`);
    } catch (err: any) {
      alert(`Error completing session: ${err.message}`);
    } finally {
      setEndingSession(false);
    }
  };

  const isThisSessionStreaming = captureTelemetry.active && captureTelemetry.sessionId === sessionId;
  const hasVisionActivity =
    (isThisSessionStreaming && captureTelemetry.framesSent > 0) ||
    visionTelemetry !== null ||
    visionEvents.length > 0 ||
    status === "COMPLETED";
  const latestVisionFlag = visionEvents[visionEvents.length - 1];
  const latestGazeFlag = [...visionEvents]
    .reverse()
    .find((v) => v.anomaly_type === "GAZE_DEVIATION" || v.anomaly_type === "TELEPROMPTER_READING");
  const latestPoseFlag = [...visionEvents]
    .reverse()
    .find((v) => v.anomaly_type === "HEAD_PITCH_DOWN" || v.anomaly_type === "CAMERA_OCCLUSION");
  const latestSubjectFlag = [...visionEvents]
    .reverse()
    .find((v) => v.anomaly_type === "MULTI_FACE" || v.anomaly_type === "ABSENT_FACE" || v.anomaly_type === "LIVENESS_SPOOF");

  const filteredUtterances = utterances.filter((u) => {
    if (filterSpeaker === "ALL") return true;
    return u.speaker.toLowerCase() === filterSpeaker.toLowerCase();
  });

  const latestAdvisoryQuestion = qaEvaluations
    .slice()
    .reverse()
    .find((q) => q.llm_verdict?.advisory_follow_up_question)
    ?.llm_verdict?.advisory_follow_up_question;

  const isElevated = riskBand === "ELEVATED";

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Command Bar */}
      <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 sm:p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-muted)] hover:text-[var(--oculus-ink)] transition-colors inline-flex items-center gap-1.5"
            >
              <span>←</span>
              <span>INDEX // 01</span>
            </Link>
            <span className="text-[var(--oculus-hairline)]">/</span>
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
              SENTINEL // {sessionId.slice(0, 8)}
            </span>
            <span className="inline-flex items-center gap-2 border border-[var(--oculus-hairline)] px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.24em] text-[var(--oculus-ink)]">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  wsConnected ? "bg-[#10B981] animate-pulse" : "bg-[#E12D1E]"
                }`}
              />
              {wsConnected ? "FEED ACTIVE" : "RECONNECTING"}
            </span>
          </div>

          <div className="flex flex-wrap items-baseline gap-3">
            <h1 className="font-display text-2xl font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
              {session?.candidate_name || "Alex Johnson"}
            </h1>
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--oculus-muted)]">
              {session?.job_title || "Technical Role"}
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleSwapSpeakers}
            className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-2.5 font-mono text-[10px] uppercase tracking-[0.2em] text-[var(--oculus-ink)] hover:border-[var(--oculus-ink)] transition-colors"
            title="Swap S1 and S2 roles if diarization detected interviewer as candidate"
          >
            ⇄ SWAP S1/S2 (S1: {speakerMap["S1"]})
          </button>

          {status === "COMPLETED" ? (
            <Link
              href={`/reports/${sessionId}`}
              className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-5 py-2.5 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-bg)] hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] transition-colors"
            >
              VIEW AUDIT REPORT →
            </Link>
          ) : (
            <button
              onClick={handleCompleteSession}
              disabled={endingSession}
              className="border border-[#E12D1E] bg-[#E12D1E] px-5 py-2.5 font-mono text-[10px] uppercase tracking-[0.24em] text-white hover:bg-transparent hover:text-[#E12D1E] transition-colors disabled:opacity-50"
            >
              {endingSession ? "COMPILING REPORT..." : "■ CONCLUDE & AUDIT"}
            </button>
          )}
        </div>
      </div>

      {/* Main Grid: 60% Left Stream / 40% Right Proctoring Gauges */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Diarized Transcript Stream & Evidence */}
        <div className="lg:col-span-7 space-y-6">
          {/* Transcript Feed Card */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 sm:p-6 space-y-4 flex flex-col h-[620px]">
            {/* Header & Filter Tabs */}
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--oculus-muted)]">
                  SEC. 01
                </span>
                <span className="font-display text-sm font-semibold tracking-[-0.01em] text-[var(--oculus-ink)]">
                  LIVE DIARIZED TRANSCRIPTION
                </span>
                <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
                  {utterances.length} TURNS
                </span>
              </div>

              <div className="inline-flex border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] text-[9.5px] font-mono uppercase tracking-[0.18em]">
                {(["ALL", "candidate", "interviewer"] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setFilterSpeaker(filter)}
                    className={`px-3 py-1.5 transition-colors ${
                      filterSpeaker === filter
                        ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)]"
                        : "text-[var(--oculus-muted)] hover:text-[var(--oculus-ink)]"
                    }`}
                  >
                    {filter === "ALL" ? "ALL" : filter.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Scrollable Message Feed */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2 font-mono text-xs">
              {filteredUtterances.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-2 text-[var(--oculus-muted)]">
                  <div className="font-mono text-xl opacity-40">◫</div>
                  <p className="font-mono text-[10.5px] uppercase tracking-[0.24em] text-[var(--oculus-ink)]">
                    BUFFERING INCOMING MULTIMODAL MEDIA STREAM...
                  </p>
                  <p className="font-mono text-[9.5px] uppercase tracking-[0.16em] max-w-xs">
                    Chunks stitched across Speechmatics diarization & Groq Whisper Turbo.
                  </p>
                </div>
              ) : (
                filteredUtterances.map((u) => {
                  const isCandidate = u.speaker.toLowerCase() === "candidate";
                  const startSec = Math.floor(u.start_ms / 1000);
                  const min = Math.floor(startSec / 60);
                  const sec = startSec % 60;
                  const timeFormatted = `${min.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;

                  return (
                    <div
                      key={u.utterance_id}
                      className={`p-4 border transition-all ${
                        isCandidate
                          ? "border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] ml-4"
                          : "border-transparent bg-transparent mr-4 opacity-80"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span
                          className={`font-mono text-[9.5px] uppercase tracking-[0.2em] font-semibold flex items-center gap-2 ${
                            isCandidate ? "text-[var(--oculus-ink)]" : "text-[var(--oculus-muted)]"
                          }`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              isCandidate ? "bg-[var(--oculus-ink)]" : "bg-[var(--oculus-muted)]"
                            }`}
                          />
                          {isCandidate ? session?.candidate_name || "CANDIDATE" : "INTERVIEWER"} ({u.speaker_label})
                        </span>
                        <span className="font-mono text-[9px] text-[var(--oculus-muted)] tabular-nums">
                          {timeFormatted}
                        </span>
                      </div>
                      <p className="text-[var(--oculus-ink)] text-[12px] leading-relaxed uppercase tracking-[0.04em]">
                        {u.text}
                      </p>
                    </div>
                  );
                })
              )}
              <div ref={transcriptEndRef} />
            </div>
          </div>

          {/* Real-Time Recruiter Probe Advisory Card */}
          {latestAdvisoryQuestion && (
            <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 space-y-2 font-mono">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#E12D1E] flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#E12D1E] animate-pulse" />
                  ADVISORY VERIFICATION PROBE (STAGE 2 LLM JUDGE)
                </span>
                <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 text-[9px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
                  LIVE TELEMETRY
                </span>
              </div>
              <p className="text-[12px] font-semibold text-[var(--oculus-ink)] leading-relaxed uppercase tracking-[0.06em]">
                &ldquo;{latestAdvisoryQuestion}&rdquo;
              </p>
              <p className="text-[9.5px] uppercase tracking-[0.16em] text-[var(--oculus-muted)]">
                Inquire with this probe to differentiate architectural comprehension from generative teleprompter recitation.
              </p>
            </div>
          )}

          {/* Evaluated Q&A Turns Stream */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 sm:p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                  SEC. 02
                </span>
                <h2 className="font-display text-sm font-semibold tracking-[-0.01em] text-[var(--oculus-ink)]">
                  CALIBRATED Q&A EVALUATIONS ({qaEvaluations.length})
                </h2>
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
                STAGE 1 + 2 PIPELINE
              </span>
            </div>

            {qaEvaluations.length === 0 ? (
              <div className="p-8 text-center font-mono text-[10.5px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
                Awaiting Question 0 baseline calibration to initiate turn scoring.
              </div>
            ) : (
              <div className="space-y-4">
                {qaEvaluations.map((qa) => {
                  const score = Math.round(qa.combined_risk_score);
                  const isTurnElevated = score >= 65;

                  return (
                    <div
                      key={qa.qa_index}
                      className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4 space-y-3 font-mono text-[11px]"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold uppercase tracking-[0.2em] text-[var(--oculus-ink)]">
                          TURN #{String(qa.qa_index + 1).padStart(2, "0")}
                        </span>
                        <span
                          className={`font-mono text-[10px] font-bold px-2 py-0.5 border tabular-nums ${
                            isTurnElevated
                              ? "border-[#E12D1E] text-[#E12D1E]"
                              : "border-[var(--oculus-hairline)] text-[var(--oculus-ink)]"
                          }`}
                        >
                          RISK: {score}%
                        </span>
                      </div>

                      <div className="space-y-1.5 text-[10.5px] uppercase tracking-[0.06em]">
                        <div className="text-[var(--oculus-muted)]">Q: {qa.question_text}</div>
                        <div className="text-[var(--oculus-ink)] line-clamp-2">A: {qa.answer_text}</div>
                      </div>

                      <div className="flex items-center gap-2 flex-wrap text-[9px] pt-1">
                        <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 text-[var(--oculus-muted)]">
                          WPM: {Math.round(qa.heuristics.wpm)}
                        </span>
                        {qa.heuristics.disfluency_drop > 0 && (
                          <span className="border border-[#E12D1E] text-[#E12D1E] px-2 py-0.5">
                            DISFLUENCY DROP: −{qa.heuristics.disfluency_drop.toFixed(1)}%
                          </span>
                        )}
                        {qa.heuristics.ai_tell_matches?.length > 0 && (
                          <span className="border border-[#E12D1E] text-[#E12D1E] px-2 py-0.5">
                            TELL: {qa.heuristics.ai_tell_matches.join(", ")}
                          </span>
                        )}
                        {qa.llm_verdict && (
                          <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 text-[var(--oculus-ink)]">
                            LLM JUDGE: {qa.llm_verdict.confidence_band}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Gauges, Vision Proctoring, Baseline */}
        <div className="lg:col-span-5 space-y-6">
          {/* Cumulative Rolling Risk Gauge */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 text-center space-y-4">
            <span className="font-mono text-[9.5px] uppercase tracking-[0.28em] text-[var(--oculus-muted)] block">
              CUMULATIVE SENTINEL RISK INDEX
            </span>

            <div className="py-4">
              <div
                className={`w-36 h-36 mx-auto border-2 flex flex-col items-center justify-center transition-all duration-500 ${
                  isElevated
                    ? "border-[#E12D1E] text-[#E12D1E]"
                    : "border-[var(--oculus-ink)] text-[var(--oculus-ink)]"
                }`}
              >
                <span className="font-display text-4xl font-semibold tracking-tight tabular-nums">
                  {Math.round(riskScore)}%
                </span>
                <span className="font-mono text-[9px] font-bold uppercase tracking-[0.24em] mt-1">
                  {riskBand}
                </span>
              </div>
            </div>

            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--oculus-muted)] max-w-xs mx-auto leading-relaxed">
              {isElevated
                ? "ELEVATED RISK: ACCELERATED FLUENCY WITH PROLONGED OPTICAL GAZE DEFLECTION OBSERVED."
                : riskBand === "MODERATE"
                ? "MODERATE VARIATION: FORMULATION INCONSISTENT WITH Q0 DISFLUENCY BASELINE."
                : "NOMINAL INTEGRITY: NATURAL CADENCE AND RETICLE GAZE PARITY."}
            </p>
          </div>

          {/* Q0 Baseline Calibration Card */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-muted)] font-semibold">
                Q0 CANDIDATE BASELINE
              </span>
              <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.2em] text-[var(--oculus-ink)]">
                {baseline ? "CALIBRATED" : "CALIBRATING..."}
              </span>
            </div>

            {baseline ? (
              <div className="grid grid-cols-2 gap-3 font-mono text-[10px]">
                <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                  <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">MEDIAN WPM</span>
                  <span className="font-display text-xl font-semibold text-[var(--oculus-ink)] tabular-nums mt-0.5 block">
                    {Math.round(baseline.median_wpm || baseline.baseline_wpm || 135)}
                  </span>
                </div>
                <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                  <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">DISFLUENCY</span>
                  <span className="font-display text-xl font-semibold text-[var(--oculus-ink)] tabular-nums mt-0.5 block">
                    {(baseline.baseline_disfluency_rate || 0).toFixed(1)}%
                  </span>
                </div>
                <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                  <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">MAD SIGMA</span>
                  <span className="font-display text-xl font-semibold text-[var(--oculus-ink)] tabular-nums mt-0.5 block">
                    {(baseline.mad_wpm || 14).toFixed(1)}
                  </span>
                </div>
                <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                  <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">PAUSE RATIO</span>
                  <span className="font-display text-xl font-semibold text-[var(--oculus-ink)] tabular-nums mt-0.5 block">
                    {Math.round((baseline.baseline_pause_ratio || 0.12) * 100)}%
                  </span>
                </div>
              </div>
            ) : (
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--oculus-muted)] leading-relaxed">
                Listening to Question 0 formulation (&quot;Tell me about yourself&quot;) to establish non-native ESL baseline.
              </p>
            )}
          </div>

          {/* Optical & Gaze Telemetry Card */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-muted)] font-semibold">
                OPTICAL SENTINEL (1 FPS)
              </span>
              <span className="border border-[var(--oculus-hairline)] px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
                DEBOUNCED ≥3000MS
              </span>
            </div>

            {/* Video Reticle Preview Frame */}
            <div className="w-full h-44 bg-[var(--oculus-bg)] border border-[var(--oculus-hairline)] flex flex-col items-center justify-center relative overflow-hidden">
              {isThisSessionStreaming && captureTelemetry.videoActive && (
                <video
                  ref={previewVideoRef}
                  autoPlay
                  muted
                  playsInline
                  className="absolute inset-0 w-full h-full object-cover opacity-80"
                />
              )}

              {/* Reticle lines */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-25">
                <div className="w-24 h-24 border border-current rounded-full" />
                <div className="absolute h-full w-px bg-current" />
                <div className="absolute w-full h-px bg-current" />
              </div>

              <div className="absolute top-3 left-3 flex items-center gap-2 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]/95 px-2 py-1 font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-ink)]">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    latestVisionFlag
                      ? "bg-[#E12D1E] animate-pulse"
                      : isThisSessionStreaming
                      ? "bg-[#10B981] animate-pulse"
                      : "bg-[var(--oculus-muted)]"
                  }`}
                />
                <span>
                  {latestVisionFlag
                    ? `FLAGGED: ${latestVisionFlag.anomaly_type}`
                    : isThisSessionStreaming
                    ? `STREAMING · ${captureTelemetry.framesSent}F / ${captureTelemetry.chunksSent}A`
                    : visionTelemetry
                    ? `TRACKING · ${visionTelemetry.engine.toUpperCase()}`
                    : status === "COMPLETED"
                    ? "MONITORING CONCLUDED"
                    : session?.ingestion_mode === "demo_file"
                    ? "REPLAY PIPELINE"
                    : "AWAITING VIDEO STREAM"}
                </span>
              </div>

              {session?.ingestion_mode === "browser_webrtc" && status !== "COMPLETED" && (
                <button
                  type="button"
                  onClick={() => {
                    if (isThisSessionStreaming) {
                      void webrtcCapture.stop();
                    } else {
                      void webrtcCapture.start(sessionId, WS_BASE_URL).catch((err) => alert(err.message));
                    }
                  }}
                  className="absolute bottom-3 right-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]/95 hover:bg-[var(--oculus-ink)] hover:text-[var(--oculus-bg)] px-2.5 py-1 font-mono text-[9px] uppercase tracking-[0.2em] transition-colors cursor-pointer"
                >
                  {isThisSessionStreaming ? "STOP CAPTURE" : "START CAPTURE"}
                </button>
              )}

              {!(isThisSessionStreaming && captureTelemetry.videoActive) && (
                <div className="text-center space-y-1 font-mono">
                  <span className="text-2xl text-[var(--oculus-ink)] opacity-60">◎</span>
                  <span className="text-[10.5px] uppercase tracking-[0.22em] font-semibold text-[var(--oculus-ink)] block">
                    {visionTelemetry?.engine === "mediapipe_tasks" || visionTelemetry?.engine === "mediapipe_facemesh"
                      ? "MEDIAPIPE 478-PT 3D TRACKER"
                      : visionTelemetry?.engine === "opencv_yunet"
                      ? "OPENCV YUNET 5-PT SOLVEPNP"
                      : "OPENCV OPTICAL GAZE TRACKER"}
                  </span>
                  <span className="text-[9px] uppercase tracking-[0.18em] text-[var(--oculus-muted)] block">
                    3-TIER FUNNEL // CPU OPTIMIZED
                  </span>
                </div>
              )}
            </div>

            {/* Vision Telemetry Status Indicators */}
            <div className="grid grid-cols-3 gap-2 text-center font-mono text-[9.5px]">
              <div className="p-2.5 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                <span className="text-[var(--oculus-muted)] uppercase tracking-[0.16em] block">GAZE OFFSET</span>
                <span className={`font-bold mt-0.5 block ${latestGazeFlag ? "text-[#E12D1E]" : "text-[var(--oculus-ink)]"}`}>
                  {latestGazeFlag
                    ? latestGazeFlag.anomaly_type === "TELEPROMPTER_READING"
                      ? "SCRIPT SCAN"
                      : "OFF-AXIS"
                    : visionTelemetry
                    ? `${visionTelemetry.gaze_direction.toUpperCase()} (${visionTelemetry.gaze_offset_x >= 0 ? "+" : ""}${Math.round(visionTelemetry.gaze_offset_x * 35)}°)`
                    : hasVisionActivity
                    ? "CENTERED"
                    : "IDLE"}
                </span>
              </div>
              <div className="p-2.5 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                <span className="text-[var(--oculus-muted)] uppercase tracking-[0.16em] block">HEAD POSE</span>
                <span className={`font-bold mt-0.5 block ${latestPoseFlag ? "text-[#E12D1E]" : "text-[var(--oculus-ink)]"}`}>
                  {latestPoseFlag
                    ? latestPoseFlag.anomaly_type === "CAMERA_OCCLUSION"
                      ? "OCCLUDED"
                      : "PITCH SHIFT"
                    : visionTelemetry
                    ? `${Math.round(visionTelemetry.head_pose_yaw)}°Y / ${Math.round(visionTelemetry.head_pose_pitch)}°P`
                    : hasVisionActivity
                    ? "ALIGNED"
                    : "IDLE"}
                </span>
              </div>
              <div className="p-2.5 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)]">
                <span className="text-[var(--oculus-muted)] uppercase tracking-[0.16em] block">SUBJECT COUNT</span>
                <span className={`font-bold mt-0.5 block ${latestSubjectFlag ? "text-[#E12D1E]" : "text-[var(--oculus-ink)]"}`}>
                  {latestSubjectFlag
                    ? latestSubjectFlag.anomaly_type === "MULTI_FACE"
                      ? "MULTI-FACE"
                      : latestSubjectFlag.anomaly_type === "ABSENT_FACE"
                      ? "0 FACES"
                      : "SPOOF RISK"
                    : visionTelemetry
                    ? `${visionTelemetry.faces_detected} ${visionTelemetry.faces_detected === 1 ? "FACE" : "FACES"}`
                    : hasVisionActivity
                    ? "1 FACE"
                    : "IDLE"}
                </span>
              </div>
            </div>

            {/* Vision Anomaly Timeline Feed */}
            <div className="space-y-2 pt-1 font-mono text-[10px]">
              <span className="text-[var(--oculus-muted)] uppercase tracking-[0.2em] block font-semibold">
                CONFIRMED OPTICAL FLAGS ({visionEvents.length})
              </span>

              {visionEvents.length === 0 ? (
                <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] text-center text-[9.5px] uppercase tracking-[0.16em] text-[var(--oculus-muted)]">
                  NO SUSTAINED (&ge;3S) GAZE DEFLECTIONS LOGGED
                </div>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {visionEvents.map((evt) => (
                    <div
                      key={evt.event_id}
                      className="p-2.5 border border-[#E12D1E]/30 bg-[#E12D1E]/5 flex items-center justify-between"
                    >
                      <div className="space-y-0.5">
                        <span className="font-bold text-[#E12D1E] block">{evt.anomaly_type}</span>
                        <span className="text-[9px] uppercase tracking-[0.14em] text-[var(--oculus-muted)]">
                          SUSTAINED {Math.round(evt.duration_ms / 1000)}S
                          {evt.vlm_verification_note && ` · VLM: ${evt.vlm_verification_note}`}
                        </span>
                      </div>
                      <span className="border border-[#E12D1E]/30 px-2 py-0.5 tabular-nums text-[9px] text-[#E12D1E]">
                        AT {Math.floor(evt.timestamp_ms / 1000)}S
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
