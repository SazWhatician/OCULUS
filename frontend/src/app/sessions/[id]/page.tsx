"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { fetchSession, swapSpeakers, completeSession, WS_BASE_URL, API_BASE_URL, SessionDetail } from "@/lib/api";

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
  const [qaEvaluations, setQaEvaluations] = useState<QAEvalItem[]>([]);

  const [filterSpeaker, setFilterSpeaker] = useState<"ALL" | "candidate" | "interviewer">("ALL");
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [endingSession, setEndingSession] = useState<boolean>(false);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

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

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const eventType = msg.event_type;
        const data = msg.data;

        if (eventType === "HYDRATE_STATE") {
          if (data.status) setStatus(data.status);
          if (data.baseline) setBaseline(data.baseline);
          if (data.utterances) setUtterances(data.utterances);
          if (data.report) {
            setRiskScore(data.report.overall_risk_score || 0);
            setRiskBand(data.report.risk_band || "LOW");
            if (data.report.qa_evaluations) setQaEvaluations(data.report.qa_evaluations);
            if (data.report.vision_timeline) setVisionEvents(data.report.vision_timeline);
          }
        } else if (eventType === "UTTERANCE") {
          setUtterances((prev) => {
            const exists = prev.some((u) => u.utterance_id === data.utterance_id);
            if (exists) return prev;
            return [...prev, data];
          });
        } else if (eventType === "QA_EVALUATED") {
          if (data.overall_risk_score !== undefined) setRiskScore(data.overall_risk_score);
          if (data.risk_band) setRiskBand(data.risk_band);
          if (data.qa_evaluation) {
            setQaEvaluations((prev) => [...prev, data.qa_evaluation]);
          }
        } else if (eventType === "VISION_ANOMALY") {
          setVisionEvents((prev) => [data, ...prev]);
        } else if (eventType === "STATUS_UPDATE") {
          if (data.status) setStatus(data.status);
          if (data.overall_risk_score !== undefined) setRiskScore(data.overall_risk_score);
          if (data.risk_band) setRiskBand(data.risk_band);
        }
      } catch (err) {
        console.error("WS message parse error:", err);
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setWsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  // Speaker Swap Action
  const handleSwapSpeakers = async () => {
    const newMap = {
      S1: speakerMap["S1"] === "candidate" ? "interviewer" : "candidate",
      S2: speakerMap["S2"] === "candidate" ? "interviewer" : "candidate",
    };
    try {
      await swapSpeakers(sessionId, newMap);
      setSpeakerMap(newMap);
    } catch (err) {
      alert("Failed to swap speaker roles.");
    }
  };

  // Complete Session Action
  const handleCompleteSession = async () => {
    if (!confirm("Are you sure you want to conclude monitoring and generate the final Audit Report?")) return;
    setEndingSession(true);
    try {
      await completeSession(sessionId);
      setStatus("COMPLETED");
      router.push(`/reports/${sessionId}`);
    } catch (err: any) {
      alert(`Error completing session: ${err.message}`);
    } finally {
      setEndingSession(false);
    }
  };

  const filteredUtterances = utterances.filter((u) => {
    if (filterSpeaker === "ALL") return true;
    return u.speaker.toLowerCase() === filterSpeaker.toLowerCase();
  });

  const latestAdvisoryQuestion = qaEvaluations
    .slice()
    .reverse()
    .find((q) => q.llm_verdict?.advisory_follow_up_question)
    ?.llm_verdict?.advisory_follow_up_question;

  return (
    <div className="space-y-6">
      {/* Top Command Bar */}
      <div className="apple-card p-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Link href="/" className="text-xs font-semibold text-[var(--apple-accent)] hover:underline">
              ← Dashboard
            </Link>
            <span className="text-xs text-[var(--apple-text-muted)]">•</span>
            <span className="apple-badge font-mono text-[10px]">Session: {sessionId}</span>
            <span
              className={`apple-badge text-[10px] font-bold ${
                wsConnected ? "bg-emerald-500/10 text-emerald-600" : "bg-rose-500/10 text-rose-600"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? "bg-emerald-500 live-pulse" : "bg-rose-500"} mr-1.5`}></span>
              {wsConnected ? "LIVE WS CONNECTED" : "RECONNECTING"}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <h1 className="text-xl font-extrabold text-[var(--apple-text-primary)]">
              {session?.candidate_name || "Candidate"}
            </h1>
            <span className="text-xs text-[var(--apple-text-secondary)] font-medium">
              ({session?.job_title || "Technical Role"})
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <button
            onClick={handleSwapSpeakers}
            className="apple-button-secondary px-3.5 py-1.5 text-xs font-bold inline-flex items-center gap-1.5"
            title="Swap S1 and S2 roles if diarization detected interviewer as candidate"
          >
            <span>⇄ Swap Roles (S1: {speakerMap["S1"]})</span>
          </button>

          {status === "COMPLETED" ? (
            <Link
              href={`/reports/${sessionId}`}
              className="apple-button-primary px-4 py-1.5 text-xs font-bold inline-flex items-center gap-1.5"
            >
              <span>View Audit Report →</span>
            </Link>
          ) : (
            <button
              onClick={handleCompleteSession}
              disabled={endingSession}
              className="apple-button-primary px-4 py-1.5 text-xs font-bold inline-flex items-center gap-1.5 bg-rose-600 hover:bg-rose-700"
            >
              <span>{endingSession ? "Finalizing Report..." : "■ End Session & View Report"}</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Grid: 60% Left Stream / 40% Right Proctoring Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Diarized Transcript Stream & Evidence */}
        <div className="lg:col-span-7 space-y-6">
          {/* Transcript Card */}
          <div className="apple-card p-6 space-y-4 flex flex-col h-[600px]">
            {/* Header & Filter Tabs */}
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider">
                  Live Diarized Transcript
                </span>
                <span className="apple-badge text-[10px] bg-blue-500/10 text-blue-600 font-mono">
                  {utterances.length} turns
                </span>
              </div>

              <div className="bg-[var(--apple-bg)] p-0.5 rounded-full border border-[var(--apple-border)] flex items-center text-[11px]">
                {(["ALL", "candidate", "interviewer"] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setFilterSpeaker(filter)}
                    className={`px-3 py-1 rounded-full transition-all ${
                      filterSpeaker === filter
                        ? "bg-[var(--apple-surface)] text-[var(--apple-text-primary)] font-bold shadow-sm"
                        : "text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
                    }`}
                  >
                    {filter === "ALL" ? "All" : filter.charAt(0).toUpperCase() + filter.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Scrollable Message Feed */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              {filteredUtterances.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-2 text-[var(--apple-text-secondary)]">
                  <span className="text-3xl">🎙️</span>
                  <p className="text-xs font-semibold">Awaiting first spoken utterance from meeting stream...</p>
                  <p className="text-[11px] text-[var(--apple-text-muted)] max-w-xs">
                    Audio chunks are routed to Speechmatics & Groq Whisper and stitched within 2500ms windows.
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
                      className={`p-3.5 rounded-2xl text-xs space-y-1.5 transition-all ${
                        isCandidate
                          ? "bg-[var(--apple-surface)] border border-[var(--apple-border)] shadow-sm ml-4"
                          : "bg-[var(--apple-bg)] border border-transparent mr-4"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`font-bold inline-flex items-center gap-1.5 ${
                            isCandidate ? "text-[var(--apple-accent)]" : "text-[var(--apple-text-secondary)]"
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${isCandidate ? "bg-[var(--apple-accent)]" : "bg-slate-400"}`}
                          ></span>
                          {isCandidate ? session?.candidate_name || "Candidate" : "Interviewer"} ({u.speaker_label})
                        </span>
                        <span className="text-[10px] text-[var(--apple-text-muted)] font-mono">{timeFormatted}</span>
                      </div>
                      <p className="text-[var(--apple-text-primary)] leading-relaxed text-[13px]">{u.text}</p>
                    </div>
                  );
                })
              )}
              <div ref={transcriptEndRef} />
            </div>
          </div>

          {/* Real-Time Advisory Follow-up Question Card */}
          {latestAdvisoryQuestion && (
            <div className="apple-card p-5 space-y-2 border-l-4 border-l-[var(--apple-accent)] bg-[var(--apple-accent-light)]">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-[var(--apple-accent)] uppercase tracking-wider flex items-center gap-1.5">
                  <span>💡</span> Live Recruiter Probe (LLM Judge Suggestion)
                </span>
                <span className="apple-badge text-[10px] font-mono">Real-time</span>
              </div>
              <p className="text-xs font-bold text-[var(--apple-text-primary)] leading-relaxed">
                &ldquo;{latestAdvisoryQuestion}&rdquo;
              </p>
              <p className="text-[11px] text-[var(--apple-text-secondary)]">
                Ask this question to verify if the candidate understands architectural depth or recited an AI summary.
              </p>
            </div>
          )}

          {/* Q&A Evaluations Feed */}
          <div className="apple-card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-3">
              <h2 className="text-xs font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider">
                Real-Time Q&A Evaluations ({qaEvaluations.length})
              </h2>
              <span className="text-[11px] text-[var(--apple-text-secondary)]">Filtered by Pre-Filter Gate</span>
            </div>

            {qaEvaluations.length === 0 ? (
              <div className="p-6 text-center text-xs text-[var(--apple-text-secondary)]">
                No complete candidate answers evaluated yet. Calibration occurs on Question 0.
              </div>
            ) : (
              <div className="space-y-4">
                {qaEvaluations.map((qa) => {
                  const score = Math.round(qa.combined_risk_score);
                  const isElevated = score >= 65;
                  const isModerate = score >= 35 && score < 65;

                  return (
                    <div
                      key={qa.qa_index}
                      className="p-4 rounded-xl border border-[var(--apple-border)] bg-[var(--apple-bg)] space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-extrabold text-[var(--apple-text-primary)]">
                          Question #{qa.qa_index + 1}
                        </span>
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs font-extrabold px-2 py-0.5 rounded-full ${
                              isElevated
                                ? "bg-rose-500/10 text-rose-600"
                                : isModerate
                                ? "bg-amber-500/10 text-amber-600"
                                : "bg-emerald-500/10 text-emerald-600"
                            }`}
                          >
                            Risk: {score}%
                          </span>
                        </div>
                      </div>

                      <div className="text-xs space-y-1">
                        <div className="font-semibold text-[var(--apple-text-secondary)]">
                          Q: {qa.question_text}
                        </div>
                        <div className="text-[var(--apple-text-primary)] line-clamp-2">
                          A: {qa.answer_text}
                        </div>
                      </div>

                      {/* Signals Pills */}
                      <div className="flex items-center gap-2 flex-wrap text-[10px]">
                        <span className="apple-badge font-mono">WPM: {Math.round(qa.heuristics.wpm)}</span>
                        {qa.heuristics.disfluency_drop > 0 && (
                          <span className="apple-badge font-mono text-amber-600 bg-amber-500/10">
                            Disfluency Drop: {qa.heuristics.disfluency_drop.toFixed(1)}%
                          </span>
                        )}
                        {qa.heuristics.ai_tell_matches?.length > 0 && (
                          <span className="apple-badge font-mono text-rose-600 bg-rose-500/10">
                            AI Tells: {qa.heuristics.ai_tell_matches.join(", ")}
                          </span>
                        )}
                        {qa.llm_verdict && (
                          <span className="apple-badge font-mono text-purple-600 bg-purple-500/10">
                            LLM Judge: {qa.llm_verdict.confidence_band} Confidence
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
          {/* Dynamic Rolling Risk Gauge Card */}
          <div className="apple-card p-6 text-center space-y-4">
            <span className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider block">
              Cumulative Authenticity Risk
            </span>

            <div className="relative inline-flex items-center justify-center">
              {/* Outer Circular Ring Visual */}
              <div
                className={`w-36 h-36 rounded-full border-8 flex flex-col items-center justify-center transition-all duration-500 shadow-inner ${
                  riskBand === "ELEVATED"
                    ? "border-rose-500 bg-rose-500/5 text-rose-600"
                    : riskBand === "MODERATE"
                    ? "border-amber-500 bg-amber-500/5 text-amber-600"
                    : "border-emerald-500 bg-emerald-500/5 text-emerald-600"
                }`}
              >
                <span className="text-4xl font-black tracking-tight">{Math.round(riskScore)}%</span>
                <span className="text-[11px] font-extrabold uppercase tracking-widest mt-1">
                  {riskBand}
                </span>
              </div>
            </div>

            <p className="text-xs text-[var(--apple-text-secondary)] max-w-xs mx-auto">
              {riskBand === "ELEVATED"
                ? "Substantial reliance on off-screen AI text or prolonged gaze deviations detected."
                : riskBand === "MODERATE"
                ? "Inconsistent disfluency or AI tell phrasing observed; check follow-up probes."
                : "Natural conversational formulation matches candidate's Q0 baseline."}
            </p>
          </div>

          {/* Q0 Baseline Calibration Card */}
          <div className="apple-card p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-2.5">
              <span className="text-xs font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider">
                Q0 Candidate Baseline
              </span>
              <span className="apple-badge text-[10px] bg-emerald-500/10 text-emerald-600 font-mono">
                {baseline ? "Calibrated" : "Calibrating..."}
              </span>
            </div>

            {baseline ? (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-2.5 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                  <span className="text-[10px] text-[var(--apple-text-secondary)] block">Median WPM</span>
                  <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                    {Math.round(baseline.median_wpm || baseline.baseline_wpm || 135)}
                  </span>
                </div>
                <div className="p-2.5 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                  <span className="text-[10px] text-[var(--apple-text-secondary)] block">Disfluency Rate</span>
                  <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                    {(baseline.baseline_disfluency_rate || 0).toFixed(1)} / 100w
                  </span>
                </div>
                <div className="p-2.5 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                  <span className="text-[10px] text-[var(--apple-text-secondary)] block">MAD Variance</span>
                  <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                    {(baseline.mad_wpm || 14).toFixed(1)}
                  </span>
                </div>
                <div className="p-2.5 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                  <span className="text-[10px] text-[var(--apple-text-secondary)] block">Pause Ratio</span>
                  <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                    {Math.round((baseline.baseline_pause_ratio || 0.12) * 100)}%
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-[var(--apple-text-secondary)]">
                Calibrating speech baseline against Question 0 (&ldquo;Tell me about yourself&rdquo;) to prevent ESL bias...
              </p>
            )}
          </div>

          {/* Vision Proctoring & Gaze Anomaly Monitor */}
          <div className="apple-card p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-2.5">
              <span className="text-xs font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider">
                Vision & Gaze Monitor (1 FPS)
              </span>
              <span className="apple-badge text-[10px] bg-sky-500/10 text-sky-600 font-mono">
                Debounced ≥3000ms
              </span>
            </div>

            {/* Video / Snapshot Preview Area */}
            <div className="w-full h-44 rounded-2xl bg-black flex flex-col items-center justify-center text-white relative overflow-hidden border border-[var(--apple-border)]">
              <div className="absolute top-3 left-3 flex items-center gap-1.5 apple-badge bg-black/60 text-white border border-white/20 text-[10px]">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 live-pulse"></span>
                <span>TELEMETRY ACTIVE</span>
              </div>

              <div className="text-center space-y-1">
                <span className="text-3xl block">👁️</span>
                <span className="text-xs font-bold">OpenCV & Haar Facial Tracking</span>
                <span className="text-[10px] text-slate-400 block">3-Tier Funnel: Zero GPU</span>
              </div>
            </div>

            {/* Vision Telemetry Status Indicators */}
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="p-2 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Gaze Offset</span>
                <span className="font-bold text-[var(--apple-text-primary)]">Center (±2°)</span>
              </div>
              <div className="p-2 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Head Pose</span>
                <span className="font-bold text-[var(--apple-text-primary)]">Normal</span>
              </div>
              <div className="p-2 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Faces in View</span>
                <span className="font-bold text-[var(--apple-text-primary)]">1 Face</span>
              </div>
            </div>

            {/* Vision Anomaly Timeline Feed */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-[var(--apple-text-secondary)] block">
                Confirmed Vision Flags ({visionEvents.length})
              </span>

              {visionEvents.length === 0 ? (
                <div className="p-3 bg-[var(--apple-bg)] rounded-xl text-center text-[11px] text-[var(--apple-text-secondary)]">
                  No sustained (&ge;3s) gaze or device anomalies detected.
                </div>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {visionEvents.map((evt) => (
                    <div
                      key={evt.event_id}
                      className="p-2.5 rounded-xl border border-rose-500/20 bg-rose-500/5 text-xs flex items-center justify-between"
                    >
                      <div className="space-y-0.5">
                        <span className="font-bold text-rose-600 block">{evt.anomaly_type}</span>
                        <span className="text-[10px] text-[var(--apple-text-secondary)]">
                          Sustained {Math.round(evt.duration_ms / 1000)}s
                          {evt.vlm_verification_note && ` • VLM: ${evt.vlm_verification_note}`}
                        </span>
                      </div>
                      <span className="apple-badge text-[10px] font-mono">
                        {Math.floor(evt.timestamp_ms / 1000)}s
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
