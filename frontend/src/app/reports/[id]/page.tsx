"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { fetchReport } from "@/lib/api";

interface QAEvaluation {
  qa_index: number;
  question_text: string;
  answer_text: string;
  question_end_ms: number;
  answer_start_ms: number;
  answer_end_ms: number;
  combined_risk_score: number;
  heuristics: {
    word_count: number;
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
    confidence_band: "LOW" | "MEDIUM" | "HIGH";
    reasoning_signals: string[];
    verbatim_evidence_quotes?: string[];
    recitation_vs_live_assessment: string;
    advisory_follow_up_question: string;
  } | null;
}

interface VisionAnomaly {
  event_id: string;
  anomaly_type: string;
  timestamp_ms: number;
  duration_ms: number;
  confidence: number;
  pitch_deg?: number;
  yaw_deg?: number;
  evidence_url?: string;
  vlm_verification_note?: string;
}

interface IntegrityReportData {
  session_id: string;
  candidate_name: string;
  job_title: string;
  overall_risk_score: number;
  risk_band: "LOW" | "MODERATE" | "ELEVATED";
  advisory_disclaimer: string;
  baseline?: {
    session_id: string;
    median_wpm: number;
    mad_wpm: number;
    baseline_disfluency_rate: number;
    baseline_pause_ratio: number;
  } | null;
  qa_evaluations: QAEvaluation[];
  vision_timeline: VisionAnomaly[];
  generated_at_iso?: string;
}

export default function ReportDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const sessionId = resolvedParams.id;

  const [report, setReport] = useState<IntegrityReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedQA, setExpandedQA] = useState<Record<number, boolean>>({});

  useEffect(() => {
    async function loadReport() {
      try {
        const data = await fetchReport(sessionId);
        setReport(data);
        if (data.qa_evaluations) {
          const map: Record<number, boolean> = {};
          data.qa_evaluations.forEach((_: any, idx: number) => {
            map[idx] = true;
          });
          setExpandedQA(map);
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadReport();
  }, [sessionId]);

  const handlePrint = () => {
    window.print();
  };

  const toggleQA = (idx: number) => {
    setExpandedQA((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  if (loading) {
    return (
      <div className="p-20 text-center space-y-4">
        <div className="font-mono text-2xl text-[var(--oculus-ink)] opacity-40 animate-pulse">◫</div>
        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-[var(--oculus-ink)]">
          SYNCHRONIZING AUDIT LEDGER...
        </p>
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
          Aggregating Q0 baseline calibration, 0ms heuristics, and LLM reasoning verdicts
        </p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-2xl mx-auto p-8 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] text-center space-y-4">
        <div className="font-mono text-xs text-[#E12D1E] tracking-[0.24em] uppercase">// AUDIT NOT FOUND</div>
        <h2 className="font-display text-xl font-semibold text-[var(--oculus-ink)]">Report File Unindexed</h2>
        <p className="font-mono text-[11px] text-[var(--oculus-muted)] uppercase tracking-[0.16em]">
          {error || "No compiled report available for this session identifier."}
        </p>
        <div className="pt-4 flex justify-center gap-3">
          <Link
            href={`/sessions/${sessionId}`}
            className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-5 py-2.5 font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-bg)]"
          >
            ← LIVE SENTINEL
          </Link>
          <Link
            href="/reports"
            className="border border-[var(--oculus-hairline)] bg-transparent px-5 py-2.5 font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-ink)]"
          >
            ARCHIVE DIRECTORY
          </Link>
        </div>
      </div>
    );
  }

  const score = Math.round(report.overall_risk_score);
  const band = report.risk_band || "LOW";
  const isElevated = band === "ELEVATED";

  return (
    <div className="max-w-5xl mx-auto space-y-8 print:p-0 print:max-w-none print:space-y-4">
      {/* Top Bar Navigation (Hidden in Print) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--oculus-hairline)] pb-5 print:hidden">
        <Link
          href="/reports"
          className="font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-muted)] hover:text-[var(--oculus-ink)] transition-colors inline-flex items-center gap-2"
        >
          <span>←</span>
          <span>INDEX // 02 AUDIT DIRECTORY</span>
        </Link>

        <div className="flex items-center gap-3">
          <Link
            href={`/sessions/${sessionId}`}
            className="border border-[var(--oculus-hairline)] bg-transparent px-4 py-2 font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-ink)] hover:border-[var(--oculus-ink)] transition-colors"
          >
            LIVE MONITOR
          </Link>
          <button
            onClick={handlePrint}
            className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-5 py-2 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-bg)] hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] transition-colors"
          >
            EXPORT PDF REPORT ↗
          </button>
        </div>
      </div>

      {/* Mandatory Advisory Disclaimer Banner */}
      <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-5 space-y-2">
        <div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.28em] text-[var(--oculus-muted)]">
          <div className="flex items-center gap-2 text-[#E12D1E]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#E12D1E]" />
            <span>MANDATORY ADVISORY SPECIFICATION</span>
          </div>
          <span>SIGNAL PHYSICS COMPLIANCE</span>
        </div>
        <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] leading-relaxed text-[var(--oculus-muted)]">
          {report.advisory_disclaimer ||
            "Behavioral inference derived strictly from meeting-plane audio and video signals. Not operating system spyware telemetry and not definitive proof of candidate misconduct. Surface strictly as advisory signal for human decision-makers."}
        </p>
      </div>

      {/* Executive Summary Card */}
      <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 sm:p-8 space-y-6">
        <div className="border-b border-[var(--oculus-hairline)] pb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="space-y-1">
            <span className="font-mono text-[9.5px] uppercase tracking-[0.3em] text-[var(--oculus-muted)] block">
              OCULUS SENTINEL // MULTIMODAL INTEGRITY AUDIT
            </span>
            <h1 className="font-display text-3xl font-semibold tracking-[-0.03em] text-[var(--oculus-ink)]">
              {report.candidate_name}
            </h1>
            <p className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-[var(--oculus-muted)]">
              TARGET ROLE: <strong className="text-[var(--oculus-ink)]">{report.job_title}</strong> · ID: {report.session_id}
            </p>
          </div>

          {/* Risk Score Pill */}
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4 text-right min-w-[140px]">
            <span className="font-mono text-[9px] uppercase tracking-[0.24em] text-[var(--oculus-muted)] block">
              CALIBRATED RISK
            </span>
            <div className={`font-display text-4xl font-semibold tracking-tight ${isElevated ? "text-[#E12D1E]" : "text-[var(--oculus-ink)]"}`}>
              {score}%
            </div>
            <span className="font-mono text-[9.5px] font-bold uppercase tracking-[0.24em] opacity-80 block mt-0.5">
              {band}
            </span>
          </div>
        </div>

        {/* Q0 Baseline Summary */}
        {report.baseline && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                SEC. 01
              </span>
              <h2 className="font-display text-base font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                CANDIDATE BASELINE CALIBRATION (QUESTION 0)
              </h2>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4">
                <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-muted)] block">
                  MEDIAN CADENCE
                </span>
                <span className="font-display text-2xl font-semibold text-[var(--oculus-ink)] tabular-nums">
                  {Math.round(report.baseline.median_wpm)}
                </span>
                <span className="font-mono text-[9px] text-[var(--oculus-muted)] block uppercase">WPM NOMINAL</span>
              </div>

              <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4">
                <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-muted)] block">
                  MAD VARIANCE
                </span>
                <span className="font-display text-2xl font-semibold text-[var(--oculus-ink)] tabular-nums">
                  {report.baseline.mad_wpm.toFixed(1)}
                </span>
                <span className="font-mono text-[9px] text-[var(--oculus-muted)] block uppercase">σ DISPERSION</span>
              </div>

              <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4">
                <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-muted)] block">
                  DISFLUENCY RATIO
                </span>
                <span className="font-display text-2xl font-semibold text-[var(--oculus-ink)] tabular-nums">
                  {report.baseline.baseline_disfluency_rate.toFixed(1)}%
                </span>
                <span className="font-mono text-[9px] text-[var(--oculus-muted)] block uppercase">NATURAL HESITATION</span>
              </div>

              <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4">
                <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-muted)] block">
                  PAUSE RATIO
                </span>
                <span className="font-display text-2xl font-semibold text-[var(--oculus-ink)] tabular-nums">
                  {Math.round(report.baseline.baseline_pause_ratio * 100)}%
                </span>
                <span className="font-mono text-[9px] text-[var(--oculus-muted)] block uppercase">COGNITIVE LATENCY</span>
              </div>
            </div>

            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--oculus-muted)]">
              All cadence acceleration spikes and disfluency drops below are calculated relative to this candidate&apos;s natural baseline.
            </p>
          </div>
        )}

        {/* Per-Answer Evaluations */}
        <div className="space-y-4 pt-4 border-t border-[var(--oculus-hairline)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                SEC. 02
              </span>
              <h2 className="font-display text-base font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                EVALUATED Q&A AUTHENTICITY TURNS ({report.qa_evaluations?.length || 0})
              </h2>
            </div>
            <span className="font-mono text-[9.5px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
              STAGE 1 HEURISTICS + STAGE 2 LLM JUDGE
            </span>
          </div>

          <div className="space-y-4">
            {report.qa_evaluations?.map((qa) => {
              const qaScore = Math.round(qa.combined_risk_score);
              const isExpanded = expandedQA[qa.qa_index] ?? true;
              const isTurnElevated = qaScore >= 65;

              return (
                <div
                  key={qa.qa_index}
                  className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-5 space-y-4"
                >
                  <div
                    className="flex items-center justify-between cursor-pointer select-none"
                    onClick={() => toggleQA(qa.qa_index)}
                  >
                    <div className="space-y-1">
                      <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-muted)]">
                        TURN #{String(qa.qa_index + 1).padStart(2, "0")} // INTERROGATION
                      </div>
                      <div className="font-display text-sm font-semibold text-[var(--oculus-ink)] line-clamp-1">
                        {qa.question_text}
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span
                        className={`font-mono text-[10px] font-bold px-2.5 py-1 border tabular-nums ${
                          isTurnElevated
                            ? "border-[#E12D1E] text-[#E12D1E]"
                            : qaScore >= 35
                            ? "border-[var(--oculus-ink)] text-[var(--oculus-ink)]"
                            : "border-[var(--oculus-hairline)] text-[var(--oculus-muted)]"
                        }`}
                      >
                        RISK: {qaScore}%
                      </span>
                      <span className="font-mono text-[10px] text-[var(--oculus-muted)]">
                        {isExpanded ? "[ − ]" : "[ + ]"}
                      </span>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="pt-4 border-t border-[var(--oculus-hairline)] space-y-4 text-xs">
                      {/* Full Q & A */}
                      <div className="space-y-3 p-4 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] font-mono text-[11px]">
                        <div>
                          <span className="text-[var(--oculus-muted)] font-bold uppercase tracking-[0.16em] block mb-1">
                            Q // INTERVIEWER INQUIRY:
                          </span>
                          <span className="text-[var(--oculus-ink)] leading-relaxed">{qa.question_text}</span>
                        </div>
                        <div className="pt-2 border-t border-[var(--oculus-hairline)]">
                          <span className="text-[var(--oculus-muted)] font-bold uppercase tracking-[0.16em] block mb-1">
                            A // CANDIDATE RESPONSE TRANSCRIPTION:
                          </span>
                          <span className="text-[var(--oculus-ink)] leading-relaxed">{qa.answer_text}</span>
                        </div>
                      </div>

                      {/* Heuristics Matrix */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-[10px]">
                        <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]">
                          <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">SPEAKING RATE</span>
                          <span className="font-bold text-[var(--oculus-ink)] text-sm tabular-nums mt-0.5 block">
                            {Math.round(qa.heuristics.wpm)} WPM
                          </span>
                          <span className="text-[var(--oculus-muted)] text-[9px] tabular-nums">
                            {qa.heuristics.wpm_robust_zscore > 0 ? "+" : ""}
                            {qa.heuristics.wpm_robust_zscore.toFixed(1)}σ vs Q0
                          </span>
                        </div>

                        <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]">
                          <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">DISFLUENCY DELTA</span>
                          <span
                            className={`font-bold text-sm tabular-nums mt-0.5 block ${
                              qa.heuristics.disfluency_drop > 2.0 ? "text-[#E12D1E]" : "text-[var(--oculus-ink)]"
                            }`}
                          >
                            {qa.heuristics.disfluency_drop > 0 ? "−" : "+"}
                            {Math.abs(qa.heuristics.disfluency_drop).toFixed(1)}%
                          </span>
                          <span className="text-[var(--oculus-muted)] text-[9px]">NATURAL RATE DROP</span>
                        </div>

                        <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]">
                          <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">FIRST-PERSON RATIO</span>
                          <span className="font-bold text-[var(--oculus-ink)] text-sm tabular-nums mt-0.5 block">
                            {(qa.heuristics.first_person_ratio * 100).toFixed(1)}%
                          </span>
                          <span className="text-[var(--oculus-muted)] text-[9px]">PRONOUN DENSITY</span>
                        </div>

                        <div className="p-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)]">
                          <span className="text-[var(--oculus-muted)] block uppercase tracking-[0.18em]">STAGE 2 PRE-FILTER</span>
                          <span className="font-bold text-[var(--oculus-ink)] text-sm mt-0.5 block">
                            {qa.heuristics.trigger_llm_judge ? "ESCALATED" : "BYPASS ($0)"}
                          </span>
                          <span className="text-[var(--oculus-muted)] text-[9px]">REASONING GATE</span>
                        </div>
                      </div>

                      {/* AI Tell Phrases Hits */}
                      {qa.heuristics.ai_tell_matches?.length > 0 && (
                        <div className="p-3 border border-[#E12D1E]/40 bg-[#E12D1E]/5 font-mono text-[10px] space-y-1.5">
                          <span className="font-bold text-[#E12D1E] tracking-[0.2em] block">
                            ⚠ LLM RECITATION PHRASES MATCHED:
                          </span>
                          <div className="flex gap-2 flex-wrap">
                            {qa.heuristics.ai_tell_matches.map((phrase, i) => (
                              <span key={i} className="border border-[#E12D1E] px-2 py-0.5 text-[#E12D1E] uppercase">
                                &ldquo;{phrase}&rdquo;
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* LLM Judge Reasoning */}
                      {qa.llm_verdict && (
                        <div className="p-4 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] space-y-3 font-mono text-[11px]">
                          <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-2">
                            <span className="font-bold uppercase tracking-[0.22em] text-[var(--oculus-ink)]">
                              LLM REASONING VERDICT ({qa.llm_verdict.confidence_band} CONFIDENCE)
                            </span>
                            <span className="tabular-nums text-[#E12D1E]">
                              SUSPICION: {Math.round(qa.llm_verdict.suspicion_score)}%
                            </span>
                          </div>

                          <p className="leading-relaxed text-[var(--oculus-muted)] uppercase tracking-[0.08em]">
                            {qa.llm_verdict.recitation_vs_live_assessment}
                          </p>

                          {qa.llm_verdict.advisory_follow_up_question && (
                            <div className="p-3 border border-[var(--oculus-ink)]/20 bg-[var(--oculus-bg)] space-y-1">
                              <span className="font-bold text-[var(--oculus-ink)] uppercase tracking-[0.18em] block text-[10px]">
                                ADVISORY VERIFICATION PROBE (STAGE 2):
                              </span>
                              <span className="italic text-[var(--oculus-ink)] block">
                                &ldquo;{qa.llm_verdict.advisory_follow_up_question}&rdquo;
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Vision Timeline Section */}
        {report.vision_timeline && report.vision_timeline.length > 0 && (
          <div className="space-y-4 pt-4 border-t border-[var(--oculus-hairline)]">
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                SEC. 03
              </span>
              <h2 className="font-display text-base font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                CONFIRMED OPTICAL & GAZE ANOMALIES ({report.vision_timeline.length})
              </h2>
            </div>

            <div className="space-y-2">
              {report.vision_timeline.map((vt) => (
                <div
                  key={vt.event_id}
                  className="p-3.5 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.16em]"
                >
                  <div className="space-y-0.5">
                    <span className="font-bold text-[#E12D1E] block">{vt.anomaly_type}</span>
                    <span className="text-[var(--oculus-muted)]">
                      SUSTAINED DEFLECTION: {Math.round(vt.duration_ms / 1000)}S
                      {vt.vlm_verification_note && ` · VLM: ${vt.vlm_verification_note}`}
                    </span>
                  </div>
                  <span className="border border-[var(--oculus-hairline)] px-2 py-1 tabular-nums text-[var(--oculus-muted)]">
                    AT {Math.floor(vt.timestamp_ms / 1000)}S
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
