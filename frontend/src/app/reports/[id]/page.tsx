"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { fetchReport } from "@/lib/api";

interface WordTiming {
  word: string;
  start_ms: number;
  end_ms: number;
}

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
        // Expand all QAs by default
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
      <div className="p-16 text-center space-y-3">
        <div className="w-8 h-8 border-4 border-[var(--apple-accent)] border-t-transparent rounded-full animate-spin mx-auto"></div>
        <p className="text-sm font-bold text-[var(--apple-text-primary)]">
          Generating AI Authenticity Report...
        </p>
        <p className="text-xs text-[var(--apple-text-secondary)]">
          Aggregating Q0 baseline calibration, 0ms heuristics, and LLM reasoning verdicts...
        </p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-2xl mx-auto p-8 apple-card text-center space-y-4">
        <span className="text-4xl block">⚠️</span>
        <h2 className="text-lg font-bold text-[var(--apple-text-primary)]">Report Not Ready</h2>
        <p className="text-xs text-[var(--apple-text-secondary)]">{error || "No report found for this session."}</p>
        <div className="pt-2 flex justify-center gap-3">
          <Link href={`/sessions/${sessionId}`} className="apple-button-primary px-5 py-2 text-xs font-bold">
            ← Return to Live Monitor
          </Link>
          <Link href="/reports" className="apple-button-secondary px-5 py-2 text-xs font-semibold">
            View All Reports
          </Link>
        </div>
      </div>
    );
  }

  const score = Math.round(report.overall_risk_score);
  const band = report.risk_band || "LOW";

  return (
    <div className="max-w-4xl mx-auto space-y-8 print:p-0 print:max-w-none print:space-y-4">
      {/* Top Bar Navigation (Hidden in Print) */}
      <div className="flex items-center justify-between print:hidden">
        <Link href="/reports" className="apple-button-secondary px-4 py-2 text-xs font-semibold inline-flex items-center gap-1.5">
          ← Back to Reports Directory
        </Link>

        <div className="flex items-center gap-3">
          <Link href={`/sessions/${sessionId}`} className="apple-button-secondary px-4 py-2 text-xs font-semibold">
            Open Live Monitor
          </Link>
          <button
            onClick={handlePrint}
            className="apple-button-primary px-5 py-2 text-xs font-bold inline-flex items-center gap-2"
          >
            <span>🖨️ Export PDF Report</span>
          </button>
        </div>
      </div>

      {/* Mandatory Advisory Legal Disclaimer Banner */}
      <div className="p-4 bg-amber-500/10 border border-amber-500/25 rounded-2xl text-xs text-amber-800 dark:text-amber-300 space-y-1">
        <div className="font-extrabold flex items-center gap-1.5 uppercase tracking-wider text-[11px]">
          <span>⚖️</span> Mandatory Advisory Disclaimer
        </div>
        <p className="leading-relaxed">
          {report.advisory_disclaimer ||
            "Advisory behavioral inference derived from meeting-plane audio and video signals. Not operating-system telemetry and not definitive proof of misconduct. Use as an advisory signal for human interviewers."}
        </p>
      </div>

      {/* Executive Summary Card */}
      <div className="apple-card p-8 space-y-6 print:border print:shadow-none print:bg-white print:text-black">
        <div className="border-b border-[var(--apple-border)] pb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="space-y-1">
            <div className="text-xs font-black text-[var(--apple-accent)] uppercase tracking-widest">
              OCULUS SENTINEL — INTEGRITY AUDIT
            </div>
            <h1 className="text-2xl font-black text-[var(--apple-text-primary)]">
              {report.candidate_name}
            </h1>
            <p className="text-xs text-[var(--apple-text-secondary)] font-medium">
              Target Role: <strong>{report.job_title}</strong> • Session ID: {report.session_id}
            </p>
          </div>

          {/* Risk Score Badge */}
          <div className="text-right">
            <div
              className={`inline-flex flex-col items-center justify-center px-6 py-3 rounded-2xl border ${
                band === "ELEVATED"
                  ? "bg-rose-500/10 border-rose-500/20 text-rose-600"
                  : band === "MODERATE"
                  ? "bg-amber-500/10 border-amber-500/20 text-amber-600"
                  : "bg-emerald-500/10 border-emerald-500/20 text-emerald-600"
              }`}
            >
              <span className="text-3xl font-black tracking-tight">{score}%</span>
              <span className="text-[10px] font-extrabold uppercase tracking-widest mt-0.5">
                {band} RISK
              </span>
            </div>
          </div>
        </div>

        {/* Q0 Baseline Summary */}
        {report.baseline && (
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
              1. Candidate Baseline Calibration (Question 0)
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Median WPM</span>
                <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                  {Math.round(report.baseline.median_wpm)}
                </span>
              </div>
              <div className="p-3 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">MAD Variance</span>
                <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                  {report.baseline.mad_wpm.toFixed(1)}
                </span>
              </div>
              <div className="p-3 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Disfluency Rate</span>
                <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                  {report.baseline.baseline_disfluency_rate.toFixed(1)}%
                </span>
              </div>
              <div className="p-3 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)]">
                <span className="text-[10px] text-[var(--apple-text-secondary)] block">Pause Ratio</span>
                <span className="text-base font-extrabold text-[var(--apple-text-primary)]">
                  {Math.round(report.baseline.baseline_pause_ratio * 100)}%
                </span>
              </div>
            </div>
            <p className="text-[11px] text-[var(--apple-text-muted)] italic">
              All cadence and disfluency metrics below are measured as deltas relative to this candidate&apos;s natural
              Q0 formulation.
            </p>
          </div>
        )}

        {/* Per-Answer Evaluations */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
              2. Evaluated Q&A Authenticity Turns ({report.qa_evaluations?.length || 0})
            </h2>
            <span className="text-[11px] text-[var(--apple-text-secondary)]">
              Stage 1 Deterministic + Stage 2 LLM Judge
            </span>
          </div>

          <div className="space-y-4">
            {report.qa_evaluations?.map((qa) => {
              const qaScore = Math.round(qa.combined_risk_score);
              const isExpanded = expandedQA[qa.qa_index] ?? true;

              return (
                <div
                  key={qa.qa_index}
                  className="rounded-2xl border border-[var(--apple-border)] bg-[var(--apple-bg)] p-5 space-y-3"
                >
                  <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => toggleQA(qa.qa_index)}
                  >
                    <div className="space-y-0.5">
                      <span className="text-xs font-extrabold text-[var(--apple-text-primary)] block">
                        Question #{qa.qa_index + 1}
                      </span>
                      <span className="text-xs text-[var(--apple-text-secondary)] font-medium line-clamp-1">
                        {qa.question_text}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs font-extrabold px-2.5 py-0.5 rounded-full ${
                          qaScore >= 65
                            ? "bg-rose-500/10 text-rose-600 border border-rose-500/20"
                            : qaScore >= 35
                            ? "bg-amber-500/10 text-amber-600 border border-amber-500/20"
                            : "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                        }`}
                      >
                        Risk: {qaScore}%
                      </span>
                      <span className="text-xs text-[var(--apple-text-muted)]">{isExpanded ? "▲" : "▼"}</span>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="pt-3 border-t border-[var(--apple-border)] space-y-3 text-xs">
                      {/* Full Q & A */}
                      <div className="space-y-1.5 p-3 rounded-xl bg-[var(--apple-surface)] border border-[var(--apple-border)]">
                        <div className="font-bold text-[var(--apple-text-secondary)]">Q: {qa.question_text}</div>
                        <div className="text-[var(--apple-text-primary)] leading-relaxed">A: {qa.answer_text}</div>
                      </div>

                      {/* Heuristics Matrix */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                        <div className="p-2 bg-[var(--apple-surface)] rounded-lg border border-[var(--apple-border)]">
                          <span className="text-[10px] text-[var(--apple-text-muted)] block">Speaking Rate</span>
                          <span className="font-bold text-[var(--apple-text-primary)]">
                            {Math.round(qa.heuristics.wpm)} WPM ({qa.heuristics.wpm_robust_zscore > 0 ? "+" : ""}
                            {qa.heuristics.wpm_robust_zscore.toFixed(1)}σ)
                          </span>
                        </div>
                        <div className="p-2 bg-[var(--apple-surface)] rounded-lg border border-[var(--apple-border)]">
                          <span className="text-[10px] text-[var(--apple-text-muted)] block">Disfluency Delta</span>
                          <span
                            className={`font-bold ${
                              qa.heuristics.disfluency_drop > 2.0 ? "text-amber-600" : "text-[var(--apple-text-primary)]"
                            }`}
                          >
                            {qa.heuristics.disfluency_drop > 0 ? "-" : "+"}
                            {Math.abs(qa.heuristics.disfluency_drop).toFixed(1)}% vs Q0
                          </span>
                        </div>
                        <div className="p-2 bg-[var(--apple-surface)] rounded-lg border border-[var(--apple-border)]">
                          <span className="text-[10px] text-[var(--apple-text-muted)] block">First-Person Density</span>
                          <span className="font-bold text-[var(--apple-text-primary)]">
                            {(qa.heuristics.first_person_ratio * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="p-2 bg-[var(--apple-surface)] rounded-lg border border-[var(--apple-border)]">
                          <span className="text-[10px] text-[var(--apple-text-muted)] block">Pre-Filter Decision</span>
                          <span className="font-bold text-[var(--apple-text-primary)]">
                            {qa.heuristics.trigger_llm_judge ? "Escalated to LLM" : "Bypassed LLM ($0)"}
                          </span>
                        </div>
                      </div>

                      {/* AI Tell Phrases Hits */}
                      {qa.heuristics.ai_tell_matches?.length > 0 && (
                        <div className="p-2.5 bg-rose-500/5 rounded-xl border border-rose-500/20 text-xs text-rose-700 dark:text-rose-300">
                          <span className="font-bold block mb-1">⚠️ LLM Tell Phrases Matched:</span>
                          <div className="flex gap-1.5 flex-wrap">
                            {qa.heuristics.ai_tell_matches.map((phrase, i) => (
                              <span key={i} className="apple-badge bg-rose-500/10 text-rose-600 font-mono text-[10px]">
                                &ldquo;{phrase}&rdquo;
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* LLM Judge Reasoning */}
                      {qa.llm_verdict && (
                        <div className="p-3 rounded-xl bg-purple-500/5 border border-purple-500/20 space-y-2 text-xs">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-purple-700 dark:text-purple-300 uppercase tracking-wider text-[10px]">
                              LLM Reasoning Verdict ({qa.llm_verdict.confidence_band} Confidence)
                            </span>
                            <span className="font-mono text-[10px] text-purple-600">
                              Suspicion Score: {Math.round(qa.llm_verdict.suspicion_score)}%
                            </span>
                          </div>

                          <p className="text-[var(--apple-text-primary)] leading-relaxed">
                            {qa.llm_verdict.recitation_vs_live_assessment}
                          </p>

                          {qa.llm_verdict.advisory_follow_up_question && (
                            <div className="p-2 bg-[var(--apple-surface)] rounded-lg border border-purple-500/20 text-xs">
                              <span className="font-bold text-[var(--apple-accent)] block mb-0.5">
                                Suggested Verification Probe:
                              </span>
                              <span className="italic text-[var(--apple-text-primary)]">
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
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
              3. Confirmed Vision & Gaze Anomalies ({report.vision_timeline.length})
            </h2>
            <div className="space-y-2">
              {report.vision_timeline.map((vt) => (
                <div
                  key={vt.event_id}
                  className="p-3 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)] flex items-center justify-between text-xs"
                >
                  <div className="space-y-0.5">
                    <span className="font-bold text-rose-600 block">{vt.anomaly_type}</span>
                    <span className="text-[11px] text-[var(--apple-text-secondary)]">
                      Sustained duration: {Math.round(vt.duration_ms / 1000)}s
                      {vt.vlm_verification_note && ` • VLM: ${vt.vlm_verification_note}`}
                    </span>
                  </div>
                  <span className="apple-badge font-mono text-[10px] text-[var(--apple-text-muted)]">
                    At {Math.floor(vt.timestamp_ms / 1000)}s
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
