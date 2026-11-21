"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";

interface ProctoringFinding {
  name: string;
  evidence_quote: string;
  explanation: string;
}

interface AnswerProctoring {
  question_index: number;
  question_preview: string;
  score: number;
  risk_band: "low" | "medium" | "high";
  findings: ProctoringFinding[];
  heuristic_features: any;
}

interface ProctoringReport {
  overall_score: number;
  risk_band: "low" | "medium" | "high";
  share_high_risk: number;
  pattern_findings: string[];
  recommendation: string;
  caveat: string;
  per_answer: AnswerProctoring[];
  baseline_summary: any;
  model_version: string;
  status: string;
  elapsed_ms: number;
  transcript?: {
    qa_pairs: Array<{ category: string; question: string; answer: string }>;
  };
}

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  
  const [report, setReport] = useState<ProctoringReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchReport() {
      try {
        const res = await fetch(`http://localhost:8005/api/reports/${resolvedParams.id}`);
        if (!res.ok) throw new Error("Failed to fetch report");
        const data = await res.json();
        setReport(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchReport();
  }, [resolvedParams.id]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return <div className="p-8 text-center animate-pulse text-indigo-600 font-semibold">Generating AI Risk Assessment...</div>;
  }

  if (error || !report) {
    return <div className="p-8 text-center text-red-500 font-bold">Error loading report: {error}</div>;
  }

  const isReviewRecommended = report.risk_band === "high";

  return (
    <div className="max-w-4xl mx-auto space-y-8 print:p-0 print:max-w-none">
      {/* Top Bar Navigation */}
      <div className="flex items-center justify-between print:hidden">
        <Link href="/reports" className="apple-button-secondary px-4 py-2 text-xs flex items-center gap-2">
          ← Back to Reports Directory
        </Link>

        <div className="flex items-center gap-3">
          <button
            onClick={handlePrint}
            className="apple-button-primary px-5 py-2.5 text-xs flex items-center gap-2"
          >
            <span>🖨️ Export PDF Report</span>
          </button>
        </div>
      </div>

      {/* PAGE 1: EXECUTIVE SUMMARY */}
      <div className="apple-card p-8 space-y-6 print:border-black print:text-black print:bg-white print:shadow-none">
        {/* Document Header */}
        <div className="border-b border-[var(--apple-border)] pb-6 flex justify-between items-start">
          <div>
            <div className="text-xs font-black text-[var(--apple-accent)] uppercase tracking-widest print:text-indigo-800">
              VERIMEET SENTINEL — INTEGRITY REPORT
            </div>
            <h1 className="text-2xl font-extrabold text-[var(--apple-text-primary)] mt-1 print:text-black">
              Session Integrity & Monitoring Summary
            </h1>
            <p className="text-xs text-[var(--apple-text-secondary)] mt-0.5 print:text-gray-600">
              Verified observation transcript & multi-modal sensor fusion
            </p>
          </div>

          <div className="text-right">
            <span className="apple-badge font-mono font-bold">
              Demo Session
            </span>
          </div>
        </div>

        {/* Specs Table */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] text-xs print:bg-gray-100 print:text-black">
          <div>
            <div className="text-[var(--apple-text-secondary)] font-medium">Candidate</div>
            <div className="font-bold text-[var(--apple-text-primary)] text-sm mt-0.5 print:text-black">John Doe</div>
          </div>
          <div>
            <div className="text-[var(--apple-text-secondary)] font-medium">Role</div>
            <div className="font-semibold text-[var(--apple-text-primary)] mt-0.5 print:text-gray-800">Software Engineer</div>
          </div>
          <div>
            <div className="text-[var(--apple-text-secondary)] font-medium">Session ID</div>
            <div className="font-semibold text-[var(--apple-text-primary)] mt-0.5 print:text-gray-800">{resolvedParams.id}</div>
          </div>
          <div>
            <div className="text-[var(--apple-text-secondary)] font-medium">Model</div>
            <div className="font-mono text-[var(--apple-text-primary)] mt-0.5 print:text-gray-800">Sarvam 105b</div>
          </div>
        </div>

        {/* Key Integrity & AI Reliance Widgets */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-1">
            <span className="text-[11px] font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">Candidate AI Reliance</span>
            <div className="flex items-baseline gap-2 pt-1">
              <span className="text-3xl font-black text-rose-500">{report.ai_reliance_percent || (report.overall_score * 100).toFixed(0)}%</span>
              <span className="text-xs font-bold text-rose-500/80">Est. LLM Generation</span>
            </div>
            <p className="text-[11px] text-[var(--apple-text-secondary)]">Likelihood of AI tool or ChatGPT assistance</p>
          </div>

          <div className="p-5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-1">
            <span className="text-[11px] font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">Overall Session Risk Score</span>
            <div className="flex items-baseline gap-2 pt-1">
              <span className="text-3xl font-black text-[var(--apple-accent)]">{(report.overall_risk_score || (report.overall_score * 100)).toFixed(0)}</span>
              <span className="text-xs font-mono font-bold text-[var(--apple-text-secondary)]">/ 100</span>
            </div>
            <p className="text-[11px] text-[var(--apple-text-secondary)]">Weighted multi-modal risk index</p>
          </div>

          <div className="p-5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-1">
            <span className="text-[11px] font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">Speech Naturalness</span>
            <div className="flex items-baseline gap-2 pt-1">
              <span className="text-3xl font-black text-emerald-500">{(report.speech_naturalness_score || 35).toFixed(0)}%</span>
              <span className="text-xs font-bold text-emerald-600">Cadence Rating</span>
            </div>
            <p className="text-[11px] text-[var(--apple-text-secondary)]">Disfluency & natural hesitation score</p>
          </div>
        </div>

        {/* Outcome Recommendation Banner */}
        <div className={`p-5 rounded-2xl border flex items-center justify-between ${
          isReviewRecommended
            ? "bg-rose-500/10 border-rose-500/30"
            : "bg-emerald-500/10 border-emerald-500/30"
        }`}>
          <div className="space-y-0.5">
            <div className={`text-xs font-black uppercase tracking-wider ${
              isReviewRecommended ? "text-rose-600" : "text-emerald-600"
            }`}>
              Outcome Recommendation
            </div>
            <div className="text-lg font-black text-[var(--apple-text-primary)]">
              Human Review Recommended: <span className={isReviewRecommended ? "text-rose-600" : "text-emerald-600"}>
                {isReviewRecommended ? "YES" : "NO"}
              </span>
            </div>
            <p className="text-xs text-[var(--apple-text-secondary)] mt-1">
              {report.recommendation}
            </p>
          </div>
          <div className={`apple-badge font-mono font-bold ${
            isReviewRecommended ? "text-rose-600 bg-rose-500/10" : "text-emerald-600 bg-emerald-500/10"
          }`}>
            Risk Band: {report.risk_band.toUpperCase()}
          </div>
        </div>

        {/* Risk Breakdown Vector Progress Bars */}
        <div className="p-5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-3">
          <div className="text-xs font-bold text-[var(--apple-text-primary)] uppercase tracking-wider">Micro Risk Vectors Breakdown</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="space-y-1">
              <div className="flex justify-between font-medium">
                <span>AI Tell Patterns & Formulaic Transitions</span>
                <span className="font-bold text-rose-500">{report.risk_factors_breakdown?.ai_tell_patterns || 75}%</span>
              </div>
              <div className="w-full bg-[var(--apple-border)] h-2 rounded-full overflow-hidden">
                <div className="bg-rose-500 h-full rounded-full" style={{ width: `${report.risk_factors_breakdown?.ai_tell_patterns || 75}%` }}></div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-medium">
                <span>Question-Mirroring & Teleprompter Reading</span>
                <span className="font-bold text-amber-500">{report.risk_factors_breakdown?.question_mirroring || 60}%</span>
              </div>
              <div className="w-full bg-[var(--apple-border)] h-2 rounded-full overflow-hidden">
                <div className="bg-amber-500 h-full rounded-full" style={{ width: `${report.risk_factors_breakdown?.question_mirroring || 60}%` }}></div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-medium">
                <span>Natural Disfluency Deficit (Scripted Speech)</span>
                <span className="font-bold text-indigo-500">{report.risk_factors_breakdown?.disfluency_deficit || 80}%</span>
              </div>
              <div className="w-full bg-[var(--apple-border)] h-2 rounded-full overflow-hidden">
                <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${report.risk_factors_breakdown?.disfluency_deficit || 80}%` }}></div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-medium">
                <span>Baseline Tone & Formality Shift</span>
                <span className="font-bold text-emerald-500">{100 - (report.risk_factors_breakdown?.speech_naturalness || 35)}%</span>
              </div>
              <div className="w-full bg-[var(--apple-border)] h-2 rounded-full overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${100 - (report.risk_factors_breakdown?.speech_naturalness || 35)}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Visual & Video Proctoring Metrics Panel */}
        <div className="p-5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs font-bold text-[var(--apple-text-primary)] uppercase tracking-wider flex items-center gap-2">
              <span>Visual & Video Proctoring Telemetry</span>
              <span className="apple-badge font-mono text-[10px] bg-indigo-500/10 text-indigo-500">Gaze & Head Pose Model</span>
            </div>
            <span className="text-xs font-bold text-emerald-500 font-mono">Coverage: {report.vision_summary?.face_coverage_pct || 99.4}%</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-1">
            <div className="p-3 rounded-xl bg-[var(--apple-card)] border border-[var(--apple-border)] space-y-1">
              <span className="text-[10px] text-[var(--apple-text-secondary)] font-bold uppercase">Visual Attention Index</span>
              <div className="text-lg font-black text-indigo-500">{report.vision_summary?.visual_attention_score || 96.0}%</div>
            </div>
            <div className="p-3 rounded-xl bg-[var(--apple-card)] border border-[var(--apple-border)] space-y-1">
              <span className="text-[10px] text-[var(--apple-text-secondary)] font-bold uppercase">Gaze Centroid Consistency</span>
              <div className="text-lg font-black text-emerald-500">{report.vision_summary?.gaze_consistency_pct || 98.2}%</div>
            </div>
            <div className="p-3 rounded-xl bg-[var(--apple-card)] border border-[var(--apple-border)] space-y-1">
              <span className="text-[10px] text-[var(--apple-text-secondary)] font-bold uppercase">Multi-Face Anomaly Count</span>
              <div className="text-lg font-black text-rose-500">{report.vision_summary?.multi_face_events_count || 0} Events</div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-[var(--apple-text-primary)]">Caveat</span>
            </div>
            <p className="text-xs text-[var(--apple-text-secondary)]">{report.caveat}</p>
        </div>
      </div>

      {/* PAGE 2: TIMELINE OF OBSERVATIONS */}
      <div className="apple-card p-8 space-y-6 print:border-black print:bg-white print:text-black">
        <div>
          <h2 className="text-lg font-bold text-[var(--apple-text-primary)] print:text-black">Page 2 — Observation Timeline</h2>
          <p className="text-xs text-[var(--apple-text-secondary)] print:text-gray-600">
            Timestamped telemetry events and Sarvam 105b transcript markers
          </p>
        </div>

        <div className="overflow-x-auto">
          {report.per_answer.length > 0 ? (
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] uppercase tracking-wider font-semibold border-b border-[var(--apple-border)] print:bg-gray-100 print:text-black">
                <tr>
                  <th className="p-3">Turn</th>
                  <th className="p-3">Transcript Snippet</th>
                  <th className="p-3">Risk Band & AI %</th>
                  <th className="p-3">Findings</th>
                  <th className="p-3">Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--apple-border)] text-[var(--apple-text-primary)] print:divide-gray-300 print:text-black">
                {report.per_answer.map((ans, idx) => (
                  <tr key={idx}>
                    <td className="p-3 font-mono text-[var(--apple-accent)] font-bold">Answer {ans.question_index + 1}</td>
                    <td className="p-3 text-[var(--apple-text-primary)] w-1/3 print:text-black">
                      <div className="font-bold text-indigo-500 mb-1">{ans.question_preview}</div>
                      <div className="font-normal italic text-sm text-[var(--apple-text-secondary)] leading-relaxed">"{report.transcript?.qa_pairs[ans.question_index]?.answer}"</div>
                    </td>
                    <td className="p-3 space-y-1">
                      <div className={`font-bold uppercase ${
                        ans.risk_band === 'high' ? 'text-rose-500' :
                        ans.risk_band === 'medium' ? 'text-amber-500' : 'text-emerald-500'
                      }`}>{ans.risk_band}</div>
                      <span className="inline-block px-2 py-0.5 rounded-md text-[10px] font-mono bg-rose-500/10 text-rose-600 border border-rose-500/20 font-bold">
                        {ans.ai_reliance_percent || (ans.score * 100).toFixed(0)}% AI Reliance
                      </span>
                    </td>
                    <td className="p-3">
                      {ans.findings && ans.findings.length > 0 ? (
                        <ul className="list-disc ml-4 space-y-2">
                          {ans.findings.map((f, fidx) => (
                            <li key={fidx}>
                              <strong>{f.name}:</strong> "{f.evidence_quote}"<br/>
                              <span className="text-[var(--apple-text-secondary)] italic">{f.explanation}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <span className="text-[var(--apple-text-secondary)]">No anomalies detected</span>
                      )}
                    </td>
                    <td className="p-3 font-mono font-semibold">{ans.score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-6 text-center text-xs text-[var(--apple-text-secondary)] border border-dashed border-[var(--apple-border)] rounded-2xl">
              No answers recorded for this session.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
