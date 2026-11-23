"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { fetchSessions, SessionItem } from "@/lib/api";

export default function HomePage() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const data = await fetchSessions();
        setSessions(data);
      } catch (err) {
        console.error("Failed to load sessions:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const activeSessionsCount = sessions.filter(
    (s) => s.status !== "COMPLETED" && s.status !== "FAILED"
  ).length;

  return (
    <div className="space-y-12">
      {/* Hero Banner */}
      <div className="text-center max-w-3xl mx-auto space-y-4 pt-4 pb-2">
        <div className="inline-flex items-center gap-2 apple-badge">
          <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
          Multimodal AI Interview Integrity Sentinel
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-[var(--apple-text-primary)]">
          Interview Integrity. <br />
          <span className="text-[var(--apple-accent)]">Reimagined for the AI Era.</span>
        </h1>

        <p className="text-[var(--apple-text-secondary)] text-base max-w-2xl mx-auto leading-relaxed">
          Detects hidden generative AI teleprompters, speech synthesis, and off-screen devices
          using <strong>Hybrid STT</strong>, <strong>Q0 baseline calibration</strong>, and a{" "}
          <strong>two-stage reasoning judge</strong> without invasive desktop spyware.
        </p>

        <div className="pt-3 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/sessions/new"
            className="apple-button-primary px-8 py-3.5 text-sm inline-flex items-center gap-2 shadow-lg"
          >
            <span>+ Launch New Sentinel</span>
            <span>→</span>
          </Link>
          <Link
            href="/reports"
            className="apple-button-secondary px-6 py-3.5 text-sm inline-flex items-center gap-2"
          >
            <span>View Audit Reports</span>
          </Link>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">
            Active Monitoring
          </span>
          <div className="flex items-center justify-between">
            <span className="text-3xl font-extrabold text-[var(--apple-text-primary)]">
              {activeSessionsCount} {activeSessionsCount === 1 ? "Session" : "Sessions"}
            </span>
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                activeSessionsCount > 0 ? "bg-emerald-500 live-pulse" : "bg-slate-400"
              }`}
            ></span>
          </div>
          <span className="text-xs text-emerald-600 font-semibold">
            {activeSessionsCount > 0 ? "Workers live on Redis streams" : "Ready for dispatch"}
          </span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">
            Hybrid STT Router
          </span>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-[var(--apple-accent)]">Speechmatics / Groq</span>
            <span className="apple-badge font-mono text-[11px]">Diarized</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">
            Custom technical dictionary injection
          </span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">
            Vision Proctoring
          </span>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-sky-600 dark:text-sky-400">1 FPS Debounced</span>
            <span className="apple-badge font-mono text-[11px]">Zero-GPU</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">
            Gaze Euler angles + Gemini Flash VLM
          </span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">
            Reasoning Judge
          </span>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">Llama 3.3 70B</span>
            <span className="apple-badge font-mono text-[11px]">Fairness Gate</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">
            Q0 Baseline (Median/MAD normalized)
          </span>
        </div>
      </div>

      {/* 3 Ingestion Modes Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-[var(--apple-text-primary)] tracking-tight">
            Select Proctoring Ingestion Mode
          </h2>
          <span className="text-xs text-[var(--apple-text-secondary)]">Zero-cost free tiers enabled</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Link
            href="/sessions/new?mode=demo_file"
            className="apple-card p-6 space-y-3 apple-card-interactive block border-l-4 border-l-blue-500"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">📁</span>
              <span className="apple-badge font-mono text-[10px] bg-blue-500/10 text-blue-600">
                $0 Free Demo
              </span>
            </div>
            <h3 className="font-extrabold text-base text-[var(--apple-text-primary)]">
              Demo File Replay
            </h3>
            <p className="text-xs text-[var(--apple-text-secondary)] leading-relaxed">
              Upload an interview recording (.mp4, .webm, .wav). FFmpeg splits audio & 1 FPS video frames
              directly into the worker mesh.
            </p>
            <span className="text-xs text-[var(--apple-accent)] font-bold inline-flex items-center gap-1">
              Upload Mock Video →
            </span>
          </Link>

          <Link
            href="/sessions/new?mode=browser_webrtc"
            className="apple-card p-6 space-y-3 apple-card-interactive block border-l-4 border-l-emerald-500"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">🌐</span>
              <span className="apple-badge font-mono text-[10px] bg-emerald-500/10 text-emerald-600">
                $0 Zero-Key Demo
              </span>
            </div>
            <h3 className="font-extrabold text-base text-[var(--apple-text-primary)]">
              Browser Tab Capture
            </h3>
            <p className="text-xs text-[var(--apple-text-secondary)] leading-relaxed">
              Open Google Meet or Zoom in a tab. Share tab audio & video directly through your browser
              without needing external bots.
            </p>
            <span className="text-xs text-emerald-600 font-bold inline-flex items-center gap-1">
              Start WebRTC Tab Stream →
            </span>
          </Link>

          <Link
            href="/sessions/new?mode=attendee_bot"
            className="apple-card p-6 space-y-3 apple-card-interactive block border-l-4 border-l-purple-500"
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">🤖</span>
              <span className="apple-badge font-mono text-[10px] bg-purple-500/10 text-purple-600">
                Live Cloud Bot
              </span>
            </div>
            <h3 className="font-extrabold text-base text-[var(--apple-text-primary)]">
              Meeting Bot Dispatch
            </h3>
            <p className="text-xs text-[var(--apple-text-secondary)] leading-relaxed">
              Dispatch a headless Attendee bot named <em>OCULUS Sentinel</em> into any live Zoom, Google Meet,
              or MS Teams URL.
            </p>
            <span className="text-xs text-purple-600 font-bold inline-flex items-center gap-1">
              Dispatch In-Call Bot →
            </span>
          </Link>
        </div>
      </div>

      {/* Sessions Directory Table */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-[var(--apple-text-primary)] tracking-tight">
              Interview Sessions Directory
            </h2>
            <p className="text-xs text-[var(--apple-text-secondary)]">
              Live status, calibrated risk scores, and audit reports
            </p>
          </div>
          <Link href="/reports" className="text-xs font-semibold text-[var(--apple-accent)] hover:underline">
            View All Reports →
          </Link>
        </div>

        <div className="apple-card overflow-hidden">
          {loading ? (
            <div className="p-12 text-center text-xs text-[var(--apple-text-secondary)] animate-pulse">
              Loading sessions from Sentinel API...
            </div>
          ) : sessions.length === 0 ? (
            <div className="p-12 text-center space-y-3">
              <span className="text-3xl">🎯</span>
              <p className="text-sm font-semibold text-[var(--apple-text-primary)]">
                No interview sessions created yet
              </p>
              <p className="text-xs text-[var(--apple-text-secondary)] max-w-sm mx-auto">
                Launch your first session with a sample demo file, browser tab capture, or Zoom bot.
              </p>
              <Link
                href="/sessions/new"
                className="apple-button-primary px-5 py-2 text-xs font-bold inline-block mt-2"
              >
                + Launch Sentinel Session
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] border-b border-[var(--apple-border)] uppercase tracking-wider font-semibold">
                  <tr>
                    <th className="py-3 px-6">Candidate & Role</th>
                    <th className="py-3 px-6">Ingestion Mode</th>
                    <th className="py-3 px-6">Status</th>
                    <th className="py-3 px-6">Risk Score</th>
                    <th className="py-3 px-6 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--apple-border)]">
                  {sessions.map((s) => {
                    const isCompleted = s.status === "COMPLETED";
                    const isLive = s.status === "IN_CALL" || s.status === "PROCESSING" || s.status === "WAITING_IN_LOBBY";
                    const riskBand = s.risk_band || "LOW";

                    return (
                      <tr key={s.id} className="hover:bg-[var(--apple-bg)] transition-colors">
                        <td className="py-4 px-6">
                          <div className="font-extrabold text-[var(--apple-text-primary)] text-sm">
                            {s.candidate_name}
                          </div>
                          <div className="text-[var(--apple-text-secondary)] text-xs">
                            {s.job_title}
                          </div>
                        </td>

                        <td className="py-4 px-6">
                          <span className="apple-badge font-mono text-[11px]">
                            {s.ingestion_mode === "demo_file"
                              ? "📁 Demo File"
                              : s.ingestion_mode === "browser_webrtc"
                              ? "🌐 WebRTC Tab"
                              : "🤖 Attendee Bot"}
                          </span>
                        </td>

                        <td className="py-4 px-6">
                          <span
                            className={`apple-badge font-semibold text-[11px] ${
                              isLive
                                ? "bg-amber-500/10 text-amber-600 border border-amber-500/20"
                                : isCompleted
                                ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                                : "bg-slate-500/10 text-slate-600"
                            }`}
                          >
                            {isLive && <span className="w-1.5 h-1.5 rounded-full bg-amber-500 live-pulse mr-1.5"></span>}
                            {s.status}
                          </span>
                        </td>

                        <td className="py-4 px-6">
                          {s.overall_risk_score !== null && s.overall_risk_score !== undefined ? (
                            <div className="flex items-center gap-2">
                              <span
                                className={`text-sm font-extrabold ${
                                  riskBand === "ELEVATED"
                                    ? "text-rose-600"
                                    : riskBand === "MODERATE"
                                    ? "text-amber-600"
                                    : "text-emerald-600"
                                }`}
                              >
                                {Math.round(s.overall_risk_score)}%
                              </span>
                              <span
                                className={`apple-badge text-[10px] font-bold ${
                                  riskBand === "ELEVATED"
                                    ? "bg-rose-500/10 text-rose-600"
                                    : riskBand === "MODERATE"
                                    ? "bg-amber-500/10 text-amber-600"
                                    : "bg-emerald-500/10 text-emerald-600"
                                }`}
                              >
                                {riskBand}
                              </span>
                            </div>
                          ) : (
                            <span className="text-[var(--apple-text-muted)] text-xs">Calibrating...</span>
                          )}
                        </td>

                        <td className="py-4 px-6 text-right space-x-2">
                          <Link
                            href={`/sessions/${s.id}`}
                            className="apple-button-secondary px-3 py-1.5 text-xs inline-flex items-center gap-1 font-semibold"
                          >
                            <span>Live Monitor</span>
                            <span>→</span>
                          </Link>

                          {isCompleted && (
                            <Link
                              href={`/reports/${s.id}`}
                              className="apple-button-primary px-3 py-1.5 text-xs inline-flex items-center gap-1 font-bold"
                            >
                              <span>Report</span>
                            </Link>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
