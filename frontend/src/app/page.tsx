"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface SessionData {
  id: string;
  candidate: string;
  role: string;
  interviewer: string;
  platform: string;
  date: string;
  duration: string;
  language: string;
  status: string;
  statusBadge: string;
  statusLabel: string;
  coverage: string;
  observationsCount: number;
  reviewRecommended: boolean;
}

export default function Home() {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSessions() {
      try {
        const res = await fetch("http://localhost:8005/api/reports");
        if (res.ok) {
          const data = await res.json();
          setSessions(data.reverse());
        }
      } catch (err) {
        console.error("Failed to fetch home sessions:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchSessions();
  }, []);

  return (
    <div className="space-y-10">
      {/* Apple Hero Banner */}
      <div className="text-center max-w-3xl mx-auto space-y-4 pt-4 pb-2">
        <div className="inline-flex items-center gap-2 apple-badge">
          <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
          Sarvam AI Saaras v2 Speech Engine
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-[var(--apple-text-primary)]">
          Interview Integrity. <br />
          <span className="text-[var(--apple-accent)]">Reimagined for AI.</span>
        </h1>

        <p className="text-slate-500 dark:text-slate-400 text-base max-w-xl mx-auto leading-relaxed">
          Proctoring agent with background Speech-to-Text (STT) and Text-to-Speech (TTS) powered by <strong>Sarvam AI Saaras v2</strong> for English and Hindi.
        </p>

        <div className="pt-2 flex items-center justify-center gap-4">
          <Link
            href="/sessions/new"
            className="apple-button-primary px-7 py-3 text-sm inline-flex items-center gap-2"
          >
            <span>Dispatch Agent Bot →</span>
          </Link>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">Active Monitoring</span>
          <div className="flex items-center justify-between">
            <span className="text-3xl font-extrabold text-[var(--apple-text-primary)]">1 Session</span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 live-pulse"></span>
          </div>
          <span className="text-xs text-emerald-600 font-semibold">Bot live in meeting</span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">Speech Processing</span>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-[var(--apple-accent)]">Sarvam Saaras v2</span>
            <span className="apple-badge font-mono text-[11px]">STT / TTS</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">English (en-IN) & Hindi (hi-IN)</span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">Monitoring Coverage</span>
          <div className="flex items-center justify-between">
            <span className="text-3xl font-extrabold text-[var(--apple-text-primary)]">97.8%</span>
            <span className="text-xs font-bold text-emerald-600 bg-emerald-500/10 px-2 py-0.5 rounded-full">High</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">Continuous media tracking</span>
        </div>

        <div className="apple-card p-6 space-y-3 apple-card-interactive">
          <span className="text-xs font-semibold text-[var(--apple-text-secondary)] uppercase tracking-wider">Bot Architecture</span>
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-[var(--apple-text-primary)]">Zoom / Meet / Teams</span>
            <span className="text-xs text-[var(--apple-accent)] font-bold">Ready</span>
          </div>
          <span className="text-xs text-[var(--apple-text-secondary)]">Modular Adapter</span>
        </div>
      </div>

      {/* Sessions Table */}
      <div className="apple-card overflow-hidden">
        <div className="p-6 border-b border-[var(--apple-border)] flex items-center justify-between">
          <div>
            <h2 className="text-lg font-extrabold text-[var(--apple-text-primary)] tracking-tight">Interview Monitoring Sessions</h2>
            <p className="text-xs text-[var(--apple-text-secondary)]">Real-time Sarvam AI speech transcription and proctoring telemetry logs.</p>
          </div>
          <span className="apple-badge font-semibold">Sarvam Saaras v2 Active</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] text-xs uppercase tracking-wider font-semibold border-b border-[var(--apple-border)]">
              <tr>
                <th className="px-6 py-4 font-bold">Candidate & Role</th>
                <th className="px-6 py-4 font-bold">Interviewer</th>
                <th className="px-6 py-4 font-bold">Platform</th>
                <th className="px-6 py-4 font-bold">Transcription Language</th>
                <th className="px-6 py-4 font-bold">Status</th>
                <th className="px-6 py-4 font-bold">Coverage</th>
                <th className="px-6 py-4 font-bold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--apple-border)] text-[var(--apple-text-primary)]">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-[var(--apple-text-secondary)]">Loading sessions...</td>
                </tr>
              ) : sessions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-[var(--apple-text-secondary)]">No sessions dispatched yet. Dispatch a bot above to get started!</td>
                </tr>
              ) : (
                sessions.map((s) => (
                <tr key={s.id} className="hover:bg-[var(--apple-bg)]/50 transition-colors">
                  <td className="px-6 py-4 font-medium">
                    <div className="font-bold text-[var(--apple-text-primary)]">{s.candidate}</div>
                    <div className="text-xs text-[var(--apple-text-secondary)]">{s.role}</div>
                  </td>
                  <td className="px-6 py-4 font-medium">{s.interviewer}</td>
                  <td className="px-6 py-4">
                    <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[var(--apple-bg)] border border-[var(--apple-border)] text-[var(--apple-text-primary)]">
                      {s.platform}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className="apple-badge flex items-center gap-1.5 w-fit">
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--apple-accent)]"></span>
                      {s.language}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold ${s.statusBadge}`}>
                      {s.status === "RECORDING" && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse mr-1"></span>}
                      {s.statusLabel || s.status.replace("_", " ")}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-mono text-xs font-bold text-[var(--apple-text-primary)]">{s.coverage}</td>
                  <td className="px-6 py-4 text-right space-x-2">
                    {s.status === "RECORDING" ? (
                      <Link
                        href={`/sessions/${s.id}`}
                        className="apple-button-primary px-4 py-1.5 text-xs inline-block"
                      >
                        Live Monitor →
                      </Link>
                    ) : (
                      <Link
                        href={`/reports/${s.id}`}
                        className="apple-button-secondary px-4 py-1.5 text-xs inline-block"
                      >
                        View Report →
                      </Link>
                    )}
                  </td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
