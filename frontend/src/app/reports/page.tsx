"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

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

export default function ReportsListPage() {
  const router = useRouter();
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
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
        console.error("Failed to fetch sessions:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchSessions();
  }, []);

  const filteredSessions = sessions.filter((s) => {
    if (filterStatus === "ALL") return true;
    if (filterStatus === "READY") return s.status === "REPORT_READY";
    if (filterStatus === "REVIEW") return s.status === "NEEDS_REVIEW";
    if (filterStatus === "LIVE") return s.status === "RECORDING";
    return true;
  });

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[var(--apple-border)] pb-6">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 apple-badge">
            <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
            VeriMeet Audit Store
          </div>
          <h1 className="text-3xl font-extrabold text-[var(--apple-text-primary)] tracking-tight">
            Integrity Reports Directory
          </h1>
          <p className="text-xs text-[var(--apple-text-secondary)]">
            Select an interview session below to view its multi-modal proctoring timeline and Sarvam AI transcript log.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="bg-[var(--apple-bg)] p-1 rounded-full border border-[var(--apple-border)] flex items-center gap-1">
          {[
            { id: "ALL", label: "All Sessions" },
            { id: "READY", label: "Ready" },
            { id: "REVIEW", label: "Needs Review" },
            { id: "LIVE", label: "Live" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilterStatus(tab.id)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${
                filterStatus === tab.id
                  ? "bg-[var(--apple-surface)] text-[var(--apple-text-primary)] shadow-sm font-bold"
                  : "text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Session Cards List */}
      <div className="grid grid-cols-1 gap-4">
        {loading ? (
          <div className="p-8 text-center text-[var(--apple-text-secondary)]">Loading sessions...</div>
        ) : filteredSessions.length === 0 ? (
          <div className="p-8 text-center text-[var(--apple-text-secondary)]">No sessions found matching this filter.</div>
        ) : (
          filteredSessions.map((s) => (
          <div
            key={s.id}
            onClick={() => {
              if (s.status === "RECORDING") {
                router.push(`/sessions/${s.id}`);
              } else {
                router.push(`/reports/${s.id}`);
              }
            }}
            className="apple-card p-6 apple-card-interactive cursor-pointer flex flex-col md:flex-row justify-between items-start md:items-center gap-4"
          >
            <div className="space-y-2 flex-1">
              <div className="flex items-center gap-3 flex-wrap">
                <span className={s.statusBadge}>
                  {s.status === "RECORDING" && <span className="w-1.5 h-1.5 rounded-full bg-blue-500 live-pulse mr-1"></span>}
                  {s.statusLabel}
                </span>

                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[var(--apple-bg)] border border-[var(--apple-border)] text-[var(--apple-text-primary)]">
                  {s.platform}
                </span>

                <span className="apple-badge font-mono text-[11px]">
                  {s.language}
                </span>

                <span className="font-mono text-xs text-[var(--apple-text-muted)]">ID: {s.id}</span>
              </div>

              <div>
                <h3 className="text-lg font-bold text-[var(--apple-text-primary)] tracking-tight">
                  {s.candidate} — <span className="text-sm font-medium text-[var(--apple-text-secondary)]">{s.role}</span>
                </h3>
                <p className="text-xs text-[var(--apple-text-secondary)] pt-0.5">
                  Interviewer: <strong className="text-[var(--apple-text-primary)]">{s.interviewer}</strong> | Date: {s.date} ({s.duration})
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6 self-end md:self-center">
              <div className="text-right">
                <div className="text-xs text-[var(--apple-text-secondary)]">Coverage</div>
                <div className="font-mono font-extrabold text-sm text-[var(--apple-text-primary)]">{s.coverage}</div>
              </div>

              <div className="text-right">
                <div className="text-xs text-[var(--apple-text-secondary)]">Observations</div>
                <div className="font-mono font-bold text-sm text-[var(--apple-text-primary)]">
                  {s.observationsCount} {s.observationsCount === 1 ? "flag" : "flags"}
                </div>
              </div>

              <div>
                {s.status === "RECORDING" ? (
                  <span className="apple-button-primary px-4 py-2 text-xs inline-block">
                    Live Monitor →
                  </span>
                ) : (
                  <span className="apple-button-secondary px-4 py-2 text-xs inline-block">
                    Open Report →
                  </span>
                )}
              </div>
            </div>
          </div>
        )))}
      </div>
    </div>
  );
}
