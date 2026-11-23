"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { fetchSessions, SessionItem } from "@/lib/api";

export default function ReportsListPage() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterBand, setFilterBand] = useState<"ALL" | "ELEVATED" | "MODERATE" | "LOW" | "LIVE">("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchSessions();
        setSessions(data);
      } catch (err) {
        console.error("Failed to load reports:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredSessions = sessions.filter((s) => {
    // Search query filter
    const matchesSearch =
      s.candidate_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.job_title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.id.toLowerCase().includes(searchQuery.toLowerCase());

    if (!matchesSearch) return false;

    // Filter pill
    if (filterBand === "ALL") return true;
    if (filterBand === "LIVE") return s.status !== "COMPLETED" && s.status !== "FAILED";
    return (s.risk_band || "LOW").toUpperCase() === filterBand;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[var(--apple-border)] pb-6">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 apple-badge">
            <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
            OCULUS Audit Store
          </div>
          <h1 className="text-3xl font-extrabold text-[var(--apple-text-primary)] tracking-tight">
            Integrity Reports Directory
          </h1>
          <p className="text-xs text-[var(--apple-text-secondary)]">
            Explore and export multimodal authenticity audits across all interview sessions.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="bg-[var(--apple-bg)] p-1 rounded-full border border-[var(--apple-border)] flex items-center gap-1 flex-wrap">
          {[
            { id: "ALL", label: "All Sessions" },
            { id: "ELEVATED", label: "Elevated Risk" },
            { id: "MODERATE", label: "Moderate Risk" },
            { id: "LOW", label: "Low Risk" },
            { id: "LIVE", label: "Active" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilterBand(tab.id as any)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${
                filterBand === tab.id
                  ? "bg-[var(--apple-surface)] text-[var(--apple-text-primary)] shadow-sm font-bold"
                  : "text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Search Input Bar */}
      <div className="flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search candidate name, job title, or session ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2.5 apple-input text-xs pl-9"
          />
          <span className="absolute left-3 top-2.5 text-xs text-[var(--apple-text-muted)]">🔍</span>
        </div>

        <Link href="/sessions/new" className="apple-button-primary px-4 py-2 text-xs font-bold inline-flex items-center gap-1.5">
          <span>+ New Sentinel Session</span>
        </Link>
      </div>

      {/* Reports Grid / Table */}
      <div className="apple-card overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-xs text-[var(--apple-text-secondary)] animate-pulse">
            Loading audit records...
          </div>
        ) : filteredSessions.length === 0 ? (
          <div className="p-12 text-center space-y-2">
            <span className="text-3xl block">📋</span>
            <p className="text-sm font-semibold text-[var(--apple-text-primary)]">
              No reports match your current filter
            </p>
            <p className="text-xs text-[var(--apple-text-secondary)]">
              Try adjusting your search query or launch a new proctoring session.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] border-b border-[var(--apple-border)] uppercase tracking-wider font-semibold">
                <tr>
                  <th className="py-3 px-6">Candidate & Target Role</th>
                  <th className="py-3 px-6">Ingestion Mode</th>
                  <th className="py-3 px-6">Status</th>
                  <th className="py-3 px-6">Authenticity Risk</th>
                  <th className="py-3 px-6 text-right">Report Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--apple-border)]">
                {filteredSessions.map((s) => {
                  const isCompleted = s.status === "COMPLETED";
                  const riskBand = s.risk_band || "LOW";
                  const score = s.overall_risk_score !== null && s.overall_risk_score !== undefined
                    ? Math.round(s.overall_risk_score)
                    : null;

                  return (
                    <tr key={s.id} className="hover:bg-[var(--apple-bg)] transition-colors">
                      <td className="py-4 px-6">
                        <div className="font-extrabold text-[var(--apple-text-primary)] text-sm">
                          {s.candidate_name}
                        </div>
                        <div className="text-[var(--apple-text-secondary)] text-xs">
                          {s.job_title} • <span className="font-mono text-[10px]">{s.id}</span>
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
                            isCompleted
                              ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                              : "bg-amber-500/10 text-amber-600 border border-amber-500/20"
                          }`}
                        >
                          {!isCompleted && <span className="w-1.5 h-1.5 rounded-full bg-amber-500 live-pulse mr-1.5"></span>}
                          {s.status}
                        </span>
                      </td>

                      <td className="py-4 px-6">
                        {score !== null ? (
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
                              {score}%
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
                        </Link>

                        <Link
                          href={`/reports/${s.id}`}
                          className="apple-button-primary px-3 py-1.5 text-xs inline-flex items-center gap-1 font-bold"
                        >
                          <span>View Report →</span>
                        </Link>
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
  );
}
