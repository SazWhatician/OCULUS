"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { fetchSessions, SessionItem } from "@/lib/api";

type RiskFilter = "ALL" | "ELEVATED" | "MODERATE" | "LOW" | "LIVE";

export default function ReportsListPage() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterBand, setFilterBand] = useState<RiskFilter>("ALL");
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
    const matchesSearch =
      s.candidate_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.job_title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.id.toLowerCase().includes(searchQuery.toLowerCase());

    if (!matchesSearch) return false;

    if (filterBand === "ALL") return true;
    if (filterBand === "LIVE") return s.status !== "COMPLETED" && s.status !== "FAILED";
    return (s.risk_band || "LOW").toUpperCase() === filterBand;
  });

  const formatMode = (m?: string) => {
    switch (m) {
      case "browser_webrtc":
        return "WEBRTC // STREAM";
      case "attendee_bot":
        return "BOT // ATTENDEE";
      case "demo_file":
      default:
        return "REPLAY // LOCAL";
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Top Header */}
      <div className="border-b border-[var(--oculus-hairline)] pb-6 space-y-3">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--oculus-muted)] hover:text-[var(--oculus-ink)] transition-colors inline-flex items-center gap-2"
          >
            <span>←</span>
            <span>INDEX // 01 DASHBOARD</span>
          </Link>
          <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-[var(--oculus-muted)] border border-[var(--oculus-hairline)] px-2.5 py-1">
            ARCHIVE LEDGER // SEC_02
          </span>
        </div>

        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="space-y-1">
            <h1 className="font-display text-3xl font-semibold tracking-[-0.03em] text-[var(--oculus-ink)] sm:text-4xl">
              AUDIT ARCHIVE DIRECTORY
            </h1>
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--oculus-muted)] max-w-2xl">
              Multimodal behavioral inference ledgers, Q0 baseline deltas, and LLM reasoning verdicts.
            </p>
          </div>

          <Link
            href="/sessions/new"
            className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-6 py-3 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-bg)] transition-colors hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] inline-flex items-center gap-2 self-start md:self-auto"
          >
            <span>+ INITIALIZE SENTINEL</span>
            <span>→</span>
          </Link>
        </div>
      </div>

      {/* Filter and Search Command Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Risk Category Filters */}
        <div className="inline-flex border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] flex-wrap">
          {(
            [
              { id: "ALL", label: "ALL SESSIONS" },
              { id: "ELEVATED", label: "ELEVATED RISK" },
              { id: "MODERATE", label: "MODERATE RISK" },
              { id: "LOW", label: "LOW RISK" },
              { id: "LIVE", label: "ACTIVE MONITOR" },
            ] as const
          ).map((tab) => {
            const active = filterBand === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setFilterBand(tab.id)}
                className={`px-4 py-2.5 font-mono text-[10px] uppercase tracking-[0.2em] transition-colors ${
                  active
                    ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)]"
                    : "text-[var(--oculus-muted)] hover:text-[var(--oculus-ink)]"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Search Bar */}
        <div className="relative w-full lg:w-80">
          <input
            type="text"
            placeholder="SEARCH CANDIDATE / ROLE / ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] px-4 py-2.5 font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
          />
        </div>
      </div>

      {/* Directory Table */}
      <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] overflow-hidden">
        {loading ? (
          <div className="divide-y divide-[var(--oculus-hairline)]">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="p-6 grid grid-cols-12 items-center gap-4 animate-pulse">
                <div className="col-span-4 space-y-2">
                  <div className="h-3.5 w-44 bg-[var(--oculus-ink)]/10" />
                  <div className="h-2.5 w-28 bg-[var(--oculus-ink)]/5" />
                </div>
                <div className="col-span-3 h-3 w-32 bg-[var(--oculus-ink)]/10" />
                <div className="col-span-3 h-3 w-24 bg-[var(--oculus-ink)]/10" />
                <div className="col-span-2 ml-auto h-7 w-20 bg-[var(--oculus-ink)]/10" />
              </div>
            ))}
          </div>
        ) : filteredSessions.length === 0 ? (
          <div className="px-6 py-24 text-center space-y-4">
            <div className="font-mono text-2xl text-[var(--oculus-ink)] opacity-30">◫</div>
            <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-[var(--oculus-ink)]">
              NO AUDIT RECORDS FOUND // FILTER EMPTY
            </p>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--oculus-muted)] max-w-md mx-auto">
              Adjust search query or deploy a new sentinel session to populate the ledger.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] font-mono text-[9.5px] uppercase tracking-[0.26em] text-[var(--oculus-muted)]">
                  <th className="px-6 py-4 sm:px-8">CANDIDATE // SPECIFICATION</th>
                  <th className="px-6 py-4">INGESTION MODE</th>
                  <th className="px-6 py-4">SENTINEL STATUS</th>
                  <th className="px-6 py-4">CALIBRATED RISK</th>
                  <th className="px-6 py-4 text-right sm:px-8">DISPATCH ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--oculus-hairline)]">
                {filteredSessions.map((s) => {
                  const isCompleted = s.status === "COMPLETED";
                  const isLive = s.status !== "COMPLETED" && s.status !== "FAILED";
                  const riskBand = s.risk_band || "LOW";
                  const score =
                    s.overall_risk_score !== null && s.overall_risk_score !== undefined
                      ? Math.round(s.overall_risk_score)
                      : null;
                  const isElevated = riskBand === "ELEVATED";

                  return (
                    <tr
                      key={s.id}
                      className="group transition-colors duration-200 hover:bg-[var(--oculus-ink)] hover:text-[var(--oculus-bg)]"
                    >
                      <td className="px-6 py-5 sm:px-8">
                        <div className="font-display text-[15px] font-semibold tracking-[-0.015em]">
                          {s.candidate_name}
                        </div>
                        <div className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--oculus-muted)] group-hover:text-[var(--oculus-bg)]/80">
                          {s.job_title} · <span className="opacity-60">{s.id.slice(0, 8)}</span>
                        </div>
                      </td>

                      <td className="px-6 py-5 font-mono text-[10px] tracking-[0.18em] text-[var(--oculus-muted)] group-hover:text-[var(--oculus-bg)]/80">
                        {formatMode(s.ingestion_mode)}
                      </td>

                      <td className="px-6 py-5">
                        <span className="inline-flex items-center gap-2 border border-current/25 px-2.5 py-1 font-mono text-[9.5px] uppercase tracking-[0.2em]">
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              isLive
                                ? "animate-pulse bg-[#E12D1E]"
                                : isCompleted
                                ? "bg-current"
                                : "bg-current opacity-40"
                            }`}
                          />
                          {s.status.replace(/_/g, " ")}
                        </span>
                      </td>

                      <td className="px-6 py-5">
                        {score !== null ? (
                          <div className="flex items-center gap-4">
                            <div className="w-28 space-y-1.5">
                              <div className="flex items-baseline justify-between font-mono text-[11px] tabular-nums">
                                <span
                                  className={`font-semibold ${
                                    isElevated ? "text-[#E12D1E]" : "text-current"
                                  }`}
                                >
                                  {String(score).padStart(2, "0")}%
                                </span>
                                <span className="text-[9px] tracking-[0.2em] opacity-60">
                                  {riskBand}
                                </span>
                              </div>
                              <div className="relative h-[3px] w-full bg-current/15">
                                <div
                                  style={{ width: `${Math.min(score, 100)}%` }}
                                  className={`h-full transition-all duration-700 ${
                                    isElevated ? "bg-[#E12D1E]" : "bg-current"
                                  }`}
                                />
                              </div>
                            </div>
                          </div>
                        ) : (
                          <span className="font-mono text-[10px] uppercase tracking-[0.22em] opacity-50">
                            CALIBRATING Q0...
                          </span>
                        )}
                      </td>

                      <td className="px-6 py-5 text-right sm:px-8">
                        <div className="inline-flex items-center gap-2">
                          <Link
                            href={`/sessions/${s.id}`}
                            className="inline-flex items-center gap-2 border border-[var(--oculus-ink)]/25 px-3.5 py-2 font-mono text-[9.5px] uppercase tracking-[0.22em] transition-colors group-hover:border-[var(--oculus-bg)]/40 group-hover:text-[var(--oculus-bg)] hover:!bg-[var(--oculus-bg)] hover:!text-[var(--oculus-ink)]"
                          >
                            <span>MONITOR</span>
                            <span>→</span>
                          </Link>

                          {isCompleted && (
                            <Link
                              href={`/reports/${s.id}`}
                              className="inline-flex items-center border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-3.5 py-2 font-mono text-[9.5px] uppercase tracking-[0.22em] text-[var(--oculus-bg)] transition-colors group-hover:border-[var(--oculus-bg)] group-hover:bg-[var(--oculus-bg)] group-hover:text-[var(--oculus-ink)]"
                            >
                              REPORT
                            </Link>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-6 py-4 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-muted)] sm:px-8">
          <span className="tabular-nums">
            INDEXED AUDIT SESSIONS: {String(filteredSessions.length).padStart(2, "0")} /{" "}
            {String(sessions.length).padStart(2, "0")}
          </span>
          <span className="text-[var(--oculus-ink)]">
            POLL CADENCE: NOMINAL (5000MS)
          </span>
        </footer>
      </div>
    </div>
  );
}
