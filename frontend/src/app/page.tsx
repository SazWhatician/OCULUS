"use client";

import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import Link from "next/link";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { fetchSessions, SessionItem } from "@/lib/api";
import {
  OpticalFieldCanvas,
  OculusMark,
  CipherReadout,
  HarmonicWaveform,
  IngestionSchematic,
  GazeTelemetry,
} from "@/components/OculusOptics";

gsap.registerPlugin(ScrollTrigger);

type FilterTab = "ALL" | "LIVE" | "COMPLETED";

export default function HomePage() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeFilter, setActiveFilter] = useState<FilterTab>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [hoveredMetric, setHoveredMetric] = useState<number | null>(null);
  const [scrollRatio, setScrollRatio] = useState(0);
  const [clock, setClock] = useState("00:00:00 UTC");
  const [gaze, setGaze] = useState<GazeTelemetry>({
    yaw: 0,
    pitch: 0,
    dilation: 1,
    velocity: 0,
  });

  const rootRef = useRef<HTMLDivElement>(null);

  const loadData = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);
    try {
      const data = await fetchSessions();
      setSessions(data);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setLoading(false);
      if (manual) setTimeout(() => setRefreshing(false), 400);
    }
  }, []);

  useEffect(() => {
    loadData();
    const poll = setInterval(() => loadData(false), 5000);
    return () => clearInterval(poll);
  }, [loadData]);

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setClock(now.toISOString().slice(11, 19) + " UTC");
    };
    updateClock();
    const t = setInterval(updateClock, 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const onScroll = () => {
      const max = Math.max(window.innerHeight, 1);
      setScrollRatio(Math.min(window.scrollY / max, 1));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(".hero-stagger", {
        y: 28,
        opacity: 0,
        duration: 0.95,
        stagger: 0.08,
        ease: "power4.out",
      });

      gsap.from(".hero-rule", {
        scaleX: 0,
        transformOrigin: "left center",
        duration: 1.1,
        stagger: 0.1,
        ease: "expo.out",
      });

      gsap.utils.toArray<HTMLElement>(".arch-reveal").forEach((el) => {
        gsap.from(el, {
          y: 32,
          opacity: 0,
          duration: 0.9,
          ease: "power3.out",
          scrollTrigger: {
            trigger: el,
            start: "top 86%",
            once: true,
          },
        });
      });

      gsap.utils.toArray<HTMLElement>(".draw-rule").forEach((el) => {
        gsap.from(el, {
          scaleX: 0,
          transformOrigin: "left center",
          duration: 1,
          ease: "expo.out",
          scrollTrigger: {
            trigger: el,
            start: "top 90%",
            once: true,
          },
        });
      });
    }, rootRef);

    return () => ctx.revert();
  }, []);

  const isLiveStatus = (status: string) =>
    status === "IN_CALL" ||
    status === "PROCESSING" ||
    status === "WAITING_IN_LOBBY";

  const activeSessionsCount = useMemo(
    () =>
      sessions.filter(
        (s) => s.status !== "COMPLETED" && s.status !== "FAILED"
      ).length,
    [sessions]
  );

  const filteredSessions = useMemo(() => {
    return sessions.filter((s) => {
      const matchesTab =
        activeFilter === "ALL"
          ? true
          : activeFilter === "LIVE"
          ? isLiveStatus(s.status)
          : s.status === "COMPLETED";

      const q = searchQuery.trim().toLowerCase();
      const matchesQuery =
        !q ||
        s.candidate_name.toLowerCase().includes(q) ||
        s.job_title.toLowerCase().includes(q) ||
        s.status.toLowerCase().includes(q);

      return matchesTab && matchesQuery;
    });
  }, [sessions, activeFilter, searchQuery]);

  const metrics = [
    {
      index: "01",
      label: "ACTIVE BUS",
      value: String(activeSessionsCount).padStart(2, "0"),
      unit: activeSessionsCount === 1 ? "SESSION" : "SESSIONS",
      spec:
        activeSessionsCount > 0
          ? "REDIS STREAMS NOMINAL"
          : "WORKERS ON STANDBY",
      tag: activeSessionsCount > 0 ? "LIVE" : "IDLE",
      freq: 2.8,
    },
    {
      index: "02",
      label: "HYBRID STT",
      value: "SPEECHMATICS",
      unit: "/ GROQ",
      spec: "DIARIZED LEXICON INJECTION",
      tag: "AUDIO",
      freq: 3.4,
    },
    {
      index: "03",
      label: "VISION MESH",
      value: "1.0 FPS",
      unit: "DEBOUNCED",
      spec: "EULER GAZE + GEMINI FLASH VLM",
      tag: "ZERO-GPU",
      freq: 1.9,
    },
    {
      index: "04",
      label: "REASONING JUDGE",
      value: "LLAMA 3.3",
      unit: "70B",
      spec: "Q0 MEDIAN / MAD FAIRNESS GATE",
      tag: "AUDIT",
      freq: 4.1,
    },
  ];

  const modes = [
    {
      num: "01",
      href: "/sessions/new?mode=demo_file",
      code: "FFMPEG / LOCAL",
      tier: "$0 SANDBOX",
      title: "Demo File Replay",
      body: "Ingest recorded interview media (.mp4, .webm, .wav). FFmpeg demuxes dual audio channels and 1 FPS keyframes straight into the worker mesh.",
      action: "INITIATE UPLOAD",
      variant: "demo" as const,
    },
    {
      num: "02",
      href: "/sessions/new?mode=browser_webrtc",
      code: "WEBRTC / CLIENT",
      tier: "$0 ZERO-KEY",
      title: "Browser Tab Capture",
      body: "Bind directly to a live Google Meet or Zoom tab inside the browser. Zero external credentials, zero desktop spyware, deterministic capture.",
      action: "OPEN TAB STREAM",
      variant: "webrtc" as const,
    },
    {
      num: "03",
      href: "/sessions/new?mode=attendee_bot",
      code: "HEADLESS / CLOUD",
      tier: "LIVE DISPATCH",
      title: "Meeting Bot Dispatch",
      body: "Deploy an autonomous OCULUS Sentinel participant into any active Zoom, Google Meet, or Microsoft Teams conference URL.",
      action: "DISPATCH SENTINEL",
      variant: "bot" as const,
    },
  ];

  const formatMode = (mode: string) => {
    if (mode === "demo_file") return "01 / DEMO FILE";
    if (mode === "browser_webrtc") return "02 / WEBRTC TAB";
    return "03 / CLOUD BOT";
  };

  const formatSign = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(1)}`;

  return (
    <div
      ref={rootRef}
      className="relative min-h-screen bg-[#FBFBF9] text-[#0A0A0A] selection:bg-[#0A0A0A] selection:text-[#FBFBF9]"
    >
      <OpticalFieldCanvas />

      <div className="relative z-10 mx-auto max-w-[1380px] border-x border-[#0A0A0A]/12 bg-[#FBFBF9]/80 backdrop-blur-[2px]">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[#0A0A0A]/12 px-5 py-3.5 font-mono text-[10px] uppercase tracking-[0.26em] text-[#525252] sm:px-8">
          <div className="flex items-center gap-3">
            <span className="inline-block h-2 w-2 bg-[#0A0A0A]" />
            <span className="font-semibold text-[#0A0A0A]">OCULUS // SENTINEL</span>
            <span className="hidden text-[#0A0A0A]/30 sm:inline">|</span>
            <span className="hidden sm:inline">MULTIMODAL INTEGRITY ENGINE</span>
          </div>

          <div className="flex flex-wrap items-center gap-5 tabular-nums">
            <span className="hidden md:inline">
              YAW {formatSign(gaze.yaw)}°
            </span>
            <span className="hidden md:inline">
              PITCH {formatSign(gaze.pitch)}°
            </span>
            <span className="hidden lg:inline">
              IRIS {gaze.dilation.toFixed(2)}X
            </span>
            <span className="text-[#0A0A0A]">{clock}</span>
          </div>
        </header>

        <section className="relative border-b border-[#0A0A0A]/12">
          <div className="grid grid-cols-1 lg:grid-cols-12">
            <div className="flex flex-col justify-between border-b border-[#0A0A0A]/12 p-6 sm:p-10 lg:col-span-7 lg:border-b-0 lg:border-r lg:p-14">
              <div className="space-y-8">
                <div className="hero-stagger flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.3em] text-[#525252]">
                  <span className="inline-flex h-2 w-2 items-center justify-center">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#0A0A0A]" />
                  </span>
                  <CipherReadout text="OPTICAL TELEMETRY // ZERO-SPYWARE ARCHITECTURE" />
                </div>

                <h1 className="hero-stagger font-display text-[clamp(2.8rem,6.5vw,5.6rem)] font-semibold leading-[0.94] tracking-[-0.045em] text-[#0A0A0A]">
                  INTERVIEW
                  <br />
                  INTEGRITY.
                  <br />
                  <span className="font-light italic tracking-[-0.03em] text-[#525252]">
                    CALIBRATED.
                  </span>
                </h1>

                <div className="hero-rule h-px w-full bg-[#0A0A0A]/15" />

                <p className="hero-stagger max-w-xl text-[15px] leading-[1.7] text-[#404040] sm:text-[16px]">
                  Exposes hidden generative teleprompters, voice synthesis, and
                  off-screen cognitive assistance via{" "}
                  <strong className="font-semibold text-[#0A0A0A]">
                    Hybrid STT
                  </strong>
                  ,{" "}
                  <strong className="font-semibold text-[#0A0A0A]">
                    Q0 Median/MAD baseline calibration
                  </strong>
                  , and a{" "}
                  <strong className="font-semibold text-[#0A0A0A]">
                    two-stage reasoning judge
                  </strong>
                  —operating purely on multimodal signal physics.
                </p>
              </div>

              <div className="hero-stagger mt-12 flex flex-wrap items-center gap-3">
                <Link
                  href="/sessions/new"
                  className="group relative inline-flex items-center gap-4 overflow-hidden border border-[#0A0A0A] bg-[#0A0A0A] px-7 py-4 font-mono text-[11px] font-medium uppercase tracking-[0.24em] text-[#FBFBF9] transition-colors duration-300 hover:bg-[#FBFBF9] hover:text-[#0A0A0A]"
                >
                  <span>+ LAUNCH SENTINEL</span>
                  <span className="transition-transform duration-300 group-hover:translate-x-1">
                    →
                  </span>
                </Link>

                <Link
                  href="/reports"
                  className="group inline-flex items-center gap-3 border border-[#0A0A0A]/25 bg-transparent px-7 py-4 font-mono text-[11px] font-medium uppercase tracking-[0.24em] text-[#0A0A0A] transition-all duration-300 hover:border-[#0A0A0A] hover:bg-[#0A0A0A] hover:text-[#FBFBF9]"
                >
                  <span>AUDIT ARCHIVE</span>
                  <span className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5">
                    ↗
                  </span>
                </Link>
              </div>
            </div>

            <div className="relative flex min-h-[380px] flex-col items-center justify-center bg-[#FFFFFF] p-8 sm:p-12 lg:col-span-5">
              <span className="pointer-events-none absolute left-4 top-4 font-mono text-[9px] tracking-[0.28em] text-[#737373]">
                FIG. 01 — APERTURE
              </span>
              <span className="pointer-events-none absolute right-4 top-4 font-mono text-[9px] tracking-[0.28em] text-[#737373] tabular-nums">
                V:{gaze.velocity.toFixed(1)}
              </span>

              <div className="hero-stagger">
                <OculusMark onTelemetry={setGaze} scrollRatio={scrollRatio} />
              </div>

              <div className="pointer-events-none absolute inset-x-0 bottom-0 grid grid-cols-3 border-t border-[#0A0A0A]/12 font-mono text-[9px] uppercase tracking-[0.22em] text-[#525252] tabular-nums">
                <div className="border-r border-[#0A0A0A]/12 px-4 py-2.5">
                  <span className="block text-[#8A8A8A]">EULER.X</span>
                  <span className="text-[#0A0A0A]">{formatSign(gaze.yaw)}°</span>
                </div>
                <div className="border-r border-[#0A0A0A]/12 px-4 py-2.5">
                  <span className="block text-[#8A8A8A]">EULER.Y</span>
                  <span className="text-[#0A0A0A]">{formatSign(gaze.pitch)}°</span>
                </div>
                <div className="px-4 py-2.5">
                  <span className="block text-[#8A8A8A]">STATE</span>
                  <span className="text-[#0A0A0A]">
                    {activeSessionsCount > 0 ? "TRACKING" : "NOMINAL"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section aria-label="Signal Telemetry Matrix" className="border-b border-[#0A0A0A]/12">
          <div className="arch-reveal flex flex-wrap items-end justify-between gap-4 border-b border-[#0A0A0A]/12 px-6 py-5 sm:px-8">
            <div className="flex items-center gap-4">
              <span className="font-mono text-[10px] tracking-[0.3em] text-[#737373]">
                SEC. 01
              </span>
              <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[#0A0A0A]">
                SIGNAL ARCHITECTURE MATRIX
              </h2>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-[0.26em] text-[#525252]">
              <CipherReadout text="FOUR-STAGE MULTIMODAL PIPELINE" />
            </span>
          </div>

          <div className="grid grid-cols-1 divide-y divide-[#0A0A0A]/12 sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4">
            {metrics.map((m, idx) => {
              const hovered = hoveredMetric === idx;
              return (
                <div
                  key={m.index}
                  onMouseEnter={() => setHoveredMetric(idx)}
                  onMouseLeave={() => setHoveredMetric(null)}
                  className="arch-reveal group relative flex flex-col justify-between overflow-hidden border-[#0A0A0A]/12 p-6 transition-colors duration-500 sm:border-b sm:odd:border-r lg:border-b-0 lg:border-r lg:last:border-r-0 lg:p-8"
                >
                  <div
                    className={`pointer-events-none absolute inset-0 origin-bottom bg-[#0A0A0A] transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] ${
                      hovered ? "scale-y-100" : "scale-y-0"
                    }`}
                  />

                  <div
                    className={`relative z-10 flex items-center justify-between font-mono text-[10px] tracking-[0.28em] transition-colors duration-300 ${
                      hovered ? "text-[#A3A3A3]" : "text-[#737373]"
                    }`}
                  >
                    <span>
                      {m.index} // {m.label}
                    </span>
                    <span
                      className={`border px-2 py-0.5 text-[9px] ${
                        hovered
                          ? "border-[#FBFBF9]/30 text-[#FBFBF9]"
                          : "border-[#0A0A0A]/20 text-[#0A0A0A]"
                      }`}
                    >
                      {m.tag}
                    </span>
                  </div>

                  <div
                    className={`relative z-10 my-8 transition-colors duration-300 ${
                      hovered ? "text-[#FBFBF9]" : "text-[#0A0A0A]"
                    }`}
                  >
                    <HarmonicWaveform freq={m.freq} active={hovered} />
                  </div>

                  <div className="relative z-10 space-y-2">
                    <div
                      className={`flex flex-wrap items-baseline gap-2 transition-colors duration-300 ${
                        hovered ? "text-[#FBFBF9]" : "text-[#0A0A0A]"
                      }`}
                    >
                      <span className="font-display text-2xl font-semibold tracking-[-0.03em] tabular-nums">
                        {m.value}
                      </span>
                      <span
                        className={`font-mono text-[11px] ${
                          hovered ? "text-[#A3A3A3]" : "text-[#525252]"
                        }`}
                      >
                        {m.unit}
                      </span>
                    </div>
                    <p
                      className={`font-mono text-[10px] tracking-[0.16em] transition-colors duration-300 ${
                        hovered ? "text-[#D4D4D4]" : "text-[#666666]"
                      }`}
                    >
                      {m.spec}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section aria-label="Ingestion Protocols" className="border-b border-[#0A0A0A]/12">
          <div className="arch-reveal flex flex-wrap items-end justify-between gap-4 border-b border-[#0A0A0A]/12 px-6 py-5 sm:px-8">
            <div className="flex items-center gap-4">
              <span className="font-mono text-[10px] tracking-[0.3em] text-[#737373]">
                SEC. 02
              </span>
              <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[#0A0A0A]">
                PROCTORING INGESTION PROTOCOLS
              </h2>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-[0.26em] text-[#525252]">
              ZERO-COST SANDBOX TIERS ENABLED
            </span>
          </div>

          <div className="grid grid-cols-1 divide-y divide-[#0A0A0A]/12 md:grid-cols-3 md:divide-x md:divide-y-0">
            {modes.map((mode) => (
              <Link
                key={mode.num}
                href={mode.href}
                className="arch-reveal group relative flex flex-col justify-between bg-[#FBFBF9] p-8 transition-colors duration-500 hover:bg-[#0A0A0A] hover:text-[#FBFBF9] lg:p-10"
              >
                <div className="space-y-8">
                  <div className="flex items-center justify-between font-mono text-[10px] tracking-[0.26em] text-[#737373] transition-colors group-hover:text-[#A3A3A3]">
                    <span>MODE {mode.num}</span>
                    <span className="border border-current px-2 py-0.5 text-[9px]">
                      {mode.tier}
                    </span>
                  </div>

                  <div className="py-2 text-[#0A0A0A] transition-colors duration-500 group-hover:text-[#FBFBF9]">
                    <IngestionSchematic variant={mode.variant} />
                  </div>

                  <div className="space-y-3">
                    <span className="block font-mono text-[9px] tracking-[0.28em] text-[#737373] group-hover:text-[#A3A3A3]">
                      {mode.code}
                    </span>
                    <h3 className="font-display text-2xl font-semibold tracking-[-0.03em]">
                      {mode.title}
                    </h3>
                    <p className="text-[13.5px] leading-[1.68] text-[#525252] transition-colors group-hover:text-[#D4D4D4]">
                      {mode.body}
                    </p>
                  </div>
                </div>

                <div className="mt-10 flex items-center justify-between border-t border-[#0A0A0A]/12 pt-5 font-mono text-[10px] tracking-[0.26em] transition-colors group-hover:border-[#FBFBF9]/20">
                  <span>{mode.action}</span>
                  <span className="inline-flex h-7 w-7 items-center justify-center border border-current transition-transform duration-300 group-hover:translate-x-1">
                    →
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </section>

        <section aria-label="Interview Sessions Ledger">
          <div className="arch-reveal flex flex-col justify-between gap-5 border-b border-[#0A0A0A]/12 px-6 py-6 sm:px-8 lg:flex-row lg:items-center">
            <div className="space-y-1">
              <div className="flex items-center gap-4">
                <span className="font-mono text-[10px] tracking-[0.3em] text-[#737373]">
                  SEC. 03
                </span>
                <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[#0A0A0A]">
                  INTERVIEW SESSIONS LEDGER
                </h2>
              </div>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#737373]">
                LIVE STREAM STATUS · CALIBRATED RISK INDICES · AUDIT DISPATCH
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="SEARCH CANDIDATE / ROLE"
                  className="w-56 border border-[#0A0A0A]/20 bg-[#FFFFFF] px-3.5 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[#0A0A0A] placeholder:text-[#8A8A8A] focus:border-[#0A0A0A] focus:outline-none"
                />
              </div>

              <div className="inline-flex border border-[#0A0A0A]/20 bg-[#FFFFFF]">
                {(["ALL", "LIVE", "COMPLETED"] as FilterTab[]).map((tab) => {
                  const active = activeFilter === tab;
                  return (
                    <button
                      key={tab}
                      type="button"
                      onClick={() => setActiveFilter(tab)}
                      className={`px-3.5 py-2 font-mono text-[10px] tracking-[0.22em] transition-colors ${
                        active
                          ? "bg-[#0A0A0A] text-[#FBFBF9]"
                          : "text-[#525252] hover:text-[#0A0A0A]"
                      }`}
                    >
                      {tab}
                    </button>
                  );
                })}
              </div>

              <button
                type="button"
                onClick={() => loadData(true)}
                className="border border-[#0A0A0A]/20 bg-[#FFFFFF] px-3.5 py-2 font-mono text-[10px] tracking-[0.22em] text-[#0A0A0A] transition-colors hover:border-[#0A0A0A] hover:bg-[#0A0A0A] hover:text-[#FBFBF9]"
              >
                {refreshing ? "SYNCING..." : "SYNC"}
              </button>
            </div>
          </div>

          <div className="draw-rule h-px w-full bg-[#0A0A0A]/12" />

          <div className="bg-[#FFFFFF]">
            {loading ? (
              <div className="divide-y divide-[#0A0A0A]/10">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="grid grid-cols-12 items-center gap-4 px-6 py-6 sm:px-8"
                  >
                    <div className="col-span-4 space-y-2">
                      <div className="h-3.5 w-40 animate-pulse bg-[#0A0A0A]/10" />
                      <div className="h-2.5 w-24 animate-pulse bg-[#0A0A0A]/5" />
                    </div>
                    <div className="col-span-3 h-3 w-28 animate-pulse bg-[#0A0A0A]/10" />
                    <div className="col-span-3 h-3 w-24 animate-pulse bg-[#0A0A0A]/10" />
                    <div className="col-span-2 ml-auto h-7 w-20 animate-pulse bg-[#0A0A0A]/10" />
                  </div>
                ))}
              </div>
            ) : sessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center px-6 py-24 text-center">
                <svg viewBox="0 0 120 60" className="mb-6 h-12 w-24 opacity-30">
                  <path
                    d="M 10 22 C 42 -2, 78 -2, 110 24 L 92 34 C 72 52, 48 52, 26 32"
                    fill="none"
                    stroke="#0A0A0A"
                    strokeWidth="6"
                  />
                  <path
                    d="M 110 38 C 78 62, 42 62, 10 36 L 28 26 C 48 8, 72 8, 94 28"
                    fill="none"
                    stroke="#0A0A0A"
                    strokeWidth="6"
                  />
                  <circle cx="60" cy="30" r="6" fill="#0A0A0A" />
                </svg>
                <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-[#0A0A0A]">
                  LEDGER EMPTY // NO ACTIVE SESSIONS
                </p>
                <p className="mt-2 max-w-md text-[13px] text-[#666666]">
                  Initialize your first sentinel stream via local recording,
                  WebRTC browser capture, or live headless bot dispatch.
                </p>
                <Link
                  href="/sessions/new"
                  className="mt-7 border border-[#0A0A0A] bg-[#0A0A0A] px-6 py-3 font-mono text-[10px] tracking-[0.26em] text-[#FBFBF9] transition-colors hover:bg-[#FBFBF9] hover:text-[#0A0A0A]"
                >
                  + INITIALIZE SESSION
                </Link>
              </div>
            ) : filteredSessions.length === 0 ? (
              <div className="px-6 py-20 text-center font-mono text-[11px] uppercase tracking-[0.26em] text-[#737373]">
                NO RECORDS MATCH CURRENT FILTER PREDICATE
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-[#0A0A0A]/12 bg-[#FBFBF9] font-mono text-[9.5px] uppercase tracking-[0.26em] text-[#737373]">
                      <th className="px-6 py-4 sm:px-8">CANDIDATE // ROLE</th>
                      <th className="px-6 py-4">INGESTION</th>
                      <th className="px-6 py-4">STATUS</th>
                      <th className="px-6 py-4">CALIBRATED RISK INDEX</th>
                      <th className="px-6 py-4 text-right sm:px-8">DISPATCH</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#0A0A0A]/10">
                    {filteredSessions.map((s) => {
                      const isCompleted = s.status === "COMPLETED";
                      const isLive = isLiveStatus(s.status);
                      const riskBand = s.risk_band || "LOW";
                      const score =
                        s.overall_risk_score !== null &&
                        s.overall_risk_score !== undefined
                          ? Math.round(s.overall_risk_score)
                          : null;
                      const isElevated = riskBand === "ELEVATED";

                      return (
                        <tr
                          key={s.id}
                          className="group transition-colors duration-200 hover:bg-[#0A0A0A] hover:text-[#FBFBF9]"
                        >
                          <td className="px-6 py-5 sm:px-8">
                            <div className="font-display text-[15px] font-semibold tracking-[-0.015em]">
                              {s.candidate_name}
                            </div>
                            <div className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-[#737373] group-hover:text-[#A3A3A3]">
                              {s.job_title}
                            </div>
                          </td>

                          <td className="px-6 py-5 font-mono text-[10px] tracking-[0.18em] text-[#525252] group-hover:text-[#D4D4D4]">
                            {formatMode(s.ingestion_mode)}
                          </td>

                          <td className="px-6 py-5">
                            <span className="inline-flex items-center gap-2 border border-current/20 px-2.5 py-1 font-mono text-[9.5px] uppercase tracking-[0.2em]">
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
                                        isElevated
                                          ? "text-[#E12D1E]"
                                          : "text-current"
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
                                        isElevated
                                          ? "bg-[#E12D1E]"
                                          : "bg-current"
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
                                className="inline-flex items-center gap-2 border border-[#0A0A0A]/25 px-3.5 py-2 font-mono text-[9.5px] uppercase tracking-[0.22em] transition-colors group-hover:border-[#FBFBF9]/40 group-hover:text-[#FBFBF9] hover:!bg-[#FBFBF9] hover:!text-[#0A0A0A]"
                              >
                                <span>MONITOR</span>
                                <span>→</span>
                              </Link>

                              {isCompleted && (
                                <Link
                                  href={`/reports/${s.id}`}
                                  className="inline-flex items-center border border-[#0A0A0A] bg-[#0A0A0A] px-3.5 py-2 font-mono text-[9.5px] uppercase tracking-[0.22em] text-[#FBFBF9] transition-colors group-hover:border-[#FBFBF9] group-hover:bg-[#FBFBF9] group-hover:text-[#0A0A0A]"
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
          </div>

          <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-[#0A0A0A]/12 bg-[#FBFBF9] px-6 py-4 font-mono text-[10px] uppercase tracking-[0.24em] text-[#737373] sm:px-8">
            <span className="tabular-nums">
              RECORDS: {String(filteredSessions.length).padStart(2, "0")} /{" "}
              {String(sessions.length).padStart(2, "0")} · POLL 5000MS
            </span>
            <Link
              href="/reports"
              className="text-[#0A0A0A] transition-opacity hover:opacity-60"
            >
              COMPLETE AUDIT LEDGER ↗
            </Link>
          </footer>
        </section>
      </div>
    </div>
  );
}
