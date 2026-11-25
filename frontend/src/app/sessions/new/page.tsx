"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { createSession, uploadDemoFile, WS_BASE_URL } from "@/lib/api";
import { webrtcCapture } from "@/lib/webrtcCapture";

function NewSessionContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [activeTab, setActiveTab] = useState<"demo_file" | "browser_webrtc" | "attendee_bot">("demo_file");

  // Common Candidate Fields
  const [candidateName, setCandidateName] = useState("Alex Johnson");
  const [interviewerName, setInterviewerName] = useState("Sarah Lead");
  const [jobTitle, setJobTitle] = useState("Senior Distributed Systems Engineer");
  const [jobDescription, setJobDescription] = useState(
    "Experience with Kubernetes, Kafka, Redis, Golang, and high-throughput microservices."
  );

  // Tab 1: Demo File Replay State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Tab 2: Browser WebRTC Tab Capture State
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureStatus, setCaptureStatus] = useState<string | null>(null);

  // Tab 3: Attendee Bot Dispatch State
  const [meetingUrl, setMeetingUrl] = useState("");
  const [botDisplayName, setBotDisplayName] = useState("OCULUS Sentinel");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const mode = searchParams.get("mode");
    if (mode === "browser_webrtc" || mode === "attendee_bot" || mode === "demo_file") {
      setActiveTab(mode);
    }
  }, [searchParams]);

  // Handle Tab 1 Demo File Submit
  const handleDemoSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      alert("Please select or drop an interview recording (.mp4, .webm, or .wav) to continue.");
      return;
    }

    setUploading(true);
    setUploadProgress(20);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("candidate_name", candidateName);
      formData.append("job_title", jobTitle);
      formData.append("job_description", jobDescription);

      setUploadProgress(60);
      const res = await uploadDemoFile(formData);
      setUploadProgress(100);

      router.push(`/sessions/${res.session_id}`);
    } catch (err: any) {
      alert(`Demo upload error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  // Handle Tab 2 WebRTC Tab Capture Submit
  const handleWebRTCStart = async () => {
    setIsCapturing(true);
    setCaptureStatus("INITIALIZING PIPELINE BUFFER...");

    try {
      const session = await createSession({
        candidate_name: candidateName,
        interviewer_name: interviewerName,
        job_title: jobTitle,
        job_description: jobDescription,
        ingestion_mode: "browser_webrtc",
      });

      setCaptureStatus("REQUESTING BROWSER SCREEN/TAB STREAM & LINKING SENTINEL...");
      await webrtcCapture.start(session.session_id, WS_BASE_URL);
      setCaptureStatus("SENTINEL LINK ESTABLISHED · REDIRECTING TO LIVE MONITOR");
      router.push(`/sessions/${session.session_id}`);
    } catch (err: any) {
      await webrtcCapture.stop();
      alert(`Capture failed: ${err.message}`);
      setIsCapturing(false);
      setCaptureStatus(null);
    }
  };

  // Handle Tab 3 Attendee Bot Dispatch Submit
  const handleBotSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!meetingUrl) {
      alert("Please enter a valid Zoom, Google Meet, or Teams URL.");
      return;
    }

    setIsSubmitting(true);
    try {
      const res = await createSession({
        candidate_name: candidateName,
        interviewer_name: interviewerName,
        job_title: jobTitle,
        job_description: jobDescription,
        ingestion_mode: "attendee_bot",
        meeting_url: meetingUrl,
        bot_display_name: botDisplayName,
      });

      router.push(`/sessions/${res.session_id}`);
    } catch (err: any) {
      alert(`Bot dispatch failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
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
            DISPATCH PROTOCOL // Q0_INIT
          </span>
        </div>

        <div className="space-y-1">
          <h1 className="font-display text-3xl font-semibold tracking-[-0.03em] text-[var(--oculus-ink)] sm:text-4xl">
            INITIALIZE SENTINEL
          </h1>
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--oculus-muted)] max-w-2xl">
            Deploy real-time multimodal stream analysis across candidate audio cadence, gaze deflection, and LLM telemetry.
          </p>
        </div>
      </div>

      {/* Mode Selector Tabs */}
      <div className="grid grid-cols-1 md:grid-cols-3 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] divide-y md:divide-y-0 md:divide-x divide-[var(--oculus-hairline)]">
        <button
          type="button"
          onClick={() => setActiveTab("demo_file")}
          className={`p-6 text-left transition-all relative ${
            activeTab === "demo_file"
              ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)]"
              : "text-[var(--oculus-ink)] hover:bg-[var(--oculus-ink)]/5"
          }`}
        >
          <div className="flex items-center justify-between mb-3 font-mono text-[9px] tracking-[0.26em] uppercase">
            <span>MODE // 01</span>
            <span className={`px-2 py-0.5 border ${activeTab === "demo_file" ? "border-[var(--oculus-bg)]/40" : "border-[var(--oculus-hairline)]"}`}>
              $0 FREE LOCAL
            </span>
          </div>
          <div className="font-display text-base font-semibold tracking-[-0.02em] mb-1">
            DEMO FILE REPLAY
          </div>
          <p className={`font-mono text-[10px] uppercase tracking-[0.16em] ${activeTab === "demo_file" ? "text-[var(--oculus-bg)]/75" : "text-[var(--oculus-muted)]"}`}>
            Upload interview recording (.mp4 / .webm / .wav) with simulated timeline.
          </p>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("browser_webrtc")}
          className={`p-6 text-left transition-all relative ${
            activeTab === "browser_webrtc"
              ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)]"
              : "text-[var(--oculus-ink)] hover:bg-[var(--oculus-ink)]/5"
          }`}
        >
          <div className="flex items-center justify-between mb-3 font-mono text-[9px] tracking-[0.26em] uppercase">
            <span>MODE // 02</span>
            <span className={`px-2 py-0.5 border ${activeTab === "browser_webrtc" ? "border-[var(--oculus-bg)]/40" : "border-[var(--oculus-hairline)]"}`}>
              REAL-TIME WEBRTC
            </span>
          </div>
          <div className="font-display text-base font-semibold tracking-[-0.02em] mb-1">
            BROWSER TAB CAPTURE
          </div>
          <p className={`font-mono text-[10px] uppercase tracking-[0.16em] ${activeTab === "browser_webrtc" ? "text-[var(--oculus-bg)]/75" : "text-[var(--oculus-muted)]"}`}>
            Live client-side capture of Meet, Zoom, or Teams tab audio & video.
          </p>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("attendee_bot")}
          className={`p-6 text-left transition-all relative ${
            activeTab === "attendee_bot"
              ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)]"
              : "text-[var(--oculus-ink)] hover:bg-[var(--oculus-ink)]/5"
          }`}
        >
          <div className="flex items-center justify-between mb-3 font-mono text-[9px] tracking-[0.26em] uppercase">
            <span>MODE // 03</span>
            <span className={`px-2 py-0.5 border ${activeTab === "attendee_bot" ? "border-[var(--oculus-bg)]/40" : "border-[var(--oculus-hairline)]"}`}>
              CLOUD BOT
            </span>
          </div>
          <div className="font-display text-base font-semibold tracking-[-0.02em] mb-1">
            HEADLESS BOT DISPATCH
          </div>
          <p className={`font-mono text-[10px] uppercase tracking-[0.16em] ${activeTab === "attendee_bot" ? "text-[var(--oculus-bg)]/75" : "text-[var(--oculus-muted)]"}`}>
            Dispatch Attendee.dev silent recorder via meeting URL join credentials.
          </p>
        </button>
      </div>

      {/* Shared Candidate Profile Configuration */}
      <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 sm:p-8 space-y-6">
        <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
              SEC. 01
            </span>
            <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
              CANDIDATE & ROLE TELEMETRY
            </h2>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
            REQUIRED SPECIFICATION
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
              Candidate Legal Name <span className="text-[#E12D1E]">*</span>
            </label>
            <input
              type="text"
              required
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-3 font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
              placeholder="E.G. ALEX JOHNSON"
            />
          </div>

          <div className="space-y-2">
            <label className="block font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
              Target Job Classification <span className="text-[#E12D1E]">*</span>
            </label>
            <input
              type="text"
              required
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-3 font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
              placeholder="E.G. PRINCIPAL SYSTEMS ARCHITECT"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="block font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
            Technical Domain Lexicon (Seeds Hybrid STT Vocabulary & Heuristics)
          </label>
          <textarea
            rows={2}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-3 font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
            placeholder="KAFKA, KUBERNETES, DISTRIBUTED TRANSACTIONS, RAFT CONSENSUS..."
          />
        </div>
      </div>

      {/* Tab 1 Content: Demo File Upload */}
      {activeTab === "demo_file" && (
        <form onSubmit={handleDemoSubmit} className="space-y-6">
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                  SEC. 02
                </span>
                <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                  INGEST AUDIO/VIDEO RECORDING
                </h2>
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.24em] text-[var(--oculus-muted)]">
                FORMATS: .MP4, .WEBM, .WAV (MAX 150MB)
              </span>
            </div>

            <div
              className={`border border-dashed p-10 text-center transition-all cursor-pointer ${
                selectedFile
                  ? "border-[var(--oculus-ink)] bg-[var(--oculus-ink)]/5"
                  : "border-[var(--oculus-hairline)] hover:border-[var(--oculus-ink)] bg-[var(--oculus-bg)]"
              }`}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <input
                id="file-input"
                type="file"
                accept=".mp4,.webm,.wav,.mp3,.mov"
                className="hidden"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />

              <div className="space-y-3">
                <div className="font-mono text-2xl text-[var(--oculus-ink)] opacity-60">◫</div>
                <p className="font-mono text-[12px] font-semibold uppercase tracking-[0.2em] text-[var(--oculus-ink)]">
                  {selectedFile ? selectedFile.name : "CLICK TO BROWSE OR DRAG RECORDING STREAM HERE"}
                </p>
                <p className="font-mono text-[9.5px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
                  Automatic demuxing into 16kHz mono audio & 1 FPS optical frames
                </p>
              </div>
            </div>

            {uploading && (
              <div className="space-y-2 border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-4">
                <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-muted)]">
                  <span>EXTRACTION & EVENT INGESTION PROGRESS</span>
                  <span className="text-[var(--oculus-ink)] tabular-nums">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-[var(--oculus-hairline)] h-1.5 overflow-hidden">
                  <div
                    className="bg-[var(--oculus-ink)] h-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-end gap-4">
            <Link
              href="/"
              className="border border-[var(--oculus-hairline)] bg-transparent px-6 py-3.5 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-ink)] hover:border-[var(--oculus-ink)] transition-colors"
            >
              ABORT
            </Link>
            <button
              type="submit"
              disabled={uploading}
              className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-8 py-3.5 font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-bg)] transition-colors hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] disabled:opacity-50"
            >
              {uploading ? "INGESTING REPLAY STREAM..." : "DISPATCH REPLAY SENTINEL →"}
            </button>
          </div>
        </form>
      )}

      {/* Tab 2 Content: WebRTC Tab Capture */}
      {activeTab === "browser_webrtc" && (
        <div className="space-y-6">
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                  SEC. 02
                </span>
                <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                  REAL-TIME BROWSER MEDIA CAPTURE
                </h2>
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.24em] text-[var(--oculus-muted)]">
                CLIENT-SIDE ZERO SUBSCRIPTION
              </span>
            </div>

            <p className="font-mono text-[11px] leading-relaxed uppercase tracking-[0.16em] text-[var(--oculus-muted)]">
              Launch your Google Meet, Zoom Web, or Microsoft Teams meeting in an adjacent browser tab.
              When prompted by the browser dialog, select the meeting tab and ensure <strong className="text-[var(--oculus-ink)]">&quot;SHARE TAB AUDIO&quot;</strong> is toggled.
            </p>

            <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] p-5 space-y-3 font-mono text-[10px] uppercase tracking-[0.2em]">
              <div className="text-[var(--oculus-ink)] font-semibold flex items-center gap-2">
                <span>//</span> PROTOCOL EXECUTION PHASES:
              </div>
              <ul className="space-y-1.5 text-[var(--oculus-muted)] pl-4">
                <li>• Real-time candidate audio streamed in 5000ms chunk intervals.</li>
                <li>• Optical gaze & face frame debouncing transmitted via WebSocket bus.</li>
                <li>• Live median/MAD Q0 baseline established in first interview turn.</li>
              </ul>
            </div>

            {captureStatus && (
              <div className="border border-[#E12D1E] bg-[#E12D1E]/5 p-4 text-[#E12D1E] font-mono text-[10px] uppercase tracking-[0.24em] flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-[#E12D1E] animate-pulse" />
                <span>{captureStatus}</span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-end gap-4">
            <Link
              href="/"
              className="border border-[var(--oculus-hairline)] bg-transparent px-6 py-3.5 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-ink)] hover:border-[var(--oculus-ink)] transition-colors"
            >
              ABORT
            </Link>
            <button
              type="button"
              onClick={handleWebRTCStart}
              disabled={isCapturing}
              className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-8 py-3.5 font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-bg)] transition-colors hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] disabled:opacity-50"
            >
              {isCapturing ? "BUFFERING STREAM..." : "SELECT TAB & START SENTINEL →"}
            </button>
          </div>
        </div>
      )}

      {/* Tab 3 Content: Attendee Bot Dispatch */}
      {activeTab === "attendee_bot" && (
        <form onSubmit={handleBotSubmit} className="space-y-6">
          <div className="border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-[var(--oculus-hairline)] pb-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                  SEC. 02
                </span>
                <h2 className="font-display text-lg font-semibold tracking-[-0.02em] text-[var(--oculus-ink)]">
                  ATTENDEE.DEV MEETING BOT DISPATCH
                </h2>
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.24em] text-[var(--oculus-muted)]">
                HEADLESS CLOUD JOIN
              </span>
            </div>

            <div className="space-y-2">
              <label className="block font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
                Target Meeting Link (Zoom, Meet, or Teams) <span className="text-[#E12D1E]">*</span>
              </label>
              <input
                type="url"
                required
                value={meetingUrl}
                onChange={(e) => setMeetingUrl(e.target.value)}
                className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-3 font-mono text-[11px] tracking-[0.12em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
                placeholder="https://zoom.us/j/9876543210?pwd=... or https://meet.google.com/xyz-abcd-efg"
              />
            </div>

            <div className="space-y-2">
              <label className="block font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--oculus-muted)]">
                Bot Participant Display Identity
              </label>
              <input
                type="text"
                value={botDisplayName}
                onChange={(e) => setBotDisplayName(e.target.value)}
                className="w-full border border-[var(--oculus-hairline)] bg-[var(--oculus-bg)] px-4 py-3 font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--oculus-ink)] placeholder:text-[var(--oculus-muted)] focus:border-[var(--oculus-ink)] focus:outline-none transition-colors"
                placeholder="OCULUS SENTINEL"
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-4">
            <Link
              href="/"
              className="border border-[var(--oculus-hairline)] bg-transparent px-6 py-3.5 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--oculus-ink)] hover:border-[var(--oculus-ink)] transition-colors"
            >
              ABORT
            </Link>
            <button
              type="submit"
              disabled={isSubmitting}
              className="border border-[var(--oculus-ink)] bg-[var(--oculus-ink)] px-8 py-3.5 font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-bg)] transition-colors hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)] disabled:opacity-50"
            >
              {isSubmitting ? "TRANSMITTING DISPATCH..." : "DISPATCH BOT TO MEETING →"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

export default function NewSessionPage() {
  return (
    <Suspense fallback={<div className="p-16 text-center font-mono text-[10px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">LOADING DISPATCH PROTOCOL...</div>}>
      <NewSessionContent />
    </Suspense>
  );
}
