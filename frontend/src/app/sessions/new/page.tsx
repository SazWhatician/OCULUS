"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { createSession, uploadDemoFile, WS_BASE_URL } from "@/lib/api";

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
    setCaptureStatus("Initializing session...");

    try {
      // 1. Create session record
      const session = await createSession({
        candidate_name: candidateName,
        interviewer_name: interviewerName,
        job_title: jobTitle,
        job_description: jobDescription,
        ingestion_mode: "browser_webrtc",
      });

      setCaptureStatus("Requesting browser tab with audio...");

      // 2. Request screen/tab capture with tab audio
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "browser" },
        audio: true,
      });

      setCaptureStatus("Connecting to OCULUS Ingestion WebSocket...");
      const wsUrl = `${WS_BASE_URL}/ws/ingest/${session.session_id}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setCaptureStatus("WebRTC tab streaming live! Redirecting to monitor...");

        // Setup MediaRecorder for 5-second audio chunks
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length > 0) {
          const audioStream = new MediaStream(audioTracks);
          const recorder = new MediaRecorder(audioStream, { mimeType: "audio/webm;codecs=opus" });
          recorder.ondataavailable = async (e) => {
            if (e.data && e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
              const buffer = await e.data.arrayBuffer();
              ws.send(buffer);
            }
          };
          recorder.start(5000);
        }

        // Setup 1 FPS video frame sampler via hidden canvas
        const video = document.createElement("video");
        video.srcObject = stream;
        video.muted = true;
        video.play();

        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        const ctx = canvas.getContext("2d");

        const frameInterval = setInterval(() => {
          if (ctx && video.readyState >= 2 && ws.readyState === WebSocket.OPEN) {
            ctx.drawImage(video, 0, 0, 640, 360);
            const jpegDataUrl = canvas.toDataURL("image/jpeg", 0.75);
            ws.send(JSON.stringify({ type: "video_frame", data_url: jpegDataUrl }));
          }
        }, 1000);

        stream.getVideoTracks()[0].onended = () => {
          clearInterval(frameInterval);
          ws.close();
        };

        // Redirect recruiter to live monitor
        setTimeout(() => {
          router.push(`/sessions/${session.session_id}`);
        }, 1500);
      };

      ws.onerror = (err) => {
        console.error("WebRTC Ingestion WS error:", err);
        router.push(`/sessions/${session.session_id}`);
      };
    } catch (err: any) {
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
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="space-y-2 text-center md:text-left">
        <Link href="/" className="text-xs font-semibold text-[var(--apple-accent)] hover:underline inline-flex items-center gap-1">
          ← Back to Dashboard
        </Link>
        <h1 className="text-3xl font-extrabold text-[var(--apple-text-primary)] tracking-tight">
          Launch New Proctoring Sentinel
        </h1>
        <p className="text-[var(--apple-text-secondary)] text-sm max-w-2xl">
          Choose an ingestion mode below. All modes route through the identical Redis Stream event pipeline with
          hybrid STT, OpenCV vision debouncing, and Q0 baseline calibration.
        </p>
      </div>

      {/* Mode Selector Tabs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-1.5 bg-[var(--apple-bg)] rounded-2xl border border-[var(--apple-border)]">
        <button
          type="button"
          onClick={() => setActiveTab("demo_file")}
          className={`px-4 py-3 rounded-xl text-left transition-all ${
            activeTab === "demo_file"
              ? "bg-[var(--apple-surface)] shadow-md border border-[var(--apple-border)] font-bold"
              : "opacity-75 hover:opacity-100"
          }`}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-lg">📁</span>
            <span className="apple-badge text-[10px] bg-blue-500/10 text-blue-600">Free $0</span>
          </div>
          <div className="text-xs font-extrabold text-[var(--apple-text-primary)]">Demo File Replay</div>
          <div className="text-[11px] text-[var(--apple-text-secondary)]">Upload MP4 / WAV mock interview</div>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("browser_webrtc")}
          className={`px-4 py-3 rounded-xl text-left transition-all ${
            activeTab === "browser_webrtc"
              ? "bg-[var(--apple-surface)] shadow-md border border-[var(--apple-border)] font-bold"
              : "opacity-75 hover:opacity-100"
          }`}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-lg">🌐</span>
            <span className="apple-badge text-[10px] bg-emerald-500/10 text-emerald-600">Zero-Key $0</span>
          </div>
          <div className="text-xs font-extrabold text-[var(--apple-text-primary)]">Browser Tab Capture</div>
          <div className="text-[11px] text-[var(--apple-text-secondary)]">Live WebRTC screen & tab audio</div>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("attendee_bot")}
          className={`px-4 py-3 rounded-xl text-left transition-all ${
            activeTab === "attendee_bot"
              ? "bg-[var(--apple-surface)] shadow-md border border-[var(--apple-border)] font-bold"
              : "opacity-75 hover:opacity-100"
          }`}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-lg">🤖</span>
            <span className="apple-badge text-[10px] bg-purple-500/10 text-purple-600">Cloud Bot</span>
          </div>
          <div className="text-xs font-extrabold text-[var(--apple-text-primary)]">Meeting Bot Dispatch</div>
          <div className="text-[11px] text-[var(--apple-text-secondary)]">Zoom / Google Meet URL</div>
        </button>
      </div>

      {/* Shared Candidate Profile Information */}
      <div className="apple-card p-6 space-y-4">
        <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
          1. Candidate & Target Role Setup
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-bold text-[var(--apple-text-primary)]">
              Candidate Full Name <span className="text-rose-500">*</span>
            </label>
            <input
              type="text"
              required
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              className="w-full px-3.5 py-2.5 apple-input text-xs"
              placeholder="e.g. Alex Johnson"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-[var(--apple-text-primary)]">
              Target Job Title <span className="text-rose-500">*</span>
            </label>
            <input
              type="text"
              required
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              className="w-full px-3.5 py-2.5 apple-input text-xs"
              placeholder="e.g. Senior Backend Engineer"
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-bold text-[var(--apple-text-primary)]">
            Job Description & Technical Keywords (Seeds Speechmatics Dictionary)
          </label>
          <textarea
            rows={2}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            className="w-full px-3.5 py-2.5 apple-input text-xs"
            placeholder="Key technical terms (Kafka, Kubernetes, PyTorch) extracted automatically..."
          />
        </div>
      </div>

      {/* Tab Specific Content */}
      {activeTab === "demo_file" && (
        <form onSubmit={handleDemoSubmit} className="space-y-6">
          <div className="apple-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
                2. Upload Interview Recording (.mp4, .webm, .wav)
              </h2>
              <span className="apple-badge text-[11px] bg-blue-500/10 text-blue-600 font-mono">
                100% Free Demo Mode
              </span>
            </div>

            <div
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer ${
                selectedFile
                  ? "border-[var(--apple-accent)] bg-[var(--apple-accent-light)]"
                  : "border-[var(--apple-border)] hover:border-[var(--apple-accent)] bg-[var(--apple-bg)]"
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

              <div className="space-y-2">
                <span className="text-3xl block">📹</span>
                <p className="text-sm font-bold text-[var(--apple-text-primary)]">
                  {selectedFile ? selectedFile.name : "Click to select or drag and drop recording here"}
                </p>
                <p className="text-xs text-[var(--apple-text-secondary)]">
                  Supports MP4, WebM, and WAV up to 150 MB
                </p>
              </div>
            </div>

            {uploading && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-[var(--apple-text-secondary)] font-semibold">
                  <span>Uploading & extracting audio/video tracks via FFmpeg...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full bg-[var(--apple-bg)] rounded-full h-2 overflow-hidden border border-[var(--apple-border)]">
                  <div
                    className="bg-[var(--apple-accent)] h-2 transition-all duration-300 rounded-full"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-end gap-3">
            <Link href="/" className="apple-button-secondary px-6 py-2.5 text-xs font-semibold">
              Cancel
            </Link>
            <button
              type="submit"
              disabled={uploading}
              className="apple-button-primary px-8 py-2.5 text-xs font-bold inline-flex items-center gap-2"
            >
              {uploading ? "Processing Replay..." : "Start Demo Replay →"}
            </button>
          </div>
        </form>
      )}

      {activeTab === "browser_webrtc" && (
        <div className="space-y-6">
          <div className="apple-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
                2. Live Browser Tab & Screen Capture
              </h2>
              <span className="apple-badge text-[11px] bg-emerald-500/10 text-emerald-600 font-mono">
                No Bot or ngrok Needed
              </span>
            </div>

            <p className="text-xs text-[var(--apple-text-secondary)] leading-relaxed">
              Open your Zoom, Google Meet, or Microsoft Teams meeting in a Chrome/Edge browser tab or window.
              When you click <strong>Capture Tab Audio & Video</strong>, choose the tab and check{" "}
              <strong>&quot;Also share tab audio&quot;</strong>.
            </p>

            <div className="p-4 bg-[var(--apple-bg)] rounded-xl border border-[var(--apple-border)] text-xs space-y-2">
              <div className="font-bold text-[var(--apple-text-primary)] flex items-center gap-2">
                <span>💡</span> How it works:
              </div>
              <ul className="list-disc list-inside space-y-1 text-[var(--apple-text-secondary)]">
                <li>Captures the candidate&apos;s speech and webcam video stream in real-time.</li>
                <li>Transmits 5s audio chunks + 1 FPS video frames directly over WebSocket to the Sentinel workers.</li>
                <li>Zero external API keys or bot subscriptions required for local testing.</li>
              </ul>
            </div>

            {captureStatus && (
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 rounded-xl text-xs font-bold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 live-pulse"></span>
                {captureStatus}
              </div>
            )}
          </div>

          <div className="flex justify-end gap-3">
            <Link href="/" className="apple-button-secondary px-6 py-2.5 text-xs font-semibold">
              Cancel
            </Link>
            <button
              type="button"
              onClick={handleWebRTCStart}
              disabled={isCapturing}
              className="apple-button-primary px-8 py-2.5 text-xs font-bold inline-flex items-center gap-2"
            >
              {isCapturing ? "Capturing..." : "Select Tab & Start Sentinel →"}
            </button>
          </div>
        </div>
      )}

      {activeTab === "attendee_bot" && (
        <form onSubmit={handleBotSubmit} className="space-y-6">
          <div className="apple-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
                2. Meeting Target URL & Bot Config
              </h2>
              <span className="apple-badge text-[11px] bg-purple-500/10 text-purple-600 font-mono">
                Attendee.dev Cloud
              </span>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">
                Meeting URL (Zoom / Google Meet / Teams) <span className="text-rose-500">*</span>
              </label>
              <input
                type="url"
                required
                value={meetingUrl}
                onChange={(e) => setMeetingUrl(e.target.value)}
                className="w-full px-3.5 py-2.5 apple-input text-xs"
                placeholder="https://zoom.us/j/9876543210?pwd=... or https://meet.google.com/abc-defg-hij"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">
                Bot Display Name in Meeting
              </label>
              <input
                type="text"
                value={botDisplayName}
                onChange={(e) => setBotDisplayName(e.target.value)}
                className="w-full px-3.5 py-2.5 apple-input text-xs"
                placeholder="OCULUS Sentinel"
              />
            </div>
          </div>

          <div className="flex justify-end gap-3">
            <Link href="/" className="apple-button-secondary px-6 py-2.5 text-xs font-semibold">
              Cancel
            </Link>
            <button
              type="submit"
              disabled={isSubmitting}
              className="apple-button-primary px-8 py-2.5 text-xs font-bold inline-flex items-center gap-2"
            >
              {isSubmitting ? "Dispatching Bot..." : "Dispatch Bot to Meeting →"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

export default function NewSessionPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-xs text-[var(--apple-text-secondary)]">Loading launcher...</div>}>
      <NewSessionContent />
    </Suspense>
  );
}
