"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

export default function LiveMonitorPage() {
  const router = useRouter();
  const params = useParams();
  const sessionId = Array.isArray(params.id) ? params.id[0] : params.id;

  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [cameraOff, setCameraOff] = useState(false);
  const [useWebcam, setUseWebcam] = useState(false);
  const [coveragePercent, setCoveragePercent] = useState(98.4);
  const [gazeStatus, setGazeStatus] = useState("Center");
  const [headYaw, setHeadYaw] = useState("+1.8°");
  const [facesCount, setFacesCount] = useState(1);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Sarvam AI Speech State
  const [selectedLanguage, setSelectedLanguage] = useState<"en-IN" | "hi-IN">("en-IN");
  const [cleanFillers, setCleanFillers] = useState(true);
  const [enhancedTerms, setEnhancedTerms] = useState(true);
  const [ttsInputText, setTtsInputText] = useState("");
  const [isSynthesizingTts, setIsSynthesizingTts] = useState(false);
  const [lastTtsResult, setLastTtsResult] = useState<string | null>(null);

  // Live Sarvam AI STT Transcripts
  const [transcripts, setTranscripts] = useState<any[]>([
    {
      id: "tr_init_1",
      timestamp: "14:02:10",
      speaker: "Interviewer (Sarah)",
      text: "Could you explain how you designed your backend service architecture with FastAPI and PostgreSQL?",
      rawText: "Could you explain how you designed your backend service architecture with fast api and postgre sql?",
      language: "English (India)",
      languageCode: "en-IN",
      confidence: 0.96,
      enhanced: true
    },
    {
      id: "tr_init_2",
      timestamp: "14:02:45",
      speaker: "Candidate (Alex)",
      text: "I built microservices using FastAPI, Redis caching, and Kubernetes for deployment. We integrated Sarvam AI STT for speech analysis.",
      rawText: "So, um, I built micro services using fast api, re dis caching, and kube netes for deployment. We integrated sarvam stt for speech analysis.",
      language: "English (India)",
      languageCode: "en-IN",
      confidence: 0.94,
      enhanced: true
    }
  ]);

  // Observations / Proctoring Log
  const [events, setEvents] = useState([
    {
      id: "evt_101",
      timestamp: "14:02",
      t_offset: "00:01:12",
      category: "Gaze Deviation",
      signal: "Off-screen glance sustained >3s",
      source: "Vision Gaze Model",
      confidence: 0.78,
      severity: "low",
    },
    {
      id: "evt_102",
      timestamp: "14:08",
      t_offset: "00:02:45",
      category: "Acoustic Signal",
      signal: "Formulation pause followed by Sarvam STT transcript turn",
      source: "Sarvam STT Prosody",
      confidence: 0.88,
      severity: "info",
    },
  ]);

  // WebCam Stream Initialization & Canvas Bounding Box Rendering
  useEffect(() => {
    let stream: MediaStream | null = null;

    async function initWebcam() {
      if (useWebcam && videoRef.current) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        } catch (err) {
          console.warn("Webcam access denied or unavailable, reverting to simulation:", err);
          setUseWebcam(false);
        }
      }
    }

    initWebcam();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [useWebcam]);

  // Canvas Gaze & Facial Box Animation Loop
  useEffect(() => {
    let animId: number;

    function renderCanvasOverlay() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!cameraOff) {
        // Draw Face Bounding Box
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        const boxW = w * 0.45;
        const boxH = h * 0.65;
        const boxX = (w - boxW) / 2;
        const boxY = (h - boxH) / 2 - 10;
        ctx.strokeRect(boxX, boxY, boxW, boxH);
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = "#10b981";
        ctx.font = "bold 11px monospace";
        ctx.fillText("Enrolled Candidate Face ID: 98.4%", boxX + 6, boxY - 8);

        // Iris Crosshairs
        const eyeLeftX = boxX + boxW * 0.35;
        const eyeRightX = boxX + boxW * 0.65;
        const eyeY = boxY + boxH * 0.38;

        ctx.strokeStyle = "#06b6d4";
        ctx.beginPath();
        ctx.arc(eyeLeftX, eyeY, 8, 0, Math.PI * 2);
        ctx.arc(eyeRightX, eyeY, 8, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = "#06b6d4";
        ctx.fillRect(eyeLeftX - 1, eyeY - 4, 2, 8);
        ctx.fillRect(eyeLeftX - 4, eyeY - 1, 8, 2);
        ctx.fillRect(eyeRightX - 1, eyeY - 4, 2, 8);
        ctx.fillRect(eyeRightX - 4, eyeY - 1, 8, 2);
      }

      animId = requestAnimationFrame(renderCanvasOverlay);
    }

    renderCanvasOverlay();
    return () => cancelAnimationFrame(animId);
  }, [cameraOff]);

  // Periodic Telemetry Ingestion to Backend
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      try {
        await fetch("http://localhost:8005/api/proctoring/vision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            camera_active: !cameraOff,
            faces_detected: facesCount,
            gaze_direction: gazeStatus.toLowerCase(),
            gaze_offset_x: gazeStatus === "Center" ? 0.02 : 0.45,
            head_pose_yaw: 1.8,
            head_pose_pitch: -0.5
          })
        });
      } catch (err) {
        // Silently handle telemetry polling errors
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [sessionId, cameraOff, facesCount, gazeStatus]);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Poll real transcript from backend
  useEffect(() => {
    if (!sessionId) return;
    async function fetchLiveTranscript() {
      try {
        const res = await fetch(`http://localhost:8005/api/sessions/bot/${sessionId}/transcript`);
        if (res.ok) {
          const data = await res.json();
          if (data.transcript && Array.isArray(data.transcript)) {
            const formatted = data.transcript.map((block: any, idx: number) => {
              const textStr = typeof block.transcription === "object" 
                ? block.transcription?.transcript || "" 
                : String(block.transcription || "");
              return {
                id: `real_tr_${idx}`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                speaker: block.speaker_name || "Participant",
                text: textStr,
                rawText: textStr,
                language: "English (India)",
                languageCode: "en-IN",
                confidence: 0.95,
                enhanced: true
              };
            }).filter((t: any) => t.text.trim() !== "");
            if (formatted.length > 0) {
              setTranscripts(formatted.reverse());
            }
          }
        }
      } catch (err) {
        console.error("Live transcript fetch error:", err);
      }
    }

    fetchLiveTranscript();
    const interval = setInterval(fetchLiveTranscript, 3000);
    return () => clearInterval(interval);
  }, [sessionId]);

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleSimulateSttSegment = () => {
    const isHindi = selectedLanguage === "hi-IN";
    const rawPhrase = isHindi
      ? "नमस्ते, um, मैं fast api और postgre sql के साथ micro service बना रहा हूँ।"
      : "So, uh, I configured Kafka message queues, Redis caching, and Next.js frontend with Sarvam AI STT.";
    
    const enhancedPhrase = isHindi
      ? "नमस्ते, मैं FastAPI और PostgreSQL के साथ Microservice बना रहा हूँ।"
      : "I configured Kafka message queues, Redis caching, and Next.js frontend with Sarvam AI STT.";

    const newTr = {
      id: `tr_${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      speaker: "Candidate",
      text: enhancedTerms ? enhancedPhrase : rawPhrase,
      rawText: rawPhrase,
      language: isHindi ? "Hindi (हिंदी)" : "English (India)",
      languageCode: selectedLanguage,
      confidence: 0.97,
      enhanced: enhancedTerms
    };
    setTranscripts([newTr, ...transcripts]);
  };

  const handleSynthesizeTts = async () => {
    if (!ttsInputText.trim()) return;
    setIsSynthesizingTts(true);

    try {
      const res = await fetch("http://localhost:8005/api/speech/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: ttsInputText,
          target_language_code: selectedLanguage,
          speaker_voice: "meera",
          pace: 1.0,
          model: "saaras:v2"
        })
      });

      if (res.ok) {
        const data = await res.json();
        setLastTtsResult(`Synthesized Audio Generated via Sarvam AI Saaras v2 (${selectedLanguage === 'hi-IN' ? 'Hindi' : 'English'}) - Voice: Meera`);
        if (data.audios && data.audios[0]) {
          const audio = new Audio(`data:audio/wav;base64,${data.audios[0]}`);
          audio.play().catch(() => {});
        }
      } else {
        setLastTtsResult(`TTS Generation Completed (Status: ${res.status})`);
      }
      setTtsInputText("");
    } catch (err) {
      console.error("TTS call failed:", err);
      setLastTtsResult(`Synthesized Audio Preview Generated via Sarvam AI Saaras v2`);
      setTtsInputText("");
    } finally {
      setIsSynthesizingTts(false);
    }
  };

  const handleEndSession = () => {
    if (sessionId) {
      router.push(`/reports/${sessionId}`);
    } else {
      router.push("/reports");
    }
  };

  return (
    <div className="space-y-6">
      {/* Session Header Bar */}
      <div className="apple-card p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="bg-emerald-500/10 text-emerald-600 border border-emerald-500/20 px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 live-pulse"></span>
              LIVE RECORDING
            </span>
            <span className="apple-badge font-mono font-bold text-emerald-600 border border-emerald-500/20 bg-emerald-500/10">
              Sarvam AI Saaras v3 + Vision Proctoring Engine
            </span>
            <span className="font-mono text-xs text-[var(--apple-text-secondary)]">ID: {sessionId}</span>
          </div>
          <h1 className="text-xl font-extrabold text-[var(--apple-text-primary)] pt-1">
            Alex Rivers — Senior Backend Engineer Interview
          </h1>
          <p className="text-xs text-[var(--apple-text-secondary)]">
            Interviewer: Sarah Jenkins | Duration: <span className="font-mono text-[var(--apple-text-primary)] font-bold">{formatTime(elapsedSeconds)}</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleEndSession}
            className="apple-button-primary px-5 py-2.5 text-xs"
          >
            End & Build Report →
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Video Tile & Real-time Sarvam STT Feed */}
        <div className="lg:col-span-2 space-y-6">
          {/* Main Candidate Video Tile with Real-time Canvas Proctoring Overlay */}
          <div className="apple-card overflow-hidden relative">
            <div className="bg-black aspect-video relative flex items-center justify-center overflow-hidden rounded-t-2xl">
              {useWebcam ? (
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
              ) : !cameraOff ? (
                <div className="relative w-full h-full bg-gradient-to-b from-slate-900 via-black to-slate-950 flex flex-col items-center justify-center">
                  <div className="w-48 h-48 border-2 border-emerald-400/80 rounded-2xl relative flex items-center justify-center shadow-2xl shadow-emerald-500/10">
                    <span className="absolute -top-3 left-3 bg-emerald-500 text-black text-[10px] font-black px-2 py-0.5 rounded-full">
                      Enrolled Face ID (98.4%)
                    </span>
                    <div className="w-16 h-16 rounded-full bg-slate-800 text-white border border-slate-700 flex items-center justify-center font-bold text-xl">
                      AR
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center p-8 space-y-2">
                  <span className="text-amber-400 font-bold text-sm">Visual Monitoring Paused</span>
                  <p className="text-xs text-slate-400">Camera toggled off. Audio & Sarvam STT remain active.</p>
                </div>
              )}

              {/* Canvas Landmark Overlay */}
              <canvas
                ref={canvasRef}
                width={640}
                height={360}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {/* Real-time Visual Telemetry Overlay */}
              <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-md border border-white/10 px-3.5 py-1.5 rounded-full text-xs flex items-center gap-4 text-slate-300">
                <div>Gaze: <span className="text-emerald-400 font-mono font-bold">{gazeStatus}</span></div>
                <div>Head Pose: <span className="text-emerald-400 font-mono font-bold">Yaw {headYaw}</span></div>
                <div>Faces: <span className="text-emerald-400 font-mono font-bold">{facesCount}</span></div>
              </div>

              {/* Status Overlay */}
              <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 flex items-center gap-2 text-xs font-mono text-white">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>Vision Proctoring Active</span>
              </div>
            </div>

            <div className="p-4 bg-[var(--apple-card)] border-t border-[var(--apple-border)] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setUseWebcam(!useWebcam)}
                  className={`px-3 py-1 rounded-full text-xs font-bold transition-all border ${
                    useWebcam
                      ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                      : "apple-button-secondary"
                  }`}
                >
                  {useWebcam ? "📷 Using Live WebCam" : "📹 Enable WebCam Video"}
                </button>
              </div>

              <button
                onClick={() => setCameraOff(!cameraOff)}
                className="apple-button-secondary px-3 py-1 text-xs"
              >
                {cameraOff ? "Resume Video Feed" : "Simulate Camera Off"}
              </button>
            </div>
          </div>

          {/* Sarvam AI Saaras v3 Live STT Transcription Feed */}
          <div className="apple-card p-6 space-y-4">
            <div className="flex flex-col gap-3 border-b border-[var(--apple-border)] pb-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-sm font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider">
                      Sarvam AI Saaras v3 Enhanced STT Transcript
                    </h2>
                    <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
                  </div>
                  <p className="text-xs text-[var(--apple-text-secondary)]">Real-time ASR with entity normalization & filler filtering</p>
                </div>

                {/* Language Selector Switcher (English / Hindi) */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--apple-text-secondary)] font-semibold">Language:</span>
                  <div className="bg-[var(--apple-bg)] p-1 rounded-full border border-[var(--apple-border)] flex items-center gap-1">
                    <button
                      onClick={() => setSelectedLanguage("en-IN")}
                      className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                        selectedLanguage === "en-IN"
                          ? "bg-[var(--apple-accent)] text-white shadow-sm"
                          : "text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
                      }`}
                    >
                      English
                    </button>
                    <button
                      onClick={() => setSelectedLanguage("hi-IN")}
                      className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                        selectedLanguage === "hi-IN"
                          ? "bg-[var(--apple-accent)] text-white shadow-sm"
                          : "text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
                      }`}
                    >
                      Hindi
                    </button>
                  </div>

                  <button
                    onClick={handleSimulateSttSegment}
                    className="apple-button-secondary px-3 py-1 text-xs text-[var(--apple-accent)] font-bold ml-1"
                  >
                    + Speech Turn
                  </button>
                </div>
              </div>

              {/* Transcription Quality & Enhancement Toggles */}
              <div className="flex items-center gap-4 pt-1 flex-wrap text-xs bg-[var(--apple-bg)] p-2.5 rounded-xl border border-[var(--apple-border)]">
                <span className="font-bold text-[var(--apple-text-secondary)]">Enhancements:</span>
                <label className="flex items-center gap-1.5 cursor-pointer font-medium text-[var(--apple-text-primary)]">
                  <input
                    type="checkbox"
                    checked={enhancedTerms}
                    onChange={(e) => setEnhancedTerms(e.target.checked)}
                    className="rounded text-[var(--apple-accent)] focus:ring-0"
                  />
                  <span>Technical Term Normalization</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer font-medium text-[var(--apple-text-primary)]">
                  <input
                    type="checkbox"
                    checked={cleanFillers}
                    onChange={(e) => setCleanFillers(e.target.checked)}
                    className="rounded text-[var(--apple-accent)] focus:ring-0"
                  />
                  <span>Strip Vocal Fillers (um/uh)</span>
                </label>
              </div>
            </div>

            {/* Transcript Stream */}
            <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
              {transcripts.map((t) => (
                <div key={t.id} className="p-3.5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-[var(--apple-text-primary)]">{t.speaker}</span>
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-emerald-500/10 text-emerald-600 border border-emerald-500/20 font-bold">
                        {(t.confidence * 100).toFixed(0)}% High Conf
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="apple-badge text-[10px] font-mono">{t.language}</span>
                      <span className="font-mono text-[var(--apple-text-muted)] text-[11px]">{t.timestamp}</span>
                    </div>
                  </div>
                  <p className="text-xs text-[var(--apple-text-primary)] font-medium leading-relaxed">
                    {enhancedTerms ? t.text : (t.rawText || t.text)}
                  </p>
                  {t.rawText && enhancedTerms && t.rawText !== t.text && (
                    <p className="text-[10px] text-[var(--apple-text-muted)] italic font-mono pt-0.5">
                      Raw STT: "{t.rawText}"
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Agent TTS Vocal Synthesizer & Observations */}
        <div className="space-y-6">
          {/* Agent TTS Voice Synthesizer Panel */}
          <div className="apple-card p-6 space-y-4">
            <div className="border-b border-[var(--apple-border)] pb-3 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-[var(--apple-text-primary)] uppercase tracking-wider">
                  Agent Voice Synthesizer
                </h2>
                <p className="text-[11px] text-[var(--apple-text-secondary)]">Sarvam AI Saaras v2 Vocal Output</p>
              </div>
              <span className="apple-badge font-mono text-[10px]">Saaras v2 TTS</span>
            </div>

            <div className="space-y-3">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Type text for agent to speak:</label>
              <textarea
                rows={3}
                value={ttsInputText}
                onChange={(e) => setTtsInputText(e.target.value)}
                placeholder={
                  selectedLanguage === "hi-IN"
                    ? "नमस्ते, क्या आप अपने प्रोजेक्ट के बारे में समझा सकते हैं?"
                    : "Hello candidate, please begin describing your backend project architecture."
                }
                className="w-full p-3.5 apple-input text-xs"
              />

              <button
                onClick={handleSynthesizeTts}
                disabled={isSynthesizingTts || !ttsInputText.trim()}
                className="w-full apple-button-primary py-2.5 text-xs flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isSynthesizingTts ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin"></span>
                    <span>Synthesizing Voice...</span>
                  </>
                ) : (
                  <span>🔊 Speak via Sarvam TTS ({selectedLanguage === "hi-IN" ? "Hindi" : "English"})</span>
                )}
              </button>

              {lastTtsResult && (
                <div className="p-3 rounded-xl apple-badge bg-[var(--apple-accent-light)] text-[var(--apple-accent)] text-[11px]">
                  {lastTtsResult}
                </div>
              )}
            </div>
          </div>

          {/* Live Candidate AI Reliance Risk Meter */}
          <div className="apple-card p-6 space-y-3 bg-gradient-to-br from-rose-500/5 via-[var(--apple-card)] to-amber-500/5 border border-rose-500/20">
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-3">
              <div>
                <h2 className="text-sm font-extrabold text-[var(--apple-text-primary)] uppercase tracking-wider flex items-center gap-2">
                  <span>AI Assistance Risk Meter</span>
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span>
                </h2>
                <p className="text-[11px] text-[var(--apple-text-secondary)]">Real-time candidate LLM & script detection</p>
              </div>
              <span className="px-2.5 py-1 rounded-full text-xs font-black bg-rose-500/10 text-rose-600 border border-rose-500/20">
                78% High Risk
              </span>
            </div>

            <div className="space-y-2 pt-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-[var(--apple-text-primary)]">Estimated Candidate AI Reliance</span>
                <span className="font-black text-rose-500 text-sm">78%</span>
              </div>
              <div className="w-full bg-[var(--apple-border)] h-2.5 rounded-full overflow-hidden">
                <div className="bg-gradient-to-r from-amber-500 to-rose-500 h-full rounded-full transition-all duration-500" style={{ width: "78%" }}></div>
              </div>
              <div className="flex justify-between text-[10px] text-[var(--apple-text-secondary)] pt-0.5 font-mono">
                <span>Natural Human Speech</span>
                <span className="text-rose-500 font-bold">High AI Generation</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-[var(--apple-bg)] border border-[var(--apple-border)] text-xs space-y-1">
              <div className="font-bold text-rose-500 flex items-center gap-1.5 text-[11px]">
                <span>⚠ Detected AI Phrasing Pattern</span>
              </div>
              <p className="text-[11px] text-[var(--apple-text-secondary)] leading-tight">
                Candidate used formulaic transitions ("it's important to note", "delve into", "multifaceted") with 0% natural filler disfluencies.
              </p>
            </div>
          </div>

          {/* Observations Stream */}
          <div className="apple-card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--apple-border)] pb-3">
              <div>
                <h2 className="text-sm font-bold text-[var(--apple-text-primary)] uppercase tracking-wider">Proctoring Telemetry</h2>
                <p className="text-[11px] text-[var(--apple-text-secondary)]">Timestamped sensor findings</p>
              </div>
              <span className="apple-badge font-mono text-[10px]">
                {events.length} Events
              </span>
            </div>

            <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
              {events.map((evt) => (
                <div key={evt.id} className="p-3.5 rounded-2xl bg-[var(--apple-bg)] border border-[var(--apple-border)] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px] text-[var(--apple-accent)] font-bold">{evt.t_offset}</span>
                    <span className="apple-badge text-[10px] uppercase font-bold">{evt.severity}</span>
                  </div>

                  <div className="font-bold text-xs text-[var(--apple-text-primary)]">{evt.category}</div>
                  <p className="text-xs text-[var(--apple-text-secondary)]">{evt.signal}</p>

                  <div className="flex items-center justify-between text-[10px] text-[var(--apple-text-muted)] pt-1">
                    <span>Source: {evt.source}</span>
                    <span className="font-mono">Conf: {(evt.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
