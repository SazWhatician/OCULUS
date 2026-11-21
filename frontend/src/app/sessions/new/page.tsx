"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function NewSessionPage() {
  const router = useRouter();
  const [meetingUrl, setMeetingUrl] = useState("");
  const [platform, setPlatform] = useState("zoom");
  const [candidateName, setCandidateName] = useState("");
  const [candidateEmail, setCandidateEmail] = useState("");
  const [interviewerName, setInterviewerName] = useState("");
  const [roleTitle, setRoleTitle] = useState("");
  const [botDisplayName, setBotDisplayName] = useState("VeriMeet Proctor Bot");
  
  // Sarvam AI Saaras v2 Language & Voice configuration (English & Hindi)
  const [preferredLanguage, setPreferredLanguage] = useState("en-IN");
  const [selectedVoice, setSelectedVoice] = useState("meera");
  
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Settings
  const [enablePerParticipantAudio, setEnablePerParticipantAudio] = useState(true);
  const [enableSarvamTts, setEnableSarvamTts] = useState(true);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const res = await fetch("http://localhost:8005/api/sessions/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          meeting_url: meetingUrl,
          candidate_name: candidateName || "Candidate",
          interviewer_name: interviewerName || "Interviewer",
          role_title: roleTitle || "Software Engineer",
          bot_display_name: botDisplayName,
          platform: platform,
          language_code: preferredLanguage,
          voice_name: selectedVoice
        })
      });

      if (!res.ok) {
        throw new Error(`Dispatch failed: ${res.statusText}`);
      }

      const data = await res.json();
      const botId = data.dispatch?.bot_id || data.session_id;
      router.push(`/sessions/${botId}`);
    } catch (err) {
      console.error("Failed to dispatch bot:", err);
      // Fallback redirect for UI preview if backend is offline
      router.push("/sessions/session-zoom-8902");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="space-y-2 text-center md:text-left">
        <div className="inline-flex items-center gap-2 apple-badge">
          <span className="w-2 h-2 rounded-full bg-[var(--apple-accent)] live-pulse"></span>
          Sarvam AI Saaras v2 Speech Configurator
        </div>
        <h1 className="text-3xl font-extrabold text-[var(--apple-text-primary)] tracking-tight">
          Dispatch Interview Agent
        </h1>
        <p className="text-[var(--apple-text-secondary)] text-sm max-w-xl">
          Enter meeting details, bind pre-registered candidate identity, and configure Sarvam AI Saaras v2 English / Hindi speech agent.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Card 1: Meeting Target */}
        <div className="apple-card p-6 space-y-4">
          <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">1. Meeting Details & Platform</h2>

          <div className="space-y-2">
            <label className="text-xs font-bold text-[var(--apple-text-primary)]">
              Meeting URL (Zoom / Google Meet / Teams) <span className="text-rose-500">*</span>
            </label>
            <input
              type="url"
              required
              placeholder="https://zoom.us/j/9876543210?pwd=... or https://meet.google.com/abc-defg-hij"
              value={meetingUrl}
              onChange={(e) => setMeetingUrl(e.target.value)}
              className="w-full px-4 py-3 apple-input text-sm"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Target Platform</label>
              <select
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm text-[var(--apple-text-primary)] font-semibold cursor-pointer"
              >
                <option value="zoom">Zoom (SDK / Ingest)</option>
                <option value="google_meet">Google Meet</option>
                <option value="teams">Microsoft Teams</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Bot Display Name</label>
              <input
                type="text"
                required
                value={botDisplayName}
                onChange={(e) => setBotDisplayName(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm"
              />
            </div>
          </div>
        </div>

        {/* Card 2: Sarvam AI Saaras v2 Speech Config */}
        <div className="apple-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">
              2. Sarvam AI Saaras v2 Speech Engine Config
            </h2>
            <span className="apple-badge font-mono">Saaras v2 Engine</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Language Selector */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">
                Preferred Transcription Language <span className="text-rose-500">*</span>
              </label>
              <div className="grid grid-cols-2 gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => setPreferredLanguage("en-IN")}
                  className={`p-4 rounded-2xl border text-xs font-bold flex flex-col items-center justify-center gap-1 transition-all ${
                    preferredLanguage === "en-IN"
                      ? "border-[var(--apple-accent)] bg-[var(--apple-accent-light)] text-[var(--apple-accent)] shadow-sm"
                      : "border-[var(--apple-border)] bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
                  }`}
                >
                  <span className="text-base">🇬🇧 English</span>
                  <span className="text-[10px] opacity-80">English (India)</span>
                </button>

                <button
                  type="button"
                  onClick={() => setPreferredLanguage("hi-IN")}
                  className={`p-4 rounded-2xl border text-xs font-bold flex flex-col items-center justify-center gap-1 transition-all ${
                    preferredLanguage === "hi-IN"
                      ? "border-[var(--apple-accent)] bg-[var(--apple-accent-light)] text-[var(--apple-accent)] shadow-sm"
                      : "border-[var(--apple-border)] bg-[var(--apple-bg)] text-[var(--apple-text-secondary)] hover:text-[var(--apple-text-primary)]"
                  }`}
                >
                  <span className="text-base">🇮🇳 Hindi</span>
                  <span className="text-[10px] opacity-80">हिंदी (Hindi)</span>
                </button>
              </div>
              <p className="text-[11px] text-[var(--apple-text-secondary)] pt-1">
                Sarvam AI Saaras v2 will transcribe meeting audio in your selected language.
              </p>
            </div>

            {/* Voice Selector */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Agent Voice (TTS Synthesizer)</label>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm text-[var(--apple-text-primary)] font-semibold cursor-pointer"
              >
                <option value="meera">Meera (Female - Natural)</option>
                <option value="pavitra">Pavitra (Female - Clear)</option>
                <option value="mahesh">Mahesh (Male - Professional)</option>
                <option value="arvind">Arvind (Male - Deep)</option>
              </select>
              <p className="text-[11px] text-[var(--apple-text-secondary)] pt-1">
                Voice announcements will be generated using Sarvam AI Saaras v2 TTS.
              </p>
            </div>
          </div>
        </div>

        {/* Card 3: Pre-Registration Identity */}
        <div className="apple-card p-6 space-y-4">
          <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">3. Candidate Identity Binding (Tier 1)</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Candidate Name <span className="text-rose-500">*</span></label>
              <input
                type="text"
                required
                placeholder="e.g. Alex Rivers"
                value={candidateName}
                onChange={(e) => setCandidateName(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Candidate Email</label>
              <input
                type="email"
                placeholder="alex.rivers@example.com"
                value={candidateEmail}
                onChange={(e) => setCandidateEmail(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Interviewer Name <span className="text-rose-500">*</span></label>
              <input
                type="text"
                required
                placeholder="e.g. Sarah Jenkins"
                value={interviewerName}
                onChange={(e) => setInterviewerName(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-[var(--apple-text-primary)]">Role / Position Title</label>
              <input
                type="text"
                placeholder="e.g. Senior Backend Engineer"
                value={roleTitle}
                onChange={(e) => setRoleTitle(e.target.value)}
                className="w-full px-4 py-3 apple-input text-sm"
              />
            </div>
          </div>
        </div>

        {/* Card 4: Audio Toggles */}
        <div className="apple-card p-6 space-y-4">
          <h2 className="text-xs font-bold text-[var(--apple-text-secondary)] uppercase tracking-wider">4. Pipeline Toggles</h2>

          <div className="space-y-3 text-sm">
            <label className="flex items-center justify-between p-3.5 apple-input cursor-pointer hover:border-[var(--apple-border-hover)] transition-all">
              <div>
                <div className="font-bold text-[var(--apple-text-primary)]">Per-Participant Discrete Audio</div>
                <div className="text-xs text-[var(--apple-text-secondary)]">Direct microphone stream isolation for Sarvam AI STT attribution.</div>
              </div>
              <input
                type="checkbox"
                checked={enablePerParticipantAudio}
                onChange={(e) => setEnablePerParticipantAudio(e.target.checked)}
                className="w-4 h-4 accent-[var(--apple-accent)] rounded"
              />
            </label>

            <label className="flex items-center justify-between p-3.5 apple-input cursor-pointer hover:border-[var(--apple-border-hover)] transition-all">
              <div>
                <div className="font-bold text-[var(--apple-text-primary)]">Enable Agent Voice Synthesis (Sarvam TTS)</div>
                <div className="text-xs text-[var(--apple-text-secondary)]">Enables bot vocal greeting upon joining session.</div>
              </div>
              <input
                type="checkbox"
                checked={enableSarvamTts}
                onChange={(e) => setEnableSarvamTts(e.target.checked)}
                className="w-4 h-4 accent-[var(--apple-accent)] rounded"
              />
            </label>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-4 pt-2">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="apple-button-secondary px-6 py-3 text-sm"
          >
            Cancel
          </button>

          <button
            type="submit"
            disabled={isSubmitting}
            className="apple-button-primary px-8 py-3 text-sm flex items-center gap-2 disabled:opacity-50"
          >
            {isSubmitting ? (
              <>
                <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></span>
                <span>Dispatching Agent...</span>
              </>
            ) : (
              <span>Dispatch Agent Bot →</span>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
