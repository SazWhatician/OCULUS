"use client";

export interface CaptureTelemetry {
  active: boolean;
  sessionId: string | null;
  videoActive: boolean;
  audioActive: boolean;
  chunksSent: number;
  framesSent: number;
  error: string | null;
}

type Listener = (state: CaptureTelemetry) => void;

class WebRTCCaptureManager {
  private state: CaptureTelemetry = {
    active: false,
    sessionId: null,
    videoActive: false,
    audioActive: false,
    chunksSent: 0,
    framesSent: 0,
    error: null,
  };

  private listeners = new Set<Listener>();
  private ws: WebSocket | null = null;
  private recorder: MediaRecorder | null = null;
  private displayStream: MediaStream | null = null;
  private micStream: MediaStream | null = null;
  private audioStream: MediaStream | null = null;
  private frameTimer: ReturnType<typeof setInterval> | null = null;
  private videoEl: HTMLVideoElement | null = null;
  private previewEl: HTMLVideoElement | null = null;
  private startedAtMs = 0;

  getState(): CaptureTelemetry {
    return { ...this.state };
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.getState());
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(patch: Partial<CaptureTelemetry>) {
    this.state = { ...this.state, ...patch };
    const snap = this.getState();
    this.listeners.forEach((fn) => fn(snap));
  }

  attachPreview(el: HTMLVideoElement | null) {
    this.previewEl = el;
    if (el && this.displayStream) {
      el.srcObject = this.displayStream;
      el.muted = true;
      el.play().catch(() => {});
    }
  }

  async start(sessionId: string, wsBaseUrl: string): Promise<void> {
    await this.stop();

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getDisplayMedia) {
      throw new Error("Screen and audio capture requires a modern browser over HTTPS or localhost.");
    }

    const displayStream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 5, max: 15 }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: true,
    });
    this.displayStream = displayStream;

    // Fallback to microphone if user did not share system/tab audio
    let audioTracks = displayStream.getAudioTracks();
    if (audioTracks.length === 0) {
      try {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.micStream = mic;
        audioTracks = mic.getAudioTracks();
      } catch {
        // Proceed video-only if mic permission denied
      }
    }

    const videoTracks = displayStream.getVideoTracks();
    const audioStream = audioTracks.length > 0 ? new MediaStream(audioTracks) : null;
    this.audioStream = audioStream;

    const ws = new WebSocket(`${wsBaseUrl}/ws/ingest/${sessionId}`);
    this.ws = ws;
    this.startedAtMs = Date.now();

    this.emit({
      active: true,
      sessionId,
      videoActive: videoTracks.length > 0,
      audioActive: audioTracks.length > 0,
      chunksSent: 0,
      framesSent: 0,
      error: null,
    });

    if (this.previewEl) {
      this.previewEl.srcObject = displayStream;
      this.previewEl.muted = true;
      this.previewEl.play().catch(() => {});
    }

    videoTracks.forEach((track) => {
      track.addEventListener("ended", () => {
        if (this.state.sessionId === sessionId && this.state.active) {
          void this.stop();
        }
      });
    });

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("WebSocket ingest connection timed out.")), 8000);
      ws.onopen = () => {
        clearTimeout(timeout);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("Unable to connect to live ingest WebSocket."));
      };
    });

    ws.onclose = () => {
      if (this.state.active && this.state.sessionId === sessionId) {
        this.emit({ error: "Ingest WebSocket closed" });
      }
    };

    // Start 5-second audio chunk recording
    const recordTarget = audioStream || displayStream;
    const preferredMimes = audioStream
      ? ["audio/webm;codecs=opus", "audio/webm", "video/webm;codecs=vp8,opus", "video/webm"]
      : ["video/webm;codecs=vp8,opus", "video/webm"];
    const mimeType = preferredMimes.find((m) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m));

    const recorder = mimeType
      ? new MediaRecorder(recordTarget, { mimeType })
      : new MediaRecorder(recordTarget);
    this.recorder = recorder;

    recorder.ondataavailable = async (ev: BlobEvent) => {
      if (!ev.data || ev.data.size === 0 || ws.readyState !== WebSocket.OPEN) return;
      try {
        const buf = await ev.data.arrayBuffer();
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(buf);
          this.emit({ chunksSent: this.state.chunksSent + 1 });
        }
      } catch (err) {
        this.emit({ error: String(err) });
      }
    };
    recorder.start(5000);

    // Start 1 FPS JPEG canvas frame sampling
    if (videoTracks.length > 0) {
      const hiddenVideo = document.createElement("video");
      hiddenVideo.srcObject = displayStream;
      hiddenVideo.muted = true;
      hiddenVideo.playsInline = true;
      this.videoEl = hiddenVideo;
      await hiddenVideo.play().catch(() => {});

      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext("2d");

      this.frameTimer = setInterval(() => {
        if (!ctx || ws.readyState !== WebSocket.OPEN || hiddenVideo.readyState < 2) return;
        try {
          ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.72);
          ws.send(
            JSON.stringify({
              kind: "frame",
              b64: dataUrl,
              ts_ms: Math.max(0, Date.now() - this.startedAtMs),
            }),
          );
          this.emit({ framesSent: this.state.framesSent + 1 });
        } catch {
          // Ignore transient canvas frame errors
        }
      }, 1000);
    }
  }

  async stop(): Promise<void> {
    if (this.frameTimer) {
      clearInterval(this.frameTimer);
      this.frameTimer = null;
    }
    if (this.recorder && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch {
        // ignore
      }
    }
    this.recorder = null;

    if (this.videoEl) {
      this.videoEl.srcObject = null;
      this.videoEl = null;
    }
    if (this.previewEl) {
      this.previewEl.srcObject = null;
    }

    [this.displayStream, this.micStream, this.audioStream].forEach((stream) => {
      stream?.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch {
          // ignore
        }
      });
    });
    this.displayStream = null;
    this.micStream = null;
    this.audioStream = null;

    const ws = this.ws;
    this.ws = null;
    if (ws) {
      try {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ kind: "end" }));
        }
        ws.close(1000, "Session capture stopped");
      } catch {
        // ignore
      }
    }

    this.emit({
      active: false,
      sessionId: null,
      videoActive: false,
      audioActive: false,
    });
  }
}

export const webrtcCapture = new WebRTCCaptureManager();
