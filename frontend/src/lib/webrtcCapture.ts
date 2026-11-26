"use client";

export interface CaptureTelemetry {
  active: boolean;
  sessionId: string | null;
  videoActive: boolean;
  audioActive: boolean;
  micActive: boolean;
  tabAudioActive: boolean;
  reconnecting: boolean;
  vadSpeaking: boolean;
  audioLevel: number;
  chunksSent: number;
  skippedSilentChunks: number;
  framesSent: number;
  droppedFrames: number;
  error: string | null;
}

type Listener = (state: CaptureTelemetry) => void;

// ---------------------------------------------------------------------------
// DSP & Transport Constants
// ---------------------------------------------------------------------------
const TARGET_SAMPLE_RATE = 16_000;          // Native Whisper / Speechmatics rate (32 KB/s mono PCM16)
const VAD_FRAME_SAMPLES = 320;              // 20ms @ 16kHz
const FFT_SIZE = 64;                        // 250Hz bin resolution @ 16kHz (bins 1..14 = 250-3500Hz voice band)
const MIN_CHUNK_SEC = 4.0;
const MAX_CHUNK_SEC = 8.0;
const PRE_ROLL_MS = 350;                    // Seam overlap preserved for stitcher.py deduplication
const SILENCE_SPLIT_FRAMES = 12;            // 240ms natural valley triggers clean turn cut
const MIN_SPEECH_FRAMES_TO_SEND = 12;       // 240ms speech required to emit chunk (skips pure room noise)
const VAD_ONSET_FRAMES = 2;                 // 40ms onset debounce (rejects keystroke clicks)
const VAD_HANGOVER_FRAMES = 18;             // 360ms tail hangover (preserves word endings)

const MAX_VIDEO_WIDTH = 640;
const MAX_VIDEO_HEIGHT = 360;
const FRAME_INTERVAL_MS = 1_000;
const MAX_WS_BUFFERED_BYTES = 512 * 1024;   // 512 KB backpressure guard
const MAX_PENDING_AUDIO_CHUNKS = 6;
const MAX_WS_RECONNECT_ATTEMPTS = 5;

// Inline zero-dependency AudioWorkletProcessor (zero-copy Transferable Float32Array blocks)
const PCM_WORKLET_SOURCE = `
class OculusPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunkSize = 1024;
    this._buf = new Float32Array(this._chunkSize);
    this._idx = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) return true;
    const ch1 = input.length > 1 ? input[1] : null;

    for (let i = 0; i < ch0.length; i++) {
      const s = ch1 ? 0.5 * (ch0[i] + ch1[i]) : ch0[i];
      // Soft tanh limiter prevents hard digital clipping on summed dual-source peaks
      this._buf[this._idx++] = Math.tanh(s);
      if (this._idx === this._chunkSize) {
        const out = this._buf;
        this.port.postMessage(out, [out.buffer]);
        this._buf = new Float32Array(this._chunkSize);
        this._idx = 0;
      }
    }
    return true;
  }
}
registerProcessor("oculus-pcm-processor", OculusPcmProcessor);
`;

// Inline Web Worker timer: immune to Chromium/Firefox background-tab setInterval throttling
const TICK_WORKER_SOURCE = `
let timer = null;
self.onmessage = (e) => {
  if (e.data && e.data.cmd === "start") {
    if (timer) clearInterval(timer);
    timer = setInterval(() => self.postMessage(Date.now()), e.data.intervalMs || 1000);
  } else if (e.data && e.data.cmd === "stop") {
    if (timer) clearInterval(timer);
    timer = null;
  }
};
`;

// ---------------------------------------------------------------------------
// Polyphase Anti-Aliased Lanczos-3 Resampler (f_in -> 16 kHz, Zero Sin in Loop)
// ---------------------------------------------------------------------------
class PolyphaseLanczosResampler {
  private readonly ratio: number;
  private readonly a = 3;
  private readonly phases = 64;
  private readonly halfSpan: number;
  private readonly kernelWidth: number;
  private readonly lut: Float32Array[];
  private tail: Float32Array = new Float32Array(0);
  private fracPos = 0;

  constructor(private readonly inputRate: number, private readonly outputRate: number = TARGET_SAMPLE_RATE) {
    this.ratio = inputRate / outputRate;
    const scale = Math.max(1.0, this.ratio);
    this.halfSpan = Math.ceil(this.a * scale);
    this.kernelWidth = 2 * this.halfSpan + 1;
    this.lut = new Array<Float32Array>(this.phases);

    for (let p = 0; p < this.phases; p++) {
      const frac = p / this.phases;
      const kernel = new Float32Array(this.kernelWidth);
      let sum = 0;
      for (let k = -this.halfSpan; k <= this.halfSpan; k++) {
        const x = (k - frac) / scale;
        const w = PolyphaseLanczosResampler.lanczos(x, this.a);
        kernel[k + this.halfSpan] = w;
        sum += w;
      }
      if (sum > 1e-8) {
        for (let i = 0; i < this.kernelWidth; i++) kernel[i] /= sum;
      }
      this.lut[p] = kernel;
    }
  }

  private static lanczos(x: number, a: number): number {
    const ax = Math.abs(x);
    if (ax < 1e-7) return 1.0;
    if (ax >= a) return 0.0;
    const pix = Math.PI * x;
    return (Math.sin(pix) / pix) * (Math.sin(pix / a) / (pix / a));
  }

  process(input: Float32Array): Float32Array {
    if (input.length === 0) return new Float32Array(0);
    if (Math.abs(this.ratio - 1.0) < 1e-6) return input;

    // Prepend previous tail so filter state stays continuous across AudioWorklet blocks
    const combined = new Float32Array(this.tail.length + input.length);
    combined.set(this.tail, 0);
    combined.set(input, this.tail.length);

    const maxCenter = combined.length - this.halfSpan - 1;
    if (maxCenter <= this.halfSpan) {
      this.tail = combined;
      return new Float32Array(0);
    }

    let pos = Math.max(this.halfSpan, this.fracPos);
    const estOut = Math.max(1, Math.floor((maxCenter - pos) / this.ratio) + 1);
    const out = new Float32Array(estOut);
    let outIdx = 0;

    while (pos < maxCenter && outIdx < estOut) {
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const phaseIdx = Math.min(this.phases - 1, Math.floor(frac * this.phases));
      const kernel = this.lut[phaseIdx];

      let acc = 0;
      const base = i0 - this.halfSpan;
      for (let j = 0; j < this.kernelWidth; j++) {
        acc += combined[base + j] * kernel[j];
      }
      out[outIdx++] = acc;
      pos += this.ratio;
    }

    const consumed = Math.max(0, Math.floor(pos) - this.halfSpan);
    this.tail = combined.slice(consumed);
    this.fracPos = pos - consumed;
    return outIdx === out.length ? out : out.subarray(0, outIdx);
  }
}

// ---------------------------------------------------------------------------
// Lightweight 64-Point Radix-2 FFT for Wiener Spectral Flatness & Voice-Band VAD
// ---------------------------------------------------------------------------
class FastVoiceSpectralAnalyzer {
  private readonly hann = new Float32Array(FFT_SIZE);
  private readonly cosTable = new Float32Array(FFT_SIZE / 2);
  private readonly sinTable = new Float32Array(FFT_SIZE / 2);
  private readonly bitRev = new Uint8Array(FFT_SIZE);
  private readonly re = new Float32Array(FFT_SIZE);
  private readonly im = new Float32Array(FFT_SIZE);

  constructor() {
    const bits = Math.log2(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) {
      this.hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
      let rev = 0;
      for (let b = 0; b < bits; b++) rev = (rev << 1) | ((i >> b) & 1);
      this.bitRev[i] = rev;
    }
    for (let i = 0; i < FFT_SIZE / 2; i++) {
      const ang = (-2 * Math.PI * i) / FFT_SIZE;
      this.cosTable[i] = Math.cos(ang);
      this.sinTable[i] = Math.sin(ang);
    }
  }

  analyze(frame: Float32Array): { voiceBandRatio: number; spectralFlatness: number } {
    // Subsample center 64-sample window (320 -> 64) or 4-window averaged power spectrum
    const bins = FFT_SIZE / 2;
    const power = new Float32Array(bins);
    const windows = Math.max(1, Math.floor(frame.length / FFT_SIZE));

    for (let w = 0; w < windows; w++) {
      const offset = w * FFT_SIZE;
      for (let i = 0; i < FFT_SIZE; i++) {
        this.re[this.bitRev[i]] = frame[offset + i] * this.hann[i];
        this.im[this.bitRev[i]] = 0;
      }

      for (let size = 2; size <= FFT_SIZE; size <<= 1) {
        const half = size >> 1;
        const step = FFT_SIZE / size;
        for (let i = 0; i < FFT_SIZE; i += size) {
          for (let j = 0; j < half; j++) {
            const tw = j * step;
            const wr = this.cosTable[tw];
            const wi = this.sinTable[tw];
            const k = i + j + half;
            const tr = wr * this.re[k] - wi * this.im[k];
            const ti = wr * this.im[k] + wi * this.re[k];
            this.re[k] = this.re[i + j] - tr;
            this.im[k] = this.im[i + j] - ti;
            this.re[i + j] += tr;
            this.im[i + j] += ti;
          }
        }
      }

      for (let k = 1; k < bins; k++) {
        power[k] += this.re[k] * this.re[k] + this.im[k] * this.im[k];
      }
    }

    // Bins 1..14 cover 250 Hz .. 3500 Hz (human vocal formant band)
    let totalPower = 1e-12;
    let voicePower = 0;
    let logSum = 0;
    let arithSum = 0;
    const sfmBins = 14;

    for (let k = 1; k < bins; k++) {
      const p = power[k] / windows + 1e-12;
      totalPower += p;
      if (k <= sfmBins) {
        voicePower += p;
        logSum += Math.log(p);
        arithSum += p;
      }
    }

    const geomMean = Math.exp(logSum / sfmBins);
    const arithMean = arithSum / sfmBins;
    return {
      voiceBandRatio: voicePower / totalPower,
      spectralFlatness: Math.min(1.0, geomMean / (arithMean + 1e-12)),
    };
  }
}

// ---------------------------------------------------------------------------
// 3-Feature Adaptive VAD & Minimum-Energy Boundary Chunker
// ---------------------------------------------------------------------------
class AdaptiveVadChunker {
  private readonly spectral = new FastVoiceSpectralAnalyzer();
  private readonly minFrames = Math.round((MIN_CHUNK_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SAMPLES);
  private readonly maxFrames = Math.round((MAX_CHUNK_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SAMPLES);
  private readonly preRollFrames = Math.round(((PRE_ROLL_MS / 1000) * TARGET_SAMPLE_RATE) / VAD_FRAME_SAMPLES);

  private pendingSamples: Float32Array = new Float32Array(0);
  private frames: Float32Array[] = [];
  private frameRms: number[] = [];
  private speechFrameCount = 0;
  private consecutiveSilence = 0;
  private consecutiveOnset = 0;
  private hangoverRemaining = 0;
  private noiseFloorDb = -52.0;
  private speaking = false;
  private latestLevel = 0;

  get isSpeaking(): boolean {
    return this.speaking;
  }

  get audioLevel(): number {
    return this.latestLevel;
  }

  push(samples16k: Float32Array): { wavs: ArrayBuffer[]; skippedSilent: number } {
    const wavs: ArrayBuffer[] = [];
    let skippedSilent = 0;

    if (samples16k.length === 0) return { wavs, skippedSilent };

    const merged = new Float32Array(this.pendingSamples.length + samples16k.length);
    merged.set(this.pendingSamples, 0);
    merged.set(samples16k, this.pendingSamples.length);

    let offset = 0;
    while (offset + VAD_FRAME_SAMPLES <= merged.length) {
      const frame = merged.slice(offset, offset + VAD_FRAME_SAMPLES);
      offset += VAD_FRAME_SAMPLES;

      const { rms, isSpeech } = this.classifyFrame(frame);
      this.frames.push(frame);
      this.frameRms.push(rms);

      if (isSpeech) {
        this.speechFrameCount++;
        this.consecutiveSilence = 0;
      } else {
        this.consecutiveSilence++;
      }

      const n = this.frames.length;

      // Clean silence-valley split between 4s and 8s (prevents cutting words mid-syllable)
      if (
        n >= this.minFrames &&
        this.speechFrameCount >= MIN_SPEECH_FRAMES_TO_SEND &&
        this.consecutiveSilence >= SILENCE_SPLIT_FRAMES
      ) {
        const splitIdx = Math.max(this.minFrames, n - Math.floor(this.consecutiveSilence / 2));
        wavs.push(this.emitAndSlide(splitIdx, 0));
      } else if (n >= this.maxFrames) {
        if (this.speechFrameCount >= MIN_SPEECH_FRAMES_TO_SEND) {
          // Force-cut at lowest-RMS valley in the trailing 1.0s + retain 350ms overlap for stitcher.py
          const searchStart = Math.max(this.minFrames, n - 50);
          let bestIdx = n;
          let minRms = Number.POSITIVE_INFINITY;
          for (let i = searchStart; i < n; i++) {
            if (this.frameRms[i] < minRms) {
              minRms = this.frameRms[i];
              bestIdx = i;
            }
          }
          wavs.push(this.emitAndSlide(bestIdx, this.preRollFrames));
        } else {
          // Pure silence window: drop without burning free-tier STT quota; keep pre-roll for next onset
          this.retainTail(this.preRollFrames);
          skippedSilent++;
        }
      }
    }

    this.pendingSamples = merged.slice(offset);
    return { wavs, skippedSilent };
  }

  flush(): ArrayBuffer | null {
    if (this.speechFrameCount < MIN_SPEECH_FRAMES_TO_SEND || this.frames.length === 0) {
      this.reset();
      return null;
    }
    const wav = encodeWav16kMono(this.concatFrames(this.frames));
    this.reset();
    return wav;
  }

  reset(): void {
    this.pendingSamples = new Float32Array(0);
    this.frames = [];
    this.frameRms = [];
    this.speechFrameCount = 0;
    this.consecutiveSilence = 0;
    this.consecutiveOnset = 0;
    this.hangoverRemaining = 0;
    this.speaking = false;
    this.latestLevel = 0;
  }

  private classifyFrame(frame: Float32Array): { rms: number; isSpeech: boolean } {
    let sumSq = 0;
    let zeroCrossings = 0;
    for (let i = 0; i < frame.length; i++) {
      const s = frame[i];
      sumSq += s * s;
      if (i > 0 && ((s >= 0 && frame[i - 1] < 0) || (s < 0 && frame[i - 1] >= 0))) {
        zeroCrossings++;
      }
    }

    const rms = Math.sqrt(sumSq / frame.length);
    const db = 20 * Math.log10(rms + 1e-7);
    const zcr = zeroCrossings / (frame.length - 1);
    this.latestLevel = Math.min(1.0, Math.max(0.0, (db + 60) / 50));

    // Fast path: skip FFT on dead-quiet frames
    let rawActive = false;
    if (db > Math.max(-46.0, this.noiseFloorDb + 7.0) && zcr >= 0.01 && zcr <= 0.48) {
      const { voiceBandRatio, spectralFlatness } = this.spectral.analyze(frame);
      rawActive = voiceBandRatio >= 0.44 && spectralFlatness <= 0.74;
    }

    // Asymmetric noise-floor estimator (fast drop, ultra-slow rise during non-speech)
    if (!rawActive) {
      const alpha = db < this.noiseFloorDb ? 0.08 : 0.004;
      this.noiseFloorDb = Math.max(-65.0, Math.min(-24.0, (1 - alpha) * this.noiseFloorDb + alpha * db));
    }

    // Hysteresis onset/hangover state machine
    if (rawActive) {
      this.consecutiveOnset++;
      if (this.consecutiveOnset >= VAD_ONSET_FRAMES) {
        this.speaking = true;
        this.hangoverRemaining = VAD_HANGOVER_FRAMES;
      }
    } else {
      this.consecutiveOnset = 0;
      if (this.hangoverRemaining > 0) {
        this.hangoverRemaining--;
      } else {
        this.speaking = false;
      }
    }

    return { rms, isSpeech: this.speaking };
  }

  private emitAndSlide(splitIdx: number, overlapFrames: number): ArrayBuffer {
    const outFrames = this.frames.slice(0, splitIdx);
    const wav = encodeWav16kMono(this.concatFrames(outFrames));

    const keepFrom = Math.max(0, splitIdx - overlapFrames);
    this.frames = this.frames.slice(keepFrom);
    this.frameRms = this.frameRms.slice(keepFrom);
    this.speechFrameCount = 0;
    this.consecutiveSilence = 0;
    return wav;
  }

  private retainTail(count: number): void {
    const start = Math.max(0, this.frames.length - count);
    this.frames = this.frames.slice(start);
    this.frameRms = this.frameRms.slice(start);
    this.speechFrameCount = 0;
    this.consecutiveSilence = 0;
  }

  private concatFrames(list: Float32Array[]): Float32Array {
    let total = 0;
    for (let i = 0; i < list.length; i++) total += list[i].length;
    const out = new Float32Array(total);
    let pos = 0;
    for (let i = 0; i < list.length; i++) {
      out.set(list[i], pos);
      pos += list[i].length;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Standalone RIFF/WAVE 16kHz Mono PCM16 Encoder (Every Chunk Self-Contained)
// ---------------------------------------------------------------------------
function encodeWav16kMono(samples: Float32Array): ArrayBuffer {
  const numSamples = samples.length;
  const dataBytes = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeStr = (off: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);                         // PCM subchunk size
  view.setUint16(20, 1, true);                          // Format = PCM
  view.setUint16(22, 1, true);                          // Channels = 1 (mono)
  view.setUint32(24, TARGET_SAMPLE_RATE, true);         // Sample rate = 16000
  view.setUint32(28, TARGET_SAMPLE_RATE * 2, true);     // Byte rate = 32000
  view.setUint16(32, 2, true);                          // Block align = 2
  view.setUint16(34, 16, true);                         // Bits per sample = 16
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1.0, Math.min(1.0, samples[i]));
    view.setInt16(offset, clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767), true);
    offset += 2;
  }
  return buffer;
}

// ---------------------------------------------------------------------------
// Production WebRTCCaptureManager
// ---------------------------------------------------------------------------
class WebRTCCaptureManager {
  private state: CaptureTelemetry = {
    active: false,
    sessionId: null,
    videoActive: false,
    audioActive: false,
    micActive: false,
    tabAudioActive: false,
    reconnecting: false,
    vadSpeaking: false,
    audioLevel: 0,
    chunksSent: 0,
    skippedSilentChunks: 0,
    framesSent: 0,
    droppedFrames: 0,
    error: null,
  };

  private listeners = new Set<Listener>();
  private ws: WebSocket | null = null;
  private wsBaseUrl = "";
  private activeSessionId: string | null = null;
  private runGeneration = 0;

  private displayStream: MediaStream | null = null;
  private micStream: MediaStream | null = null;

  // Web Audio DSP graph handles
  private audioCtx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private resampler: PolyphaseLanczosResampler | null = null;
  private vadChunker: AdaptiveVadChunker | null = null;
  private lastUiAudioEmitMs = 0;

  // Background-safe frame sampling
  private videoEl: HTMLVideoElement | null = null;
  private previewEl: HTMLVideoElement | null = null;
  private frameCanvas: HTMLCanvasElement | null = null;
  private frameCtx: CanvasRenderingContext2D | null = null;
  private tickWorker: Worker | null = null;
  private fallbackTimer: ReturnType<typeof setInterval> | null = null;

  // Resilience & reconnect buffering
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private pendingAudioQueue: ArrayBuffer[] = [];
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

  private emit(patch: Partial<CaptureTelemetry>): void {
    this.state = { ...this.state, ...patch };
    const snap = this.getState();
    this.listeners.forEach((fn) => fn(snap));
  }

  attachPreview(el: HTMLVideoElement | null): void {
    this.previewEl = el;
    if (el && this.displayStream) {
      el.srcObject = this.displayStream;
      el.muted = true;
      el.play().catch(() => { });
    }
  }

  async start(sessionId: string, wsBaseUrl: string): Promise<void> {
    await this.stop();
    const gen = ++this.runGeneration;

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getDisplayMedia) {
      throw new Error("Screen and audio capture requires a modern browser over HTTPS or localhost.");
    }

    // 1. Capture interview tab/screen (video + remote tab audio)
    const displayStream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        frameRate: { ideal: 5, max: 15 },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      } as MediaTrackConstraints,
    });

    if (gen !== this.runGeneration) {
      displayStream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.displayStream = displayStream;

    // 2. ALWAYS attempt local microphone capture in parallel so BOTH Interviewer & Candidate are recorded
    let micStream: MediaStream | null = null;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
    } catch {
      // Proceed gracefully with tab audio only (or video-only) if mic permission is denied
      micStream = null;
    }

    if (gen !== this.runGeneration) {
      micStream?.getTracks().forEach((t) => t.stop());
      displayStream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.micStream = micStream;

    const videoTracks = displayStream.getVideoTracks();
    const tabAudioTracks = displayStream.getAudioTracks();
    const micAudioTracks = micStream?.getAudioTracks() ?? [];
    const hasAudio = tabAudioTracks.length > 0 || micAudioTracks.length > 0;

    this.activeSessionId = sessionId;
    this.wsBaseUrl = wsBaseUrl.replace(/\/+$/, "");
    this.startedAtMs = Date.now();
    this.reconnectAttempts = 0;
    this.pendingAudioQueue = [];

    this.emit({
      active: true,
      sessionId,
      videoActive: videoTracks.length > 0,
      audioActive: hasAudio,
      tabAudioActive: tabAudioTracks.length > 0,
      micActive: micAudioTracks.length > 0,
      reconnecting: false,
      vadSpeaking: false,
      audioLevel: 0,
      chunksSent: 0,
      skippedSilentChunks: 0,
      framesSent: 0,
      droppedFrames: 0,
      error: null,
    });

    if (this.previewEl) {
      this.previewEl.srcObject = displayStream;
      this.previewEl.muted = true;
      this.previewEl.play().catch(() => { });
    }

    videoTracks.forEach((track) => {
      track.addEventListener("ended", () => {
        if (this.runGeneration === gen && this.state.active) {
          void this.stop();
        }
      });
    });

    // 3. Open resilient ingest WebSocket
    try {
      await this.openWebSocket(sessionId, gen);
    } catch (err) {
      await this.stop();
      throw err;
    }

    // 4. Start Web Audio Dual-Source Mixer + Lanczos-3 Resampler + Adaptive VAD WAV Pipeline
    if (hasAudio) {
      await this.startAudioDspPipeline(tabAudioTracks, micAudioTracks, gen);
    }

    // 5. Start Background-Safe 1 FPS Aspect-Preserving JPEG Sampler
    if (videoTracks.length > 0) {
      await this.startVideoSampler(displayStream, gen);
    }
  }

  private async openWebSocket(sessionId: string, gen: number): Promise<void> {
    const url = `${this.wsBaseUrl}/ws/ingest/${sessionId}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        try {
          ws.close();
        } catch {
          // ignore
        }
        reject(new Error("WebSocket ingest connection timed out."));
      }, 8_000);

      ws.onopen = () => {
        clearTimeout(timeout);
        this.reconnectAttempts = 0;
        this.emit({ reconnecting: false, error: null });
        this.flushPendingAudioQueue();
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("Unable to connect to live ingest WebSocket."));
      };
    });

    ws.onclose = () => {
      if (this.runGeneration !== gen || !this.state.active) return;
      this.scheduleReconnect(sessionId, gen);
    };
  }

  private scheduleReconnect(sessionId: string, gen: number): void {
    if (this.reconnectTimer || this.runGeneration !== gen) return;
    if (this.reconnectAttempts >= MAX_WS_RECONNECT_ATTEMPTS) {
      this.emit({
        reconnecting: false,
        error: "Ingest WebSocket lost permanently after retries.",
      });
      return;
    }

    const attempt = ++this.reconnectAttempts;
    const delayMs = Math.min(10_000, 1_000 * 2 ** (attempt - 1) + Math.floor(Math.random() * 350));
    this.emit({
      reconnecting: true,
      error: `Ingest connection dropped. Reconnecting (${attempt}/${MAX_WS_RECONNECT_ATTEMPTS})...`,
    });

    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      if (this.runGeneration !== gen || !this.state.active) return;
      try {
        await this.openWebSocket(sessionId, gen);
      } catch {
        this.scheduleReconnect(sessionId, gen);
      }
    }, delayMs);
  }

  private async startAudioDspPipeline(
    tabAudioTracks: MediaStreamTrack[],
    micAudioTracks: MediaStreamTrack[],
    gen: number,
  ): Promise<void> {
    const AudioCtxCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

    const ctx = new AudioCtxCtor();
    if (ctx.state === "suspended") {
      await ctx.resume().catch(() => { });
    }
    this.audioCtx = ctx;

    // Master summing bus with soft leveling so Speechmatics/Whisper hears both speakers cleanly
    const masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    const masterComp = ctx.createDynamicsCompressor();
    masterComp.threshold.value = -16;
    masterComp.knee.value = 18;
    masterComp.ratio.value = 4;
    masterComp.attack.value = 0.005;
    masterComp.release.value = 0.15;

    masterGain.connect(masterComp);

    const attachSource = (tracks: MediaStreamTrack[], gainValue: number) => {
      if (tracks.length === 0) return;
      const src = ctx.createMediaStreamSource(new MediaStream(tracks));

      // 80 Hz 2nd-order Butterworth high-pass filter removes desk rumble & AC hum
      const hpf = ctx.createBiquadFilter();
      hpf.type = "highpass";
      hpf.frequency.value = 80;
      hpf.Q.value = 0.707;

      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -24;
      comp.knee.value = 24;
      comp.ratio.value = 3.5;
      comp.attack.value = 0.004;
      comp.release.value = 0.18;

      const gain = ctx.createGain();
      gain.gain.value = gainValue;

      src.connect(hpf);
      hpf.connect(comp);
      comp.connect(gain);
      gain.connect(masterGain);
    };

    // Dual-source balanced mixing (slight attenuation when both active avoids sum clipping)
    const dualActive = tabAudioTracks.length > 0 && micAudioTracks.length > 0;
    attachSource(tabAudioTracks, dualActive ? 0.85 : 1.0);
    attachSource(micAudioTracks, dualActive ? 0.80 : 1.0);

    this.resampler = new PolyphaseLanczosResampler(ctx.sampleRate, TARGET_SAMPLE_RATE);
    this.vadChunker = new AdaptiveVadChunker();

    const handlePcmBlock = (pcm: Float32Array) => {
      if (this.runGeneration !== gen || !this.state.active || !this.resampler || !this.vadChunker) {
        return;
      }
      const downsampled = this.resampler.process(pcm);
      if (downsampled.length === 0) return;

      const { wavs, skippedSilent } = this.vadChunker.push(downsampled);
      for (const wav of wavs) {
        this.sendAudioWav(wav);
      }

      const now = Date.now();
      if (skippedSilent > 0 || now - this.lastUiAudioEmitMs >= 150) {
        this.lastUiAudioEmitMs = now;
        this.emit({
          vadSpeaking: this.vadChunker.isSpeaking,
          audioLevel: Number(this.vadChunker.audioLevel.toFixed(3)),
          skippedSilentChunks: this.state.skippedSilentChunks + skippedSilent,
        });
      }
    };

    // Prefer off-main-thread AudioWorklet; fall back to ScriptProcessorNode if restricted
    let workletReady = false;
    if (typeof AudioWorkletNode !== "undefined" && ctx.audioWorklet) {
      let blobUrl: string | null = null;
      try {
        blobUrl = URL.createObjectURL(new Blob([PCM_WORKLET_SOURCE], { type: "application/javascript" }));
        await ctx.audioWorklet.addModule(blobUrl);
        if (this.runGeneration !== gen) return;

        const node = new AudioWorkletNode(ctx, "oculus-pcm-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 0,
          channelCount: 1,
        });
        node.port.onmessage = (ev: MessageEvent<Float32Array>) => {
          if (ev.data) handlePcmBlock(ev.data);
        };
        masterComp.connect(node);
        this.workletNode = node;
        workletReady = true;
      } catch {
        workletReady = false;
      } finally {
        if (blobUrl) URL.revokeObjectURL(blobUrl);
      }
    }

    if (!workletReady && this.runGeneration === gen) {
      const scriptNode = ctx.createScriptProcessor(2048, 1, 1);
      const silentSink = ctx.createGain();
      silentSink.gain.value = 0; // Zero speaker loopback echo
      scriptNode.onaudioprocess = (ev: AudioProcessingEvent) => {
        const ch = ev.inputBuffer.getChannelData(0);
        const copy = new Float32Array(ch.length);
        for (let i = 0; i < ch.length; i++) copy[i] = Math.tanh(ch[i]);
        handlePcmBlock(copy);
      };
      masterComp.connect(scriptNode);
      scriptNode.connect(silentSink);
      silentSink.connect(ctx.destination);
      this.scriptNode = scriptNode;
    }
  }

  private sendAudioWav(wavBuffer: ArrayBuffer): void {
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < MAX_WS_BUFFERED_BYTES * 2) {
      try {
        ws.send(wavBuffer);
        this.emit({ chunksSent: this.state.chunksSent + 1 });
        return;
      } catch {
        // Fall through to bounded retry queue
      }
    }

    if (this.pendingAudioQueue.length >= MAX_PENDING_AUDIO_CHUNKS) {
      this.pendingAudioQueue.shift();
    }
    this.pendingAudioQueue.push(wavBuffer);
  }

  private flushPendingAudioQueue(): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    while (this.pendingAudioQueue.length > 0 && ws.bufferedAmount < MAX_WS_BUFFERED_BYTES * 2) {
      const next = this.pendingAudioQueue.shift();
      if (!next) break;
      try {
        ws.send(next);
        this.emit({ chunksSent: this.state.chunksSent + 1 });
      } catch {
        this.pendingAudioQueue.unshift(next);
        break;
      }
    }
  }

  private async startVideoSampler(displayStream: MediaStream, gen: number): Promise<void> {
    const hiddenVideo = document.createElement("video");
    hiddenVideo.srcObject = displayStream;
    hiddenVideo.muted = true;
    hiddenVideo.playsInline = true;
    this.videoEl = hiddenVideo;
    await hiddenVideo.play().catch(() => { });

    const canvas = document.createElement("canvas");
    canvas.width = MAX_VIDEO_WIDTH;
    canvas.height = MAX_VIDEO_HEIGHT;
    const ctx = canvas.getContext("2d", { willReadFrequently: false });
    this.frameCanvas = canvas;
    this.frameCtx = ctx;

    const sampleFrame = () => {
      if (this.runGeneration !== gen || !this.state.active) return;
      const ws = this.ws;
      if (!ctx || !ws || ws.readyState !== WebSocket.OPEN || hiddenVideo.readyState < 2) return;

      // Drop video frame under socket congestion so audio chunks never stall
      if (ws.bufferedAmount > MAX_WS_BUFFERED_BYTES) {
        this.emit({ droppedFrames: this.state.droppedFrames + 1 });
        return;
      }

      const vw = hiddenVideo.videoWidth || MAX_VIDEO_WIDTH;
      const vh = hiddenVideo.videoHeight || MAX_VIDEO_HEIGHT;
      // Aspect-preserving resize keeps square pixels for solvePnP 3D pose in vision_worker.py
      const scale = Math.min(MAX_VIDEO_WIDTH / vw, MAX_VIDEO_HEIGHT / vh, 1.0);
      const tw = Math.max(16, Math.round(vw * scale));
      const th = Math.max(16, Math.round(vh * scale));
      if (canvas.width !== tw || canvas.height !== th) {
        canvas.width = tw;
        canvas.height = th;
      }

      try {
        ctx.drawImage(hiddenVideo, 0, 0, tw, th);
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
        // Ignore transient decode/canvas frame errors
      }
    };

    // Use Web Worker ticker so 1 FPS capture keeps running when user switches to Google Meet tab
    if (typeof Worker !== "undefined") {
      try {
        const workerUrl = URL.createObjectURL(
          new Blob([TICK_WORKER_SOURCE], { type: "application/javascript" }),
        );
        const worker = new Worker(workerUrl);
        URL.revokeObjectURL(workerUrl);
        worker.onmessage = () => sampleFrame();
        worker.postMessage({ cmd: "start", intervalMs: FRAME_INTERVAL_MS });
        this.tickWorker = worker;
        return;
      } catch {
        // Fall through to standard interval
      }
    }

    this.fallbackTimer = setInterval(sampleFrame, FRAME_INTERVAL_MS);
  }

  async stop(): Promise<void> {
    this.runGeneration++;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.tickWorker) {
      try {
        this.tickWorker.postMessage({ cmd: "stop" });
        this.tickWorker.terminate();
      } catch {
        // ignore
      }
      this.tickWorker = null;
    }
    if (this.fallbackTimer) {
      clearInterval(this.fallbackTimer);
      this.fallbackTimer = null;
    }

    // Flush trailing candidate/interviewer speech before tearing down WebSocket
    if (this.vadChunker) {
      const finalWav = this.vadChunker.flush();
      if (finalWav) {
        this.sendAudioWav(finalWav);
      }
      this.vadChunker = null;
    }
    this.resampler = null;

    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch {
        // ignore
      }
      this.workletNode = null;
    }
    if (this.scriptNode) {
      try {
        this.scriptNode.onaudioprocess = null;
        this.scriptNode.disconnect();
      } catch {
        // ignore
      }
      this.scriptNode = null;
    }
    if (this.audioCtx) {
      try {
        await this.audioCtx.close();
      } catch {
        // ignore
      }
      this.audioCtx = null;
    }

    if (this.videoEl) {
      this.videoEl.srcObject = null;
      this.videoEl = null;
    }
    if (this.previewEl) {
      this.previewEl.srcObject = null;
    }
    this.frameCanvas = null;
    this.frameCtx = null;

    [this.displayStream, this.micStream].forEach((stream) => {
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

    const ws = this.ws;
    this.ws = null;
    this.activeSessionId = null;
    this.pendingAudioQueue = [];

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
      micActive: false,
      tabAudioActive: false,
      reconnecting: false,
      vadSpeaking: false,
      audioLevel: 0,
    });
  }
}

export const webrtcCapture = new WebRTCCaptureManager();