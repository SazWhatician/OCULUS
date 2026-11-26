import asyncio
import io
import logging
import math
import os
import re
import socket
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from app.adapters.stt.base import STTRouter
from app.config import settings
from app.db.postgres import SessionModel, UtteranceModel, async_session_factory
from app.db.redis_bus import redis_bus
from app.db.storage import get_storage_adapter
from app.schemas.contracts import AudioChunkEvent, SpeakerRole, UtteranceEvent
from app.services.stitcher import UtteranceStitcher, _coerce_speaker_role
from app.telemetry import log_structured, metrics

logger = logging.getLogger("oculus.worker.stt")

_active_stt_worker: Optional["STTWorker"] = None

MAX_TRACKED_SESSIONS = 500
DEFAULT_IDLE_FLUSH_SECONDS = 8.0
SWEEPER_TICK_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Precomputed Zero-GPU Acoustic Feature Matrices (16 kHz, 25ms Hamming, 10ms Hop)
# ---------------------------------------------------------------------------
_SR = 16_000
_FRAME_LEN = 400      # 25ms
_HOP_LEN = 160        # 10ms
_N_FFT = 512
_N_BINS = _N_FFT // 2 + 1
_N_MELS = 26
_N_MFCC = 12          # c1..c12 (skips c0 for gain/volume invariance)
_MIN_VOICED_FRAMES = 12
_AC_FFT = 1024        # Zero-padded > 2 * _FRAME_LEN for linear autocorrelation
_F0_MIN_LAG = 45      # ~355 Hz upper vocal bound
_F0_MAX_LAG = 214     # ~75 Hz lower vocal bound


def _build_mel_filterbank() -> np.ndarray:
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_pts = np.linspace(_hz_to_mel(80.0), _hz_to_mel(7600.0), _N_MELS + 2)
    hz_pts = np.array([_mel_to_hz(float(m)) for m in mel_pts], dtype=np.float64)
    bins = np.floor((_N_FFT + 1) * hz_pts / _SR).astype(int)

    fb = np.zeros((_N_MELS, _N_BINS), dtype=np.float32)
    for m in range(1, _N_MELS + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            for k in range(max(0, left), min(_N_BINS, center)):
                fb[m - 1, k] = (k - left) / float(center - left)
        if right > center:
            for k in range(max(0, center), min(_N_BINS, right)):
                fb[m - 1, k] = (right - k) / float(right - center)
    return fb


def _build_dct_matrix() -> np.ndarray:
    n = np.arange(_N_MELS, dtype=np.float32)[None, :]
    k = np.arange(1, _N_MFCC + 1, dtype=np.float32)[:, None]
    return np.sqrt(2.0 / _N_MELS) * np.cos(math.pi * k * (n + 0.5) / _N_MELS)


_HAMMING = np.hamming(_FRAME_LEN).astype(np.float32)
_MEL_FB = _build_mel_filterbank()
_DCT_MAT = _build_dct_matrix()
_FREQ_NORM = np.linspace(0.0, 1.0, _N_BINS, dtype=np.float32)

# Pragmatic discourse cues for resolving Interviewer vs Candidate on non-diarized audio
INTERVIEWER_PROMPT_RE = re.compile(
    r"\b(?:tell\s+me\s+about|walk\s+me\s+through|can\s+you\s+(?:explain|describe|walk|design)|"
    r"how\s+would\s+you|what\s+(?:is|are|would\s+happen)|why\s+did\s+you|"
    r"could\s+you\s+(?:elaborate|explain|compare)|let'?s\s+move\s+on|next\s+question|"
    r"follow[- ]?up\s+question|design\s+a\s+system)\b",
    re.IGNORECASE,
)
CANDIDATE_NARRATIVE_RE = re.compile(
    r"\b(?:in\s+my\s+(?:previous|last|current)\s+(?:role|project|team|company)|"
    r"i\s+(?:built|designed|implemented|deployed|debugged|migrated|optimized|used)|"
    r"we\s+(?:built|designed|implemented|deployed|used|partitioned|cached)|"
    r"my\s+experience|personally\s+i)\b",
    re.IGNORECASE,
)


class AcousticVoiceprintExtractor:
    """
    Zero-GPU, pure-NumPy 40-D speaker embedding extractor (<1.5ms/utterance):
      - 12 MFCC means + 12 MFCC stds (vocal-tract timbre, volume-invariant)
      - 12 Delta-MFCC means (articulatory dynamics)
      - 2 FFT-autocorrelation log2(F0) pitch stats (mean + std)
      - 2 Spectral centroid stats (mean + std)
    Gracefully returns None on non-WAV formats or silence-only slices.
    """

    @staticmethod
    def decode_wav_mono_16k(audio_bytes: bytes) -> Optional[np.ndarray]:
        if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return None
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                rate = wf.getframerate()
                n_frames = wf.getnframes()
                if sampwidth != 2 or channels < 1 or rate <= 0 or n_frames <= 0:
                    return None
                raw = wf.readframes(n_frames)
        except Exception:
            return None

        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            trimmed = (pcm.size // channels) * channels
            pcm = pcm[:trimmed].reshape(-1, channels).mean(axis=1)

        if rate != _SR and pcm.size >= 4:
            target_len = max(1, int(round(pcm.size * (_SR / float(rate)))))
            x_old = np.linspace(0.0, 1.0, pcm.size, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
            pcm = np.interp(x_new, x_old, pcm).astype(np.float32)

        return pcm

    @classmethod
    def extract_batch(
        cls,
        audio_bytes: bytes,
        offset_ms: int,
        utterances: List[UtteranceEvent],
    ) -> List[Optional[List[float]]]:
        pcm = cls.decode_wav_mono_16k(audio_bytes)
        if pcm is None or pcm.size < _FRAME_LEN * _MIN_VOICED_FRAMES:
            return [None] * len(utterances)

        total_ms = int((pcm.size / float(_SR)) * 1000.0)
        embeddings: List[Optional[List[float]]] = []

        for utt in utterances:
            rel_start = max(0, utt.start_ms - offset_ms)
            rel_end = min(total_ms, utt.end_ms - offset_ms)
            if (rel_end - rel_start) < 220:
                if len(utterances) == 1:
                    s_idx, e_idx = 0, pcm.size
                else:
                    embeddings.append(None)
                    continue
            else:
                s_idx = max(0, int((rel_start / 1000.0) * _SR))
                e_idx = min(pcm.size, int((rel_end / 1000.0) * _SR))

            embeddings.append(cls.extract_embedding(pcm[s_idx:e_idx]))
        return embeddings

    @staticmethod
    def extract_embedding(signal: np.ndarray) -> Optional[List[float]]:
        n = signal.size
        if n < _FRAME_LEN + _HOP_LEN * _MIN_VOICED_FRAMES:
            return None

        # Pre-emphasis boosts vocal formants
        emph = np.empty_like(signal)
        emph[0] = signal[0]
        emph[1:] = signal[1:] - 0.97 * signal[:-1]

        num_frames = 1 + (n - _FRAME_LEN) // _HOP_LEN
        if num_frames < _MIN_VOICED_FRAMES:
            return None

        # Zero-copy frame slicing via vectorized index matrix
        idx = np.arange(_FRAME_LEN)[None, :] + (np.arange(num_frames) * _HOP_LEN)[:, None]
        raw_frames = signal[idx]
        emph_frames = emph[idx]

        # Voiced energy gate: keeps active speech frames, drops room silence
        rms = np.sqrt(np.mean(raw_frames * raw_frames, axis=1) + 1e-12)
        energy_thresh = max(0.008, float(np.percentile(rms, 85)) * 0.25)
        voiced_mask = rms >= energy_thresh
        if int(np.count_nonzero(voiced_mask)) < _MIN_VOICED_FRAMES:
            return None

        v_raw = raw_frames[voiced_mask]
        v_emph = emph_frames[voiced_mask] * _HAMMING[None, :]

        # 1. Power spectrum -> Mel log-energies -> 12 volume-invariant MFCCs + Deltas
        spec = np.fft.rfft(v_emph, n=_N_FFT, axis=1)
        power = (spec.real * spec.real + spec.imag * spec.imag) / float(_N_FFT)
        mel_e = np.log(np.maximum(power @ _MEL_FB.T, 1e-8))
        mfcc = mel_e @ _DCT_MAT.T  # (V, 12)

        if mfcc.shape[0] >= 3:
            delta = 0.5 * (mfcc[2:] - mfcc[:-2])
            delta_mean = np.mean(delta, axis=0)
        else:
            delta_mean = np.zeros(_N_MFCC, dtype=np.float32)

        mfcc_mean = np.mean(mfcc, axis=0) / 10.0
        mfcc_std = np.std(mfcc, axis=0) / 5.0
        delta_scaled = delta_mean / 3.0

        # 2. Vectorized FFT autocorrelation pitch (F0 in 75-355 Hz)
        v_centered = v_raw - np.mean(v_raw, axis=1, keepdims=True)
        ac_spec = np.fft.rfft(v_centered, n=_AC_FFT, axis=1)
        acf = np.fft.irfft(ac_spec.real * ac_spec.real + ac_spec.imag * ac_spec.imag, n=_AC_FFT, axis=1)
        acf_norm = acf[:, _F0_MIN_LAG:_F0_MAX_LAG] / (acf[:, 0:1] + 1e-10)
        peak_lags = np.argmax(acf_norm, axis=1)
        peak_vals = acf_norm[np.arange(acf_norm.shape[0]), peak_lags]
        tonal = peak_vals >= 0.35

        if int(np.count_nonzero(tonal)) >= 4:
            f0_hz = _SR / (_F0_MIN_LAG + peak_lags[tonal].astype(np.float32))
            log_f0 = np.log2(f0_hz / 150.0)
            f0_mean = float(np.median(log_f0)) * 1.6
            f0_std = float(np.std(log_f0)) * 1.0
        else:
            f0_mean = 0.0
            f0_std = 0.0

        # 3. Spectral centroid (voice brightness / timbre)
        mag = np.sqrt(power + 1e-12)
        centroid = np.sum(mag * _FREQ_NORM[None, :], axis=1) / (np.sum(mag, axis=1) + 1e-8)
        c_mean = float(np.mean(centroid)) * 2.0
        c_std = float(np.std(centroid)) * 2.0

        vec = np.concatenate([
            mfcc_mean.astype(np.float32),
            mfcc_std.astype(np.float32),
            delta_scaled.astype(np.float32),
            np.array([f0_mean, f0_std, c_mean, c_std], dtype=np.float32),
        ])
        norm = float(np.linalg.norm(vec))
        if not math.isfinite(norm) or norm < 1e-6:
            return None
        return (vec / norm).tolist()


@dataclass
class _SpeakerClusterState:
    centroids: Dict[SpeakerRole, np.ndarray] = field(default_factory=dict)
    sample_counts: Dict[SpeakerRole, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OnlineSpeakerDiarizer:
    """
    Maintains bounded per-session speaker voiceprint centroids to solve:
      1. Speechmatics Batch cross-chunk label permutation (S1/S2 flipping between chunks)
      2. Non-diarized mixed-audio speaker attribution (Groq Whisper $0 fallback)
    """

    def __init__(self, max_sessions: int = MAX_TRACKED_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._states: "OrderedDict[str, _SpeakerClusterState]" = OrderedDict()

    def get_state(self, session_id: str) -> _SpeakerClusterState:
        st = self._states.get(session_id)
        if st is not None:
            self._states.move_to_end(session_id)
            return st
        st = _SpeakerClusterState()
        self._states[session_id] = st
        if len(self._states) > self._max_sessions:
            self._states.popitem(last=False)
        return st

    def clear_session(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    def swap_centroids(self, session_id: str) -> None:
        """Swaps enrolled INTERVIEWER and CANDIDATE voiceprint centroids when UI swaps speakers."""
        st = self._states.get(session_id)
        if st is None:
            return
        int_c = st.centroids.pop(SpeakerRole.INTERVIEWER, None)
        cand_c = st.centroids.pop(SpeakerRole.CANDIDATE, None)
        if int_c is not None:
            st.centroids[SpeakerRole.CANDIDATE] = int_c
        if cand_c is not None:
            st.centroids[SpeakerRole.INTERVIEWER] = cand_c

        int_n = st.sample_counts.pop(SpeakerRole.INTERVIEWER, 0)
        cand_n = st.sample_counts.pop(SpeakerRole.CANDIDATE, 0)
        if int_n:
            st.sample_counts[SpeakerRole.CANDIDATE] = int_n
        if cand_n:
            st.sample_counts[SpeakerRole.INTERVIEWER] = cand_n

    def resolve_chunk_speakers(
        self,
        session_id: str,
        utterances: List[UtteranceEvent],
        embeddings: List[Optional[List[float]]],
        is_mixed_audio: bool,
        speaker_hint: SpeakerRole,
        speaker_role_map: Optional[Mapping[str, Any]],
    ) -> Tuple[List[UtteranceEvent], bool]:
        """
        Returns (role_aligned_utterances, is_diarized_chunk).
        """
        if not utterances:
            return ([], True)

        state = self.get_state(session_id)

        # Case 1: Isolated per-participant track with explicit speaker_hint -> trust & enroll
        if not is_mixed_audio and speaker_hint != SpeakerRole.UNKNOWN:
            label = "S1" if speaker_hint == SpeakerRole.INTERVIEWER else "S2"
            out: List[UtteranceEvent] = []
            for utt, emb in zip(utterances, embeddings):
                updated = utt.model_copy(update={"speaker": speaker_hint, "speaker_label": label})
                out.append(updated)
                if emb is not None and (utt.end_ms - utt.start_ms) >= 900:
                    self._update_centroid(state, speaker_hint, emb)
            return (out, True)

        unique_labels = list(dict.fromkeys(u.speaker_label for u in utterances))
        is_groq_nondiarized = all(u.utterance_id.startswith("groq_") for u in utterances)
        is_diarized = not is_groq_nondiarized

        # Case 2: Multi-speaker diarized chunk (Speechmatics) -> solve 2x2 global permutation
        if is_diarized and len(unique_labels) >= 2:
            return (
                self._align_multi_speaker_diarized(state, utterances, embeddings, speaker_role_map),
                True,
            )

        # Case 3: Single-label chunk (either single-speaker Speechmatics chunk OR non-diarized Groq)
        return (
            self._align_single_label_chunk(
                state=state,
                utterances=utterances,
                embeddings=embeddings,
                speaker_role_map=speaker_role_map,
                is_diarized=is_diarized,
            ),
            is_diarized,
        )

    def _align_multi_speaker_diarized(
        self,
        state: _SpeakerClusterState,
        utterances: List[UtteranceEvent],
        embeddings: List[Optional[List[float]]],
        speaker_role_map: Optional[Mapping[str, Any]],
    ) -> List[UtteranceEvent]:
        # Pool acoustic embedding per local speaker label inside this chunk
        label_vecs: Dict[str, List[np.ndarray]] = {}
        label_texts: Dict[str, List[str]] = {}
        for utt, emb in zip(utterances, embeddings):
            label_texts.setdefault(utt.speaker_label, []).append(utt.text)
            if emb is not None:
                label_vecs.setdefault(utt.speaker_label, []).append(np.asarray(emb, dtype=np.float32))

        pooled: Dict[str, np.ndarray] = {}
        for lbl, vecs in label_vecs.items():
            mean_v = np.mean(np.stack(vecs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_v))
            if norm > 1e-6:
                pooled[lbl] = mean_v / norm

        labels = list(dict.fromkeys(u.speaker_label for u in utterances))
        resolved_map: Dict[str, SpeakerRole] = {}

        # Exact 2x2 Hungarian assignment against enrolled acoustic centroids when available
        int_c = state.centroids.get(SpeakerRole.INTERVIEWER)
        cand_c = state.centroids.get(SpeakerRole.CANDIDATE)
        top_two = [lbl for lbl in labels if lbl in pooled][:2]

        if len(top_two) == 2 and (int_c is not None or cand_c is not None):
            l0, l1 = top_two
            s0_i = float(np.dot(pooled[l0], int_c)) if int_c is not None else 0.0
            s0_c = float(np.dot(pooled[l0], cand_c)) if cand_c is not None else 0.0
            s1_i = float(np.dot(pooled[l1], int_c)) if int_c is not None else 0.0
            s1_c = float(np.dot(pooled[l1], cand_c)) if cand_c is not None else 0.0

            score_direct = s0_i + s1_c
            score_swapped = s0_c + s1_i
            if abs(score_direct - score_swapped) >= 0.08:
                if score_direct >= score_swapped:
                    resolved_map[l0] = SpeakerRole.INTERVIEWER
                    resolved_map[l1] = SpeakerRole.CANDIDATE
                else:
                    resolved_map[l0] = SpeakerRole.CANDIDATE
                    resolved_map[l1] = SpeakerRole.INTERVIEWER

        # Fill any unassigned labels via speaker_role_map or pragmatic question heuristic
        for lbl in labels:
            if lbl in resolved_map:
                continue
            if speaker_role_map and lbl in speaker_role_map:
                resolved_map[lbl] = _coerce_speaker_role(speaker_role_map[lbl], SpeakerRole.CANDIDATE)
            else:
                joined = " ".join(label_texts.get(lbl, []))
                p_score = self._pragmatic_interviewer_score(joined)
                resolved_map[lbl] = SpeakerRole.INTERVIEWER if p_score > 0.25 else SpeakerRole.CANDIDATE

        out: List[UtteranceEvent] = []
        for utt, emb in zip(utterances, embeddings):
            role = resolved_map.get(utt.speaker_label, _coerce_speaker_role(utt.speaker, SpeakerRole.CANDIDATE))
            canon_label = "S1" if role == SpeakerRole.INTERVIEWER else "S2"
            out.append(utt.model_copy(update={"speaker": role, "speaker_label": canon_label}))
            if emb is not None and (utt.end_ms - utt.start_ms) >= 900:
                self._update_centroid(state, role, emb)
        return out

    def _align_single_label_chunk(
        self,
        state: _SpeakerClusterState,
        utterances: List[UtteranceEvent],
        embeddings: List[Optional[List[float]]],
        speaker_role_map: Optional[Mapping[str, Any]],
        is_diarized: bool,
    ) -> List[UtteranceEvent]:
        out: List[UtteranceEvent] = []
        int_c = state.centroids.get(SpeakerRole.INTERVIEWER)
        cand_c = state.centroids.get(SpeakerRole.CANDIDATE)

        for utt, emb in zip(utterances, embeddings):
            p_score = self._pragmatic_interviewer_score(utt.text)
            role: Optional[SpeakerRole] = None

            if emb is not None:
                vec = np.asarray(emb, dtype=np.float32)
                if int_c is not None and cand_c is not None:
                    sim_i = float(np.dot(vec, int_c))
                    sim_c = float(np.dot(vec, cand_c))
                    # Bayesian log-odds fusion: acoustic similarity dominates, pragmatics break ties
                    log_odds = 2.6 * (sim_i - sim_c) + 0.50 * p_score
                    role = SpeakerRole.INTERVIEWER if log_odds >= 0.0 else SpeakerRole.CANDIDATE
                elif int_c is not None:
                    sim_i = float(np.dot(vec, int_c))
                    if sim_i >= 0.72:
                        role = SpeakerRole.INTERVIEWER
                    elif sim_i <= 0.58 and p_score < 0.65:
                        role = SpeakerRole.CANDIDATE
                elif cand_c is not None:
                    sim_c = float(np.dot(vec, cand_c))
                    if sim_c >= 0.72:
                        role = SpeakerRole.CANDIDATE
                    elif sim_c <= 0.58 and p_score > -0.35:
                        role = SpeakerRole.INTERVIEWER

            if role is None:
                if is_diarized and speaker_role_map and utt.speaker_label in speaker_role_map:
                    role = _coerce_speaker_role(speaker_role_map[utt.speaker_label], SpeakerRole.CANDIDATE)
                elif p_score >= 0.45:
                    role = SpeakerRole.INTERVIEWER
                elif p_score <= -0.25:
                    role = SpeakerRole.CANDIDATE
                else:
                    role = _coerce_speaker_role(utt.speaker, SpeakerRole.CANDIDATE)

            canon_label = "S1" if role == SpeakerRole.INTERVIEWER else "S2"
            out.append(utt.model_copy(update={"speaker": role, "speaker_label": canon_label}))

            if emb is not None and (utt.end_ms - utt.start_ms) >= 900:
                self._update_centroid(state, role, emb)
                int_c = state.centroids.get(SpeakerRole.INTERVIEWER)
                cand_c = state.centroids.get(SpeakerRole.CANDIDATE)

        return out

    @staticmethod
    def _update_centroid(state: _SpeakerClusterState, role: SpeakerRole, emb: List[float]) -> None:
        if role not in (SpeakerRole.INTERVIEWER, SpeakerRole.CANDIDATE):
            return
        vec = np.asarray(emb, dtype=np.float32)
        prev = state.centroids.get(role)
        count = state.sample_counts.get(role, 0)
        if prev is None:
            state.centroids[role] = vec
            state.sample_counts[role] = 1
            return

        # Fast adaptation early, stabilizing to 0.12 EMA once enrolled
        alpha = max(0.12, 1.0 / float(count + 1))
        updated = (1.0 - alpha) * prev + alpha * vec
        norm = float(np.linalg.norm(updated))
        if norm > 1e-6:
            state.centroids[role] = updated / norm
            state.sample_counts[role] = count + 1

    @staticmethod
    def _pragmatic_interviewer_score(text: str) -> float:
        """Returns score in [-1.0, +1.0]: positive favors INTERVIEWER, negative favors CANDIDATE."""
        t = (text or "").strip()
        if not t:
            return 0.0
        words = t.split()
        wc = len(words)
        score = 0.0

        if t.endswith("?"):
            score += 0.55 if wc <= 35 else 0.25
        if INTERVIEWER_PROMPT_RE.search(t):
            score += 0.55
        if CANDIDATE_NARRATIVE_RE.search(t):
            score -= 0.70
        if wc >= 45:
            score -= 0.45
        elif wc >= 28 and not t.endswith("?"):
            score -= 0.20

        return max(-1.0, min(1.0, score))


def get_active_stt_worker() -> Optional["STTWorker"]:
    return _active_stt_worker


class STTWorker:
    def __init__(self) -> None:
        self.router = STTRouter()
        self.stitcher = UtteranceStitcher(
            max_gap_ms=int(getattr(settings, "STITCH_GAP_MAX_MS", 2_500)),
            hard_max_gap_ms=int(getattr(settings, "STITCH_HARD_GAP_MAX_MS", 6_500)),
        )
        self.diarizer = OnlineSpeakerDiarizer()
        self.storage = get_storage_adapter()
        self.idle_flush_seconds: float = float(
            getattr(settings, "STT_IDLE_FLUSH_SECONDS", DEFAULT_IDLE_FLUSH_SECONDS)
        )

    async def handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        if not event.storage_key:
            if getattr(event, "is_final", False):
                await self.flush_session(event.session_id)
            return

        try:
            audio_bytes = await self.storage.get_bytes(event.storage_key)
        except Exception as exc:
            logger.error("Failed retrieving audio for key %s: %s", event.storage_key, exc)
            return

        custom_vocab: List[str] = []
        speaker_map: Dict[str, Any] = {}
        async with async_session_factory() as session:
            db_session = await session.get(SessionModel, event.session_id)
            if db_session:
                custom_vocab = list(db_session.custom_vocab or [])
                speaker_map = dict(db_session.speaker_role_map or {})

        # 1. Transcribe via Hybrid Cost Router (Speechmatics / Groq Whisper)
        raw_utterances = await self.router.route_and_transcribe(
            event=event,
            audio_bytes=audio_bytes,
            custom_vocab=custom_vocab,
        )

        # 2. Extract 40-D acoustic voiceprints off the asyncio loop (CPU-only, <1.5ms)
        embeddings: List[Optional[List[float]]] = []
        if raw_utterances:
            embeddings = await asyncio.to_thread(
                AcousticVoiceprintExtractor.extract_batch,
                audio_bytes,
                event.timestamp_ms,
                raw_utterances,
            )

        # 3. Serialize per-session speaker clustering + dynamic stitching
        cluster_state = self.diarizer.get_state(event.session_id)
        async with cluster_state.lock:
            speaker_hint = getattr(event, "speaker_hint", SpeakerRole.UNKNOWN) or SpeakerRole.UNKNOWN
            aligned_utts, is_diarized = self.diarizer.resolve_chunk_speakers(
                session_id=event.session_id,
                utterances=raw_utterances,
                embeddings=embeddings,
                is_mixed_audio=bool(getattr(event, "is_mixed_audio", True)),
                speaker_hint=speaker_hint,
                speaker_role_map=speaker_map,
            )

            # Once acoustic centroids resolve roles canonically (S1=Interviewer, S2=Candidate),
            # only pass DB speaker_role_map if acoustic embeddings were unavailable.
            effective_map = speaker_map if all(e is None for e in embeddings) else None

            for utt, emb in zip(aligned_utts, embeddings):
                flushed_list = self.stitcher.process_incoming(
                    utt,
                    speaker_role_map=effective_map,
                    acoustic_embedding=emb,
                    is_diarized=is_diarized,
                )
                for flushed in flushed_list:
                    await self._persist_and_publish_utterance(flushed)

            if getattr(event, "is_final", False):
                flushed = self.stitcher.flush_session(event.session_id)
                if flushed is not None:
                    await self._persist_and_publish_utterance(flushed)

        metrics.inc("audio_chunks_processed")

    async def swap_session_roles(
        self,
        session_id: str,
        speaker_role_map: Mapping[str, Any],
    ) -> None:
        cluster_state = self.diarizer.get_state(session_id)
        async with cluster_state.lock:
            self.diarizer.swap_centroids(session_id)
            self.stitcher.update_speaker_roles(session_id, speaker_role_map)

    async def flush_session(self, session_id: str) -> Optional[UtteranceEvent]:
        cluster_state = self.diarizer.get_state(session_id)
        async with cluster_state.lock:
            flushed = self.stitcher.flush_session(session_id)
            if flushed is not None:
                await self._persist_and_publish_utterance(flushed)
            return flushed

    async def flush_idle_sessions(self) -> int:
        flushed_count = 0
        now = time.monotonic()
        threshold = max(0.5, float(self.idle_flush_seconds))
        for sid in self.stitcher.active_session_ids():
            cluster_state = self.diarizer.get_state(sid)
            async with cluster_state.lock:
                state = self.stitcher._buffers.get(sid)
                if state is not None and (now - state.last_updated_mono) >= threshold:
                    flushed = self.stitcher.flush_session(sid)
                    if flushed is not None:
                        try:
                            await self._persist_and_publish_utterance(flushed)
                            flushed_count += 1
                        except Exception as exc:
                            logger.error("Idle STT flush failed for %s: %s", sid, exc)
        return flushed_count

    async def flush_all_pending(self) -> None:
        for sid in self.stitcher.active_session_ids():
            try:
                await self.flush_session(sid)
            except Exception as exc:
                logger.error("Shutdown STT flush failed for %s: %s", sid, exc)

    async def _persist_and_publish_utterance(self, utt: UtteranceEvent) -> None:
        words_payload = [w.model_dump(mode="json") for w in utt.words]
        speaker_val = utt.speaker.value if hasattr(utt.speaker, "value") else str(utt.speaker)

        async with async_session_factory() as session:
            existing = await session.get(UtteranceModel, utt.utterance_id)
            if existing is not None:
                existing.speaker = speaker_val
                existing.speaker_label = utt.speaker_label
                existing.text = utt.text
                existing.start_ms = utt.start_ms
                existing.end_ms = utt.end_ms
                existing.words_json = words_payload
            else:
                session.add(
                    UtteranceModel(
                        id=utt.utterance_id,
                        session_id=utt.session_id,
                        speaker=speaker_val,
                        speaker_label=utt.speaker_label,
                        text=utt.text,
                        start_ms=utt.start_ms,
                        end_ms=utt.end_ms,
                        words_json=words_payload,
                    )
                )
            await session.commit()

        await redis_bus.publish_stream("stream:utterances", utt)
        await redis_bus.broadcast_ui(
            session_id=utt.session_id,
            event_type="UTTERANCE",
            data=utt.model_dump(mode="json"),
        )
        metrics.inc("utterances_stitched")

    async def aclose(self) -> None:
        await self.flush_all_pending()
        aclose_fn = getattr(self.router, "aclose", None)
        if callable(aclose_fn):
            await aclose_fn()


async def _idle_sweeper_loop(
    worker: STTWorker,
    stop_event: Optional[asyncio.Event],
) -> None:
    while stop_event is None or not stop_event.is_set():
        try:
            if stop_event is not None:
                await asyncio.wait_for(stop_event.wait(), timeout=SWEEPER_TICK_SECONDS)
                break
            await asyncio.sleep(SWEEPER_TICK_SECONDS)
        except asyncio.TimeoutError:
            pass

        try:
            await worker.flush_idle_sessions()
        except Exception as exc:
            logger.error("STT idle sweeper error: %s", exc, exc_info=True)


async def run_stt_worker(stop_event: Optional[asyncio.Event] = None) -> None:
    global _active_stt_worker
    worker = STTWorker()
    _active_stt_worker = worker
    consumer_id = f"stt_{socket.gethostname()}_{os.getpid()}"
    sweeper_task = asyncio.create_task(_idle_sweeper_loop(worker, stop_event))
    logger.info("STTWorker '%s' started consuming from 'stream:audio'", consumer_id)

    try:
        await redis_bus.consume_stream(
            stream="stream:audio",
            group="stt_workers",
            consumer=consumer_id,
            model_cls=AudioChunkEvent,
            handler=worker.handle_audio_chunk,
            stop_event=stop_event,
        )
    finally:
        sweeper_task.cancel()
        await asyncio.gather(sweeper_task, return_exceptions=True)
        try:
            await worker.aclose()
        finally:
            if _active_stt_worker is worker:
                _active_stt_worker = None
            logger.info("STTWorker '%s' shut down cleanly.", consumer_id)