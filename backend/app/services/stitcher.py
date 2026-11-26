import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.schemas.contracts import SpeakerRole, UtteranceEvent, WordTiming

logger = logging.getLogger("oculus.services.stitcher")

DEFAULT_MAX_GAP_MS = 2_500
DEFAULT_HARD_MAX_GAP_MS = 6_500
DEFAULT_MIN_QUESTION_GAP_MS = 1_500
DEFAULT_MAX_UTTERANCE_DURATION_MS = 120_000
DEFAULT_MAX_SESSIONS = 500
MAX_SEAM_WINDOW_WORDS = 8
MAX_MERGED_IDS_PER_TURN = 64

# Acoustic similarity thresholds for cross-chunk speaker continuity
ACOUSTIC_SAME_SPEAKER_MIN = 0.78
ACOUSTIC_DIFF_SPEAKER_MAX = 0.45

# Temporal guards separating chunk-overlap echoes from genuine live human stutters
SEAM_MULTI_WORD_MAX_GAP_MS = 650
SEAM_SINGLE_WORD_MAX_GAP_MS = 80

TOKEN_EDGE_PUNCT_RE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$", re.IGNORECASE)
LEADING_PUNCT_RE = re.compile(r"^[,.;:!?]+")

# Trailing tokens indicating the speaker paused mid-clause (thinking / hesitating)
INCOMPLETE_TRAILING_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "so", "because", "since", "while",
    "although", "though", "if", "when", "where", "which", "that", "who", "whose",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "into", "about",
    "like", "as", "than", "um", "uh", "uhh", "umm", "er", "erm", "hmm",
    "basically", "actually", "literally", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "will", "would", "could", "should",
    "can", "may", "might", "we", "i", "they", "it",
})

INCOMPLETE_TRAILING_PHRASES = (
    "for example",
    "for instance",
    "such as",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "in order to",
    "so that",
    "meaning that",
)

# Opening tokens indicating the new chunk continues the previous answer
CONTINUATION_LEADING_WORDS = frozenset({
    "and", "or", "but", "so", "because", "which", "that", "also", "then",
    "plus", "however", "therefore", "specifically", "basically", "actually",
    "additionally", "furthermore", "moreover", "second", "secondly", "third",
    "thirdly", "finally", "another", "meaning", "especially",
})

CONTINUATION_LEADING_PHRASES = (
    "for example",
    "for instance",
    "in addition",
    "on top of",
    "as a result",
    "that means",
    "this allows",
    "this ensures",
    "which means",
    "to handle",
    "to avoid",
)


def _coerce_speaker_role(
    raw_role: Any,
    fallback: SpeakerRole = SpeakerRole.UNKNOWN,
) -> SpeakerRole:
    """Safely coerces SpeakerRole enums or raw DB/JSON strings into SpeakerRole."""
    if isinstance(raw_role, SpeakerRole):
        return raw_role
    if isinstance(raw_role, str):
        normalized = raw_role.strip().lower()
        if not normalized:
            return fallback
        try:
            return SpeakerRole(normalized)
        except ValueError:
            try:
                return SpeakerRole[normalized.upper()]
            except KeyError:
                return fallback
    return fallback


def _cosine_similarity(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    denom = math.sqrt(na * nb)
    if denom < 1e-8:
        return None
    return max(-1.0, min(1.0, dot / denom))


def _blend_embeddings(
    base: Optional[List[float]],
    incoming: Optional[Sequence[float]],
    weight_in: float = 0.35,
) -> Optional[List[float]]:
    if not incoming:
        return base
    in_list = [float(x) for x in incoming]
    if not base or len(base) != len(in_list):
        norm = math.sqrt(sum(v * v for v in in_list))
        return [v / norm for v in in_list] if norm > 1e-8 else None
    blended = [(1.0 - weight_in) * b + weight_in * n for b, n in zip(base, in_list)]
    norm = math.sqrt(sum(v * v for v in blended))
    return [v / norm for v in blended] if norm > 1e-8 else base


def _norm_token(token: str) -> str:
    return TOKEN_EDGE_PUNCT_RE.sub("", (token or "").strip().lower())


def _tokenize_raw(text: str) -> List[str]:
    return [tok for tok in (text or "").strip().split() if tok]


def _normalize_words(words: Sequence[WordTiming]) -> List[WordTiming]:
    if not words:
        return []
    cleaned = [
        w.model_copy(update={"end_ms": max(w.end_ms, w.start_ms)})
        if w.end_ms < w.start_ms
        else w
        for w in words
        if (w.word or "").strip()
    ]
    cleaned.sort(key=lambda w: (w.start_ms, w.end_ms))
    return cleaned


def _mean_confidence(words: Sequence[WordTiming]) -> float:
    if not words:
        return 0.5
    return sum(float(getattr(w, "confidence", 1.0) or 1.0) for w in words) / len(words)


def _find_seam_overlap_k(
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
    gap_ms: int,
) -> int:
    """
    Longest suffix-prefix alignment over normalized tokens at a chunk boundary.
    Handles negative timestamp overlap (e.g., 350ms WebRTC pre-roll) while protecting
    authentic single-word human stutters when `gap_ms >= 80ms`.
    """
    if not left_tokens or not right_tokens or gap_ms > SEAM_MULTI_WORD_MAX_GAP_MS:
        return 0

    max_k = min(len(left_tokens), len(right_tokens), MAX_SEAM_WINDOW_WORDS)
    left_norm = [_norm_token(t) for t in left_tokens[-max_k:]]
    right_norm = [_norm_token(t) for t in right_tokens[:max_k]]

    for k in range(max_k, 0, -1):
        if k == 1 and gap_ms >= SEAM_SINGLE_WORD_MAX_GAP_MS:
            continue
        l_slice = left_norm[-k:]
        r_slice = right_norm[:k]
        if l_slice == r_slice and any(l_slice):
            return k
    return 0


def _join_clean_text(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> str:
    if not left_tokens:
        return " ".join(right_tokens).strip()
    if not right_tokens:
        return " ".join(left_tokens).strip()
    first_right = right_tokens[0]
    if LEADING_PUNCT_RE.fullmatch(first_right):
        left_merged = [*left_tokens[:-1], f"{left_tokens[-1]}{first_right}"]
        return " ".join([*left_merged, *right_tokens[1:]]).strip()
    return " ".join([*left_tokens, *right_tokens]).strip()


def _merge_chronological(
    left: UtteranceEvent,
    right: UtteranceEvent,
    resolved_role: SpeakerRole,
) -> UtteranceEvent:
    left_tokens = _tokenize_raw(left.text)
    right_tokens = _tokenize_raw(right.text)
    left_words = _normalize_words(left.words)
    right_words = _normalize_words(right.words)

    gap_ms = right.start_ms - left.end_ms
    k = _find_seam_overlap_k(left_tokens, right_tokens, gap_ms)

    if k > 0:
        left_seam_conf = _mean_confidence(left_words[-k:]) if len(left_words) >= k else 0.5
        right_seam_conf = _mean_confidence(right_words[:k]) if len(right_words) >= k else 0.5

        if right_seam_conf > left_seam_conf + 0.03:
            kept_left_tokens = left_tokens[:-k]
            kept_right_tokens = right_tokens
            kept_left_words = left_words[:-k] if len(left_words) >= k else left_words
            kept_right_words = right_words
        else:
            kept_left_tokens = left_tokens
            kept_right_tokens = right_tokens[k:]
            kept_left_words = left_words
            kept_right_words = right_words[k:] if len(right_words) >= k else right_words
    else:
        kept_left_tokens = left_tokens
        kept_right_tokens = right_tokens
        kept_left_words = left_words
        kept_right_words = right_words

    # Remove overlapping duplicate boundary words across pre-roll seams
    if kept_left_words and kept_right_words:
        seam_cutoff_ms = kept_left_words[-1].end_ms
        last_left_norm = _norm_token(kept_left_words[-1].word)
        pruned_right: List[WordTiming] = []
        for w in kept_right_words:
            if w.end_ms <= seam_cutoff_ms and _norm_token(w.word) == last_left_norm:
                continue
            pruned_right.append(w)
        kept_right_words = pruned_right

    merged_words = _normalize_words([*kept_left_words, *kept_right_words])
    merged_text = _join_clean_text(kept_left_tokens, kept_right_tokens)

    start_ms = min(left.start_ms, right.start_ms)
    end_ms = max(left.end_ms, right.end_ms)
    if merged_words:
        start_ms = min(start_ms, merged_words[0].start_ms)
        end_ms = max(end_ms, merged_words[-1].end_ms)

    return UtteranceEvent(
        session_id=left.session_id,
        utterance_id=left.utterance_id,
        speaker=resolved_role,
        speaker_label=left.speaker_label,
        text=merged_text,
        start_ms=start_ms,
        end_ms=max(end_ms, start_ms),
        words=merged_words,
    )


def _is_subset_duplicate(pending: UtteranceEvent, incoming: UtteranceEvent) -> bool:
    if incoming.start_ms < (pending.start_ms - 200) or incoming.end_ms > (pending.end_ms + 200):
        return False
    p_set = {_norm_token(t) for t in _tokenize_raw(pending.text) if _norm_token(t)}
    i_set = {_norm_token(t) for t in _tokenize_raw(incoming.text) if _norm_token(t)}
    if not i_set:
        return True
    return (len(p_set & i_set) / len(i_set)) >= 0.85


def _assess_continuation_extension_ms(
    pending_text: str,
    incoming_text: str,
    pending_role: SpeakerRole,
    acoustic_sim: Optional[float],
    adaptive_pause_ema_ms: float,
    base_max_gap_ms: int,
    hard_max_gap_ms: int,
) -> int:
    """
    Computes the syntactic-prosodic-acoustic dynamic merge threshold `tau_merge` (ms).
    Prevents 3-5s mid-answer thinking pauses from splitting continuous technical answers
    while keeping clean terminal sentences and interviewer questions tightly bounded.
    """
    p_strip = pending_text.strip()
    i_strip = incoming_text.strip()
    if not p_strip:
        return base_max_gap_ms

    p_lower = p_strip.lower()
    i_lower = i_strip.lower()
    p_tokens = _tokenize_raw(p_strip)
    i_tokens = _tokenize_raw(i_strip)

    # 1. Interviewer question yield: shrink merge window once a clear question finishes
    if p_strip.endswith("?") and pending_role == SpeakerRole.INTERVIEWER:
        if i_tokens and _norm_token(i_tokens[0]) not in CONTINUATION_LEADING_WORDS:
            return min(base_max_gap_ms, DEFAULT_MIN_QUESTION_GAP_MS)

    bonus_ms = 0

    # 2. Trailing syntactic incompleteness on pending text
    ends_terminal = p_strip[-1] in ".?!"
    ends_comma_or_dash = p_strip.endswith((",", ";", ":", "-", "—", "..."))
    last_norm = _norm_token(p_tokens[-1]) if p_tokens else ""

    if ends_comma_or_dash:
        bonus_ms += 2_000
    elif not ends_terminal:
        bonus_ms += 1_500

    if last_norm in INCOMPLETE_TRAILING_WORDS or any(p_lower.endswith(ph) for ph in INCOMPLETE_TRAILING_PHRASES):
        bonus_ms += 1_500

    # 3. Explicit discourse continuation at the start of incoming text
    first_raw = i_tokens[0] if i_tokens else ""
    first_norm = _norm_token(first_raw)
    starts_lowercase = bool(first_raw) and first_raw[0].islower()
    starts_connector = first_norm in CONTINUATION_LEADING_WORDS or any(
        i_lower.startswith(ph) for ph in CONTINUATION_LEADING_PHRASES
    )
    if starts_lowercase:
        bonus_ms += 1_200
    if starts_connector:
        bonus_ms += 1_400

    # 4. Candidate acoustic voiceprint verification (real audio continuity across thinking pauses)
    if acoustic_sim is not None and not p_strip.endswith("?"):
        if acoustic_sim >= ACOUSTIC_SAME_SPEAKER_MIN:
            # Scales smoothly from +1100ms at 0.78 up to +2000ms at 0.95
            sim_scale = min(1.0, (acoustic_sim - ACOUSTIC_SAME_SPEAKER_MIN) / 0.17)
            bonus_ms += int(1_100 + 900 * sim_scale)
        elif acoustic_sim <= 0.58:
            # Ambiguous acoustic match: tighten threshold conservatively
            bonus_ms = min(bonus_ms, 500)

    # 5. Slow-speaker / ESL adaptive pause compensation (only when other signals support continuation)
    if bonus_ms > 0 and adaptive_pause_ema_ms > 900.0:
        bonus_ms += int(min(1_000.0, (adaptive_pause_ema_ms - 900.0) * 0.6))

    effective = base_max_gap_ms + bonus_ms
    return max(DEFAULT_MIN_QUESTION_GAP_MS, min(hard_max_gap_ms, effective))


@dataclass
class _OpenUtteranceState:
    utterance: UtteranceEvent
    last_updated_mono: float = field(default_factory=time.monotonic)
    merged_ids: set[str] = field(default_factory=set)
    acoustic_embedding: Optional[List[float]] = None
    intra_pause_ema_ms: float = 600.0

    def track_id(self, utterance_id: str) -> None:
        if not utterance_id:
            return
        if len(self.merged_ids) >= MAX_MERGED_IDS_PER_TURN:
            self.merged_ids.clear()
        self.merged_ids.add(utterance_id)

    def update_pause_ema(self, gap_ms: int) -> None:
        if 150 <= gap_ms <= 5_000:
            self.intra_pause_ema_ms = 0.25 * float(gap_ms) + 0.75 * self.intra_pause_ema_ms


class UtteranceStitcher:
    """
    Syntactic-Prosodic-Acoustic Utterance Stitcher:
      - Safe SpeakerRole enum coercion (fixes DB string AttributeError crashes)
      - Dynamic continuation-aware merge horizon (handles 3-5s mid-answer candidate thinking pauses)
      - Cross-chunk acoustic speaker consistency (splits mis-merged non-diarized chunks & confirms same voice)
      - Confidence-weighted suffix-prefix seam deduplication preserving genuine human stutters
      - Bounded LRU memory, 120s monologue cap, and idle/shutdown flushing (zero dropped final answers)
    """

    def __init__(
        self,
        max_gap_ms: int = DEFAULT_MAX_GAP_MS,
        hard_max_gap_ms: int = DEFAULT_HARD_MAX_GAP_MS,
        max_utterance_duration_ms: int = DEFAULT_MAX_UTTERANCE_DURATION_MS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self.max_gap_ms = max(500, int(max_gap_ms))
        self.hard_max_gap_ms = max(self.max_gap_ms, int(hard_max_gap_ms))
        self.max_utterance_duration_ms = max(10_000, int(max_utterance_duration_ms))
        self.max_sessions = max(1, int(max_sessions))
        self._buffers: "OrderedDict[str, _OpenUtteranceState]" = OrderedDict()

    def active_session_ids(self) -> List[str]:
        return list(self._buffers.keys())

    def get_open_utterance(self, session_id: str) -> Optional[UtteranceEvent]:
        state = self._buffers.get(session_id)
        return state.utterance if state is not None else None

    def update_speaker_roles(
        self,
        session_id: str,
        speaker_role_map: Mapping[str, Any],
    ) -> None:
        state = self._buffers.get(session_id)
        if state is None or not speaker_role_map:
            return
        utt = state.utterance
        if utt.speaker_label in speaker_role_map:
            new_role = _coerce_speaker_role(speaker_role_map[utt.speaker_label], fallback=utt.speaker)
            if new_role != utt.speaker:
                state.utterance = utt.model_copy(update={"speaker": new_role})

    def process_incoming(
        self,
        incoming: UtteranceEvent,
        speaker_role_map: Optional[Mapping[str, Any]] = None,
        acoustic_embedding: Optional[Sequence[float]] = None,
        is_diarized: bool = True,
    ) -> List[UtteranceEvent]:
        """
        Processes one STT utterance chunk and returns 0..N finalized utterances.
        Backward-compatible with 1- and 2-argument callers; accepts optional 40-D
        `acoustic_embedding` and `is_diarized` flag from STTWorker for voice continuity.
        """
        clean_text = (incoming.text or "").strip()
        if not clean_text:
            return []

        resolved_in_role = self._resolve_role(incoming, speaker_role_map, is_diarized=is_diarized)
        norm_in = incoming.model_copy(
            update={
                "speaker": resolved_in_role,
                "text": clean_text,
                "end_ms": max(incoming.end_ms, incoming.start_ms),
                "words": _normalize_words(incoming.words),
            }
        )

        flushed: List[UtteranceEvent] = []
        state = self._buffers.get(norm_in.session_id)

        if state is None:
            self._put_state(norm_in.session_id, norm_in, acoustic_embedding)
            return flushed

        self._buffers.move_to_end(norm_in.session_id)
        now_mono = time.monotonic()

        if norm_in.utterance_id and norm_in.utterance_id in state.merged_ids:
            return flushed

        pending = state.utterance
        resolved_pending_role = self._resolve_role(pending, speaker_role_map, is_diarized=is_diarized)
        if resolved_pending_role != pending.speaker:
            pending = pending.model_copy(update={"speaker": resolved_pending_role})
            state.utterance = pending

        if _is_subset_duplicate(pending, norm_in):
            state.track_id(norm_in.utterance_id)
            state.last_updated_mono = now_mono
            return flushed

        acoustic_sim = _cosine_similarity(state.acoustic_embedding, acoustic_embedding)
        is_same_speaker = self._is_same_speaker(
            pending=pending,
            incoming=norm_in,
            speaker_role_map=speaker_role_map,
            acoustic_sim=acoustic_sim,
            is_diarized=is_diarized,
        )

        # Out-of-order delayed chunk strictly preceding open buffer
        if norm_in.end_ms <= pending.start_ms:
            reverse_gap_ms = pending.start_ms - norm_in.end_ms
            combined_dur = pending.end_ms - norm_in.start_ms
            if (
                is_same_speaker
                and reverse_gap_ms <= self.max_gap_ms
                and combined_dur <= self.max_utterance_duration_ms
            ):
                state.utterance = _merge_chronological(norm_in, pending, resolved_pending_role)
                state.acoustic_embedding = _blend_embeddings(state.acoustic_embedding, acoustic_embedding)
                state.track_id(norm_in.utterance_id)
                state.last_updated_mono = now_mono
            else:
                flushed.append(norm_in)
            return flushed

        gap_ms = norm_in.start_ms - pending.end_ms
        projected_dur = max(pending.end_ms, norm_in.end_ms) - min(pending.start_ms, norm_in.start_ms)

        effective_max_gap = _assess_continuation_extension_ms(
            pending_text=pending.text,
            incoming_text=norm_in.text,
            pending_role=resolved_pending_role,
            acoustic_sim=acoustic_sim,
            adaptive_pause_ema_ms=state.intra_pause_ema_ms,
            base_max_gap_ms=self.max_gap_ms,
            hard_max_gap_ms=self.hard_max_gap_ms,
        )

        if (
            is_same_speaker
            and gap_ms <= effective_max_gap
            and projected_dur <= self.max_utterance_duration_ms
        ):
            state.utterance = _merge_chronological(pending, norm_in, resolved_pending_role)
            state.acoustic_embedding = _blend_embeddings(state.acoustic_embedding, acoustic_embedding)
            state.update_pause_ema(gap_ms)
            state.track_id(norm_in.utterance_id)
            state.last_updated_mono = now_mono
        else:
            flushed.append(pending)
            self._put_state(norm_in.session_id, norm_in, acoustic_embedding)

        return flushed

    def flush_session(self, session_id: str) -> Optional[UtteranceEvent]:
        state = self._buffers.pop(session_id, None)
        return state.utterance if state is not None else None

    def flush_idle(self, max_idle_seconds: float = 8.0) -> List[UtteranceEvent]:
        threshold = max(0.5, float(max_idle_seconds))
        now = time.monotonic()
        flushed: List[UtteranceEvent] = []

        for session_id in list(self._buffers.keys()):
            state = self._buffers.get(session_id)
            if state is not None and (now - state.last_updated_mono) >= threshold:
                self._buffers.pop(session_id, None)
                flushed.append(state.utterance)
        return flushed

    def flush_all(self) -> List[UtteranceEvent]:
        flushed = [state.utterance for state in self._buffers.values()]
        self._buffers.clear()
        return flushed

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------
    def _put_state(
        self,
        session_id: str,
        utt: UtteranceEvent,
        acoustic_embedding: Optional[Sequence[float]],
    ) -> None:
        state = _OpenUtteranceState(
            utterance=utt,
            acoustic_embedding=_blend_embeddings(None, acoustic_embedding),
        )
        state.track_id(utt.utterance_id)
        self._buffers[session_id] = state
        self._buffers.move_to_end(session_id)

        if len(self._buffers) > self.max_sessions:
            evicted_id, _ = self._buffers.popitem(last=False)
            logger.warning("Evicted oldest stitcher buffer '%s' (LRU cap=%d).", evicted_id, self.max_sessions)

    @staticmethod
    def _resolve_role(
        utt: UtteranceEvent,
        speaker_role_map: Optional[Mapping[str, Any]],
        is_diarized: bool,
    ) -> SpeakerRole:
        base_role = _coerce_speaker_role(utt.speaker, fallback=SpeakerRole.UNKNOWN)
        # Apply DB speaker_role_map when explicitly provided; if non-diarized worker already resolved
        # acoustic role, preserve it unless base_role is UNKNOWN.
        if speaker_role_map and utt.speaker_label in speaker_role_map:
            if is_diarized or base_role == SpeakerRole.UNKNOWN:
                return _coerce_speaker_role(speaker_role_map[utt.speaker_label], fallback=base_role)
        return base_role

    @staticmethod
    def _is_same_speaker(
        pending: UtteranceEvent,
        incoming: UtteranceEvent,
        speaker_role_map: Optional[Mapping[str, Any]],
        acoustic_sim: Optional[float],
        is_diarized: bool,
    ) -> bool:
        both_known = (
            pending.speaker != SpeakerRole.UNKNOWN
            and incoming.speaker != SpeakerRole.UNKNOWN
        )
        if both_known and pending.speaker != incoming.speaker:
            return False

        # On non-diarized chunks (e.g., Groq Whisper mixed audio), acoustic divergence overrides
        # identical placeholder labels ("S2" == "S2") so interviewer/candidate turns never fuse.
        if not is_diarized and acoustic_sim is not None and acoustic_sim <= ACOUSTIC_DIFF_SPEAKER_MAX:
            return False

        if pending.speaker_label == incoming.speaker_label:
            return True

        # Cross-chunk label permutation recovery: strong acoustic match bridges flipped labels
        if acoustic_sim is not None and acoustic_sim >= 0.84 and (not both_known or pending.speaker == incoming.speaker):
            return True

        return bool(speaker_role_map) and both_known and pending.speaker == incoming.speaker