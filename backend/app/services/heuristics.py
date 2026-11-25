import math
import re
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    VisionAnomalyEvent,
    VisionAnomalyType,
)

# ---------------------------------------------------------------------------
# Forensic Linguistic & Psycholinguistic Regex Banks (<1ms, Zero ML Deps)
# ---------------------------------------------------------------------------

VOCAL_HESITATION_RE = re.compile(
    r"\b(um+|uh+|er+m?|hm+m?|ah+)\b",
    re.IGNORECASE,
)

DISCOURSE_FILLER_RE = re.compile(
    r"(?:\b(you know|i mean|sort of|kind of|basically|literally)\b"
    r"|\b(?:was|were|is|are|it'?s|that'?s)\s+like\b"
    r"|,\s*like\b|\blike,\s*)",
    re.IGNORECASE,
)

STUTTER_RE = re.compile(r"\b([a-z]{2,})\s+\1\b", re.IGNORECASE)
NON_STUTTER_DOUBLES = frozenset({"had", "that", "very"})

SELF_REPAIR_RE = re.compile(
    r"\b(wait no|sorry i meant|let me rephrase|actually no|or rather|scratch that|"
    r"i should say|well actually|i meant to say|hold on|no wait)\b",
    re.IGNORECASE,
)

CONTRACTION_RE = re.compile(
    r"\b(?:i'm|i've|i'd|i'll|we're|we've|we'd|we'll|you're|you've|they're|they've|"
    r"it's|that's|there's|here's|what's|don't|doesn't|didn't|won't|wouldn't|"
    r"can't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)

# Exculpatory live-engineering markers: concrete war stories LLMs rarely volunteer unprompted
ENGINEERING_SCAR_RE = re.compile(
    r"\b(?:debugg(?:ed|ing)|rolled\s+back|rollback|hotfix|post[- ]?mortem|incident|"
    r"outage|broke\s+prod|production\s+bug|memory\s+leak|oom(?:killed)?|deadlock|"
    r"race\s+condition|flaky|throttl(?:ed|ing)|profil(?:ed|ing)|benchmark(?:ed|ing)|"
    r"migrated|re[- ]?indexed|backfill(?:ed|ing)?|p9[59]|50[0234]|429)\b",
    re.IGNORECASE,
)
CONCRETE_METRIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|mins?|hours?|%|kb|mb|gb|tb|rps|qps|k|m|x)\b",
    re.IGNORECASE,
)

# Tier A: High-precision LLM prose artifacts rarely spoken spontaneously
TIER_A_AI_TELLS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\bdelve\s+into\b", re.I), "delve into", 12.0),
    (re.compile(r"\bmultifaceted\b", re.I), "multifaceted", 12.0),
    (re.compile(r"\bfoster(?:s|ing)?\s+(?:a\s+)?(?:sense\s+of\s+)?synergy\b", re.I), "foster synergy", 12.0),
    (re.compile(r"\bplay(?:s|ing)?\s+a\s+pivotal\s+role\b", re.I), "plays a pivotal role", 12.0),
    (re.compile(r"\bever[- ]evolving\s+(?:landscape|ecosystem|field)\b", re.I), "ever-evolving landscape", 12.0),
    (re.compile(r"\bseamlessly\s+integrat(?:e|es|ed|ing)\b", re.I), "seamlessly integrate", 10.0),
    (re.compile(r"\bholistic\s+(?:approach|view|perspective|strategy)\b", re.I), "holistic approach", 10.0),
    (re.compile(r"\bharness(?:ing)?\s+the\s+power\s+of\b", re.I), "harness the power of", 12.0),
    (re.compile(r"\bcomprehensive\s+(?:approach|understanding|framework|overview)\b", re.I), "comprehensive approach", 10.0),
    (re.compile(r"\btestament\s+to\b", re.I), "testament to", 10.0),
    (re.compile(r"\btapestry\s+of\b", re.I), "tapestry of", 12.0),
    (re.compile(r"\bunderscores?\s+the\s+importance\s+of\b", re.I), "underscores the importance", 11.0),
)

# Tier B: Formulaic essay signposts (mild individually; diagnostic when clustered)
TIER_B_AI_TELLS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\bit'?s\s+(?:crucial|important|essential)\s+to\s+note\b", re.I), "it's crucial to note", 6.0),
    (re.compile(r"\bfirstly\b.*\bsecondly\b", re.I | re.S), "firstly...secondly", 7.0),
    (re.compile(r"\bin\s+summary\b", re.I), "in summary", 5.0),
    (re.compile(r"\bin\s+conclusion\b", re.I), "in conclusion", 6.0),
    (re.compile(r"\bkey\s+takeaways?\b", re.I), "key takeaways", 5.0),
    (re.compile(r"\b(?:furthermore|moreover|consequently|additionally)\b", re.I), "formal essay connector", 4.5),
    (re.compile(r"\bthat'?s\s+a\s+(?:great|excellent|fantastic|really\s+good)\s+question\b", re.I), "formulaic stall opener", 4.0),
    (re.compile(r"\blet\s+me\s+break\s+(?:this|that|it)\s+down\b", re.I), "break this down opener", 4.0),
)

# Verbalized Markdown list / triad templates (catches stealth ChatGPT output without clichés)
VERBAL_ENUMERATION_RE = re.compile(
    r"(?:\b(?:first(?:ly)?|number\s+one)\b.{10,240}\b(?:second(?:ly)?|number\s+two)\b"
    r"(?:.{10,240}\b(?:third(?:ly)?|number\s+three|finally)\b)?)",
    re.IGNORECASE | re.DOTALL,
)
ESSAY_FRAMING_RE = re.compile(
    r"\b(?:when\s+it\s+comes\s+to|in\s+terms\s+of|on\s+the\s+other\s+hand|"
    r"by\s+leveraging|ensuring\s+(?:that|optimal|high)|thereby\s+[a-z]+ing)\b",
    re.IGNORECASE,
)

FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
WORD_TOKEN_RE = re.compile(r"[a-z0-9']+", re.I)
ALPHA_WORD_RE = re.compile(r"[a-z]+", re.I)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

# STT-Resilient Security Scanner: catches spoken numbers ("zero") and word-level overrides
INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+(?:of\s+)?)?(?:your\s+)?(?:previous|prior|above|system)\s+instructions?\b", re.I),
    re.compile(
        r"\b(?:disregard|override|bypass|forget)\s+(?:all\s+)?(?:your\s+)?(?:system|grading|scoring|proctoring)\s+(?:prompt|rules?|instructions?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:set|output|return|give(?:\s+me)?)\s+(?:the\s+|a\s+)?(?:suspicion|risk)[_\s]?score\s+(?:to|of|as)\s+(?:0|zero|low)\b",
        re.I,
    ),
    re.compile(r"\bmark\s+(?:this\s+)?candidate\s+as\s+(?:honest|authentic|clean|low\s+risk)\b", re.I),
    re.compile(r"\bsystem\s+override\b.*\b(?:score|authentic|honest|zero|0)\b", re.I),
)

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "would", "should",
    "will", "just", "you", "your", "me", "my", "we", "our", "how", "what", "why",
    "which", "who", "tell", "explain", "describe", "walk", "think", "of", "that", "this",
    "it", "they", "them", "their", "so", "not", "no", "yes", "as", "very", "also",
})

# Population priors for Empirical Bayes shrinkage when Q0 sample is short
PRIOR_WPM = 135.0
PRIOR_MAD_WPM = 18.0
PRIOR_DISFLUENCY_RATE = 3.0
PRIOR_TTFS_MS = 1200.0
SHRINKAGE_FULL_WORDS = 60.0

VISION_BASE_WEIGHTS: Dict[str, float] = {
    "DEVICE_DETECTED": 24.0,
    "PHONE_DETECTED": 24.0,
    "MULTI_FACE": 18.0,
    "LIVENESS_SPOOF": 20.0,
    "CAMERA_OCCLUSION": 12.0,
    "HEAD_PITCH_DOWN": 10.0,
    "GAZE_DEVIATION": 7.5,
}

CATEGORY_FALLBACK_MAP: Dict[str, str] = {
    "phone / device detected": "DEVICE_DETECTED",
    "multi-face detection": "MULTI_FACE",
    "liveness / spoof risk": "LIVENESS_SPOOF",
    "frozen video feed": "LIVENESS_SPOOF",
    "camera occlusion": "CAMERA_OCCLUSION",
    "face absent from frame": "CAMERA_OCCLUSION",
    "gaze deviation (looking away)": "GAZE_DEVIATION",
    "teleprompter reading pattern": "GAZE_DEVIATION",
    "head pose shift": "HEAD_PITCH_DOWN",
}

HIGH_SEVERITY_VISION_TYPES = frozenset({
    "DEVICE_DETECTED",
    "PHONE_DETECTED",
    "MULTI_FACE",
    "LIVENESS_SPOOF",
})


# ---------------------------------------------------------------------------
# Bounded Per-Session Stylometric Fingerprint Cache (Zero Schema Impact)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StylometricProfile:
    word_count: int
    mean_word_len: float
    long_word_ratio: float      # Share of alpha words >= 8 chars
    lexical_density: float      # Share of non-stopword content tokens
    contraction_rate: float     # Contractions per 100 words


_MAX_STYLE_PROFILES = 500
_STYLE_PROFILES: "OrderedDict[str, _StylometricProfile]" = OrderedDict()


def _extract_stylometric_profile(text: str) -> _StylometricProfile:
    alpha_words = ALPHA_WORD_RE.findall((text or "").lower())
    n = len(alpha_words)
    if n == 0:
        return _StylometricProfile(0, 4.4, 0.12, 0.48, 2.0)

    mean_len = sum(len(w) for w in alpha_words) / n
    long_ratio = sum(1 for w in alpha_words if len(w) >= 8) / n
    content_ratio = sum(1 for w in alpha_words if len(w) >= 3 and w not in STOPWORDS) / n
    contractions = len(CONTRACTION_RE.findall(text))
    contraction_rate = (contractions / n) * 100.0
    return _StylometricProfile(
        word_count=n,
        mean_word_len=mean_len,
        long_word_ratio=long_ratio,
        lexical_density=content_ratio,
        contraction_rate=contraction_rate,
    )


def _store_style_profile(session_id: str, profile: _StylometricProfile) -> None:
    if not session_id:
        return
    _STYLE_PROFILES[session_id] = profile
    _STYLE_PROFILES.move_to_end(session_id)
    if len(_STYLE_PROFILES) > _MAX_STYLE_PROFILES:
        _STYLE_PROFILES.popitem(last=False)


def _get_style_profile(session_id: Optional[str]) -> Optional[_StylometricProfile]:
    if not session_id:
        return None
    prof = _STYLE_PROFILES.get(session_id)
    if prof is not None:
        _STYLE_PROFILES.move_to_end(session_id)
    return prof


def clear_session_style_profile(session_id: str) -> None:
    _STYLE_PROFILES.pop(session_id, None)


# ---------------------------------------------------------------------------
# Pure Signal Extractors
# ---------------------------------------------------------------------------

def _extract_word_timing(item: Any) -> Optional[Tuple[int, int]]:
    if isinstance(item, dict):
        start = item.get("start_ms")
        end = item.get("end_ms")
    else:
        start = getattr(item, "start_ms", None)
        end = getattr(item, "end_ms", None)
    if start is None or end is None:
        return None
    try:
        s_ms, e_ms = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if s_ms < 0 or e_ms < s_ms:
        return None
    return (s_ms, e_ms)


def _count_natural_disfluencies(text: str) -> Tuple[int, int]:
    """Returns (total_disfluency_count, self_repair_and_stutter_count)."""
    vocal = len(VOCAL_HESITATION_RE.findall(text))
    discourse = len(DISCOURSE_FILLER_RE.findall(text))
    stutters = sum(
        1 for m in STUTTER_RE.finditer(text) if m.group(1).lower() not in NON_STUTTER_DOUBLES
    )
    repairs = len(SELF_REPAIR_RE.findall(text))
    return (vocal + discourse + stutters + repairs, stutters + repairs)


def _linear_ramp(value: float, low: float, high: float, max_points: float) -> float:
    if value <= low or high <= low:
        return 0.0
    if value >= high:
        return max_points
    return max_points * ((value - low) / (high - low))


def _length_confidence(word_count: int, min_words: int, full_words: int, min_scale: float = 0.35) -> float:
    if word_count < min_words:
        return 0.0
    if word_count >= full_words or full_words <= min_words:
        return 1.0
    return min_scale + (1.0 - min_scale) * ((word_count - min_words) / (full_words - min_words))


def _compute_real_wpm_windows(
    timings: Sequence[Tuple[int, int]],
    window_ms: int = 10_000,
    step_ms: int = 5_000,
) -> List[float]:
    if len(timings) < 6:
        return []

    sorted_t = sorted(timings, key=lambda x: x[0])
    t_start = sorted_t[0][0]
    t_end = sorted_t[-1][1]
    total_span = t_end - t_start
    if total_span < 6_000:
        return []

    midpoints = [(s + e) * 0.5 for s, e in sorted_t]
    effective_window = min(window_ms, max(total_span // 3, 4_000))
    effective_step = max(effective_window // 2, 2_000)

    wpms: List[float] = []
    cursor = float(t_start)
    while cursor + effective_window <= t_end + 1:
        win_end = cursor + effective_window
        count = sum(1 for m in midpoints if cursor <= m < win_end)
        if count >= 3:
            wpms.append(count / (effective_window / 60_000.0))
        cursor += effective_step
    return wpms


def _compute_robust_cadence_cv(inter_word_gaps: Optional[Sequence[int]]) -> Tuple[float, int, float]:
    """
    Computes trimmed pause Coefficient of Variation (teleprompter uniformity)
    and early-vs-late pause acceleration ratio ("stall-then-firehose" streaming signature).
    Returns (cv, valid_pause_count, stall_to_flow_ratio).
    """
    if not inter_word_gaps:
        return (0.75, 0, 1.0)

    ordered_valid = [int(g) for g in inter_word_gaps if 60 <= int(g) <= 2_500]
    n = len(ordered_valid)
    if n < 5:
        return (0.75, n, 1.0)

    sorted_pauses = sorted(ordered_valid)
    trim = max(1, n // 10) if n >= 10 else 0
    core = sorted_pauses[trim : n - trim] if trim else sorted_pauses
    mean_gap = statistics.fmean(core)
    cv = (statistics.pstdev(core) / mean_gap) if mean_gap > 0.0 else 0.75

    # Compare opening 25% pauses against body 75% pauses (detects waiting for LLM token stream)
    stall_to_flow = 1.0
    if n >= 8:
        split = max(2, n // 4)
        early_mean = statistics.fmean(ordered_valid[:split])
        body_mean = statistics.fmean(ordered_valid[split:])
        if body_mean > 0.0:
            stall_to_flow = early_mean / body_mean

    return (cv, n, stall_to_flow)


def _compute_sentence_uniformity(text: str) -> Tuple[Optional[float], int]:
    """
    Detects written-essay rhythmic uniformity across punctuated sentences.
    Returns (sentence_len_cv, sentence_count). Safely returns (None, count)
    when STT output lacks punctuation so raw transcripts are never penalized.
    """
    raw_sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    lengths = [len(WORD_TOKEN_RE.findall(s)) for s in raw_sents]
    substantive = [l for l in lengths if l >= 6]
    if len(substantive) < 4:
        return (None, len(substantive))

    mean_len = statistics.fmean(substantive)
    if mean_len < 11.0:
        return (None, len(substantive))
    return (statistics.pstdev(substantive) / mean_len, len(substantive))


def _compute_stylometric_shift(
    curr_profile: _StylometricProfile,
    base_profile: Optional[_StylometricProfile],
    word_count: int,
) -> Tuple[float, Optional[str]]:
    """
    Detects stealth register jumps (casual Q0 speaker suddenly reciting dense academic LLM prose).
    Relative to the candidate's own Q0 baseline so naturally formal/academic speakers are never penalized.
    """
    if base_profile is None or base_profile.word_count < 30 or word_count < 40:
        return (0.0, None)

    len_delta = curr_profile.mean_word_len - base_profile.mean_word_len
    long_word_delta = curr_profile.long_word_ratio - base_profile.long_word_ratio
    density_delta = curr_profile.lexical_density - base_profile.lexical_density
    contraction_collapse = (
        base_profile.contraction_rate >= 2.0 and curr_profile.contraction_rate <= 0.2
    )

    score = 0.0
    score += _linear_ramp(len_delta, low=0.55, high=1.35, max_points=6.0)
    score += _linear_ramp(long_word_delta, low=0.07, high=0.18, max_points=6.0)
    score += _linear_ramp(density_delta, low=0.08, high=0.20, max_points=4.0)
    if contraction_collapse and (len_delta >= 0.45 or long_word_delta >= 0.06):
        score += 3.5

    scaled = min(score, 15.0) * _length_confidence(word_count, min_words=40, full_words=75, min_scale=0.50)
    if scaled < 5.0:
        return (0.0, None)

    return (
        scaled,
        f"Sharp academic register shift vs Q0 (avg word len {len_delta:+.2f}, long-word ratio {long_word_delta:+.0%})",
    )


def _compute_exculpatory_credit(
    text: str,
    word_count: int,
    self_repair_count: int,
    vision_risk: float,
    injection_detected: bool,
) -> float:
    """
    Computes bounded authenticity credit for hard-to-fake live human signals:
    spontaneous cognitive self-repairs and grounded engineering war-story specifics.
    Disabled when hard vision proof or prompt injection is present.
    """
    if injection_detected or vision_risk >= 15.0 or word_count < 25:
        return 0.0

    credit = 0.0
    if self_repair_count >= 2:
        credit += 8.0
    elif self_repair_count == 1 and word_count >= 40:
        credit += 3.0

    scar_hits = len(ENGINEERING_SCAR_RE.findall(text))
    metric_hits = len(CONCRETE_METRIC_RE.findall(text))
    fp_hits = len(FIRST_PERSON_RE.findall(text))

    # Concrete metrics + incident ownership ("I debugged... p99 dropped to 38ms")
    if fp_hits >= 2 and (scar_hits >= 1 or metric_hits >= 1):
        credit += min(8.0, (scar_hits * 3.0) + (metric_hits * 2.5))

    return min(credit, 14.0)


def _compute_question_mirroring_ratio(question_text: Optional[str], answer_tokens: Sequence[str]) -> float:
    if not question_text or len(answer_tokens) < 20:
        return 0.0

    q_words = {
        w.lower()
        for w in WORD_TOKEN_RE.findall(question_text)
        if len(w) >= 4 and w.lower() not in STOPWORDS
    }
    if len(q_words) < 4:
        return 0.0

    opening_words = {w.lower() for w in answer_tokens[:25]}
    return len(q_words & opening_words) / len(q_words)


def _score_vision_events(events: Optional[Sequence[VisionAnomalyEvent]]) -> Tuple[float, int, List[str]]:
    if not events:
        return (0.0, 0, [])

    total_risk = 0.0
    notes: List[str] = []
    high_severity_types: set[str] = set()

    for ev in events:
        raw_type = getattr(ev, "anomaly_type", None)
        type_name = getattr(raw_type, "value", str(raw_type)).strip().upper() if raw_type is not None else ""
        if not type_name:
            cat_key = str(getattr(ev, "category", "") or "").strip().lower()
            type_name = CATEGORY_FALLBACK_MAP.get(cat_key, "GAZE_DEVIATION")
        base_weight = VISION_BASE_WEIGHTS.get(type_name, 8.0)

        conf = max(0.4, min(1.0, float(getattr(ev, "confidence", 0.8) or 0.8)))
        duration_ms = int(getattr(ev, "duration_ms", 3_000) or 3_000)

        if type_name in ("GAZE_DEVIATION", "HEAD_PITCH_DOWN", "CAMERA_OCCLUSION"):
            dur_scale = min(1.5, max(0.35, duration_ms / 4_000.0))
        else:
            dur_scale = 1.0

        if type_name in HIGH_SEVERITY_VISION_TYPES:
            high_severity_types.add(type_name)

        total_risk += base_weight * conf * dur_scale

    capped_risk = min(total_risk, 30.0)
    if high_severity_types:
        notes.append(
            f"High-severity concurrent vision telemetry ({', '.join(sorted(high_severity_types))})"
        )
    elif len(events) > 0:
        notes.append(f"{len(events)} concurrent sustained gaze/posture flags")

    return (capped_risk, len(events), notes)


# ---------------------------------------------------------------------------
# Public Stage 0 & Stage 1 APIs
# ---------------------------------------------------------------------------

def compute_candidate_baseline(
    session_id: str,
    q0_text: str,
    duration_ms: int,
    words: Optional[List[Dict[str, Any]]] = None,
    ttfs_ms: float = 1200.0,
) -> CandidateBaseline:
    """
    Stage 0: Calibrate candidate baseline on Question 0 ('Tell me about yourself')
    using time-windowed median/MAD, Empirical Bayes shrinkage, and cached stylometric
    fingerprinting to eliminate ESL/neurodivergent bias and catch stealth register jumps.
    """
    raw_tokens = WORD_TOKEN_RE.findall(q0_text or "")
    word_count = len(raw_tokens)
    dur_min = max(duration_ms / 60_000.0, 0.05)
    overall_wpm = word_count / dur_min

    # Store private Q0 stylometric fingerprint for downstream stealth register-jump detection
    _store_style_profile(session_id, _extract_stylometric_profile(q0_text or ""))

    disfluency_count, _ = _count_natural_disfluencies(q0_text or "")
    raw_disfluency_rate = (disfluency_count / max(word_count, 1)) * 100.0

    timings: List[Tuple[int, int]] = []
    if words:
        for item in words:
            parsed = _extract_word_timing(item)
            if parsed is not None:
                timings.append(parsed)

    pause_ratio = 0.10
    wpm_subwindows: List[float] = []

    if len(timings) >= 6:
        sorted_t = sorted(timings, key=lambda x: x[0])
        gaps = [
            sorted_t[i][0] - sorted_t[i - 1][1]
            for i in range(1, len(sorted_t))
            if sorted_t[i][0] > sorted_t[i - 1][1]
        ]
        if gaps:
            long_pauses_ms = sum(g for g in gaps if 600 <= g <= 5_000)
            pause_ratio = min(long_pauses_ms / max(duration_ms, 1), 0.95)
        wpm_subwindows = _compute_real_wpm_windows(sorted_t)

    if len(wpm_subwindows) >= 3:
        raw_median_wpm = float(statistics.median(wpm_subwindows))
        raw_mad_wpm = float(statistics.median([abs(x - raw_median_wpm) for x in wpm_subwindows]))
        calibrated_mad = max(12.0, min(raw_mad_wpm, 45.0))
    else:
        raw_median_wpm = overall_wpm
        calibrated_mad = PRIOR_MAD_WPM

    alpha = min(word_count / SHRINKAGE_FULL_WORDS, 1.0)
    shrunk_wpm = (alpha * raw_median_wpm) + ((1.0 - alpha) * PRIOR_WPM)
    shrunk_disfluency = (alpha * raw_disfluency_rate) + ((1.0 - alpha) * PRIOR_DISFLUENCY_RATE)
    clamped_ttfs = max(200.0, min(float(ttfs_ms or PRIOR_TTFS_MS), 6_000.0))
    shrunk_ttfs = (alpha * clamped_ttfs) + ((1.0 - alpha) * PRIOR_TTFS_MS)

    return CandidateBaseline(
        session_id=session_id,
        baseline_wpm=round(shrunk_wpm, 1),
        mad_wpm=round(calibrated_mad, 1),
        baseline_disfluency_rate=round(shrunk_disfluency, 2),
        baseline_pause_ratio=round(pause_ratio, 3),
        baseline_ttfs_ms=round(shrunk_ttfs, 1),
        sample_word_count=word_count,
        baseline_gaze_variance=0.0,
    )


def compute_deterministic_heuristics(
    answer_text: str,
    duration_ms: int,
    baseline: Optional[CandidateBaseline] = None,
    inter_word_gaps: Optional[List[int]] = None,
    ttfs_latency_ms: int = 0,
    concurrent_vision_events: Optional[List[VisionAnomalyEvent]] = None,
    question_text: Optional[str] = None,
) -> HeuristicScores:
    """
    Stage 1: Sub-2ms multimodal forensic engine combining baseline-relative prosody,
    stealth stylometric register-shift detection, verbalized-markdown spotting,
    exculpatory engineering-scar credits, and orthogonal cross-modal synergy fusion.
    """
    safe_text = (answer_text or "").strip()
    tokens = WORD_TOKEN_RE.findall(safe_text)
    word_count = len(tokens)
    dur_min = max(duration_ms / 60_000.0, 0.05)
    curr_wpm = word_count / dur_min

    # 1. Robust WPM shift against calibrated baseline
    base_wpm = baseline.baseline_wpm if baseline else PRIOR_WPM
    mad = baseline.mad_wpm if (baseline and baseline.mad_wpm > 0) else PRIOR_MAD_WPM
    wpm_robust_zscore = 0.6745 * (curr_wpm - base_wpm) / max(mad, 12.0)

    # 2. Context-aware disfluency rate & baseline-relative collapse
    disfluency_count, self_repair_count = _count_natural_disfluencies(safe_text)
    curr_disfluency_rate = (disfluency_count / max(word_count, 1)) * 100.0
    base_disfluency = baseline.baseline_disfluency_rate if baseline else PRIOR_DISFLUENCY_RATE
    disfluency_drop = max(base_disfluency - curr_disfluency_rate, 0.0)

    # 3. Trimmed Pause Cadence Uniformity CV + Stall-to-Flow Streaming Ratio
    cadence_uniformity_cv, valid_pause_count, stall_to_flow = _compute_robust_cadence_cv(inter_word_gaps)

    # 4. Tiered Spoken LLM Tell Bank + Verbalized Markdown + Stylometric Register Shift
    ai_tell_matches: List[str] = []
    tier_a_hits = 0
    tier_b_hits = 0
    tier_a_risk = 0.0
    tier_b_risk = 0.0

    for pattern, name, weight in TIER_A_AI_TELLS:
        if pattern.search(safe_text):
            ai_tell_matches.append(name)
            tier_a_hits += 1
            tier_a_risk += weight

    for pattern, name, weight in TIER_B_AI_TELLS:
        if pattern.search(safe_text):
            ai_tell_matches.append(name)
            tier_b_hits += 1
            tier_b_risk += weight

    curr_style = _extract_stylometric_profile(safe_text)
    base_style = _get_style_profile(baseline.session_id if baseline else None)

    contraction_count = len(CONTRACTION_RE.findall(safe_text))
    contraction_deficit = word_count >= 45 and contraction_count == 0 and (tier_a_hits + tier_b_hits) >= 2
    tell_risk = tier_a_risk + min(tier_b_risk, 20.0) + (6.0 if contraction_deficit else 0.0)

    if tier_a_hits >= 3 or len(ai_tell_matches) >= 5:
        tell_cap = 48.0
    elif tier_a_hits >= 2 or len(ai_tell_matches) >= 3:
        tell_cap = 38.0
    else:
        tell_cap = 30.0
    tell_risk = min(tell_risk, tell_cap)

    # Stealth-AI structural signals (catch cliché-suppressed ChatGPT/Claude recitations)
    style_shift_pts, style_shift_reason = _compute_stylometric_shift(curr_style, base_style, word_count)
    has_verbal_enum = word_count >= 45 and bool(VERBAL_ENUMERATION_RE.search(safe_text))
    essay_frame_hits = len(ESSAY_FRAMING_RE.findall(safe_text)) if word_count >= 45 else 0
    sent_cv, sent_count = _compute_sentence_uniformity(safe_text)
    sent_uniform_pts = (
        _linear_ramp(0.28 - sent_cv, low=0.0, high=0.18, max_points=7.0)
        if (sent_cv is not None and sent_cv < 0.28)
        else 0.0
    )

    # 5. First-person narrative ownership density & opening question-mirroring stall
    first_person_matches = FIRST_PERSON_RE.findall(safe_text)
    first_person_ratio = len(first_person_matches) / max(word_count, 1)
    mirroring_ratio = _compute_question_mirroring_ratio(question_text, tokens)

    # 6. Baseline-relative TTFS (AI generation latency stall before polished speech)
    base_ttfs = baseline.baseline_ttfs_ms if baseline else PRIOR_TTFS_MS
    ttfs_excess_ms = max(0.0, float(ttfs_latency_ms) - max(base_ttfs, 1_000.0))

    # 7. Severity-weighted concurrent vision telemetry
    vision_risk, concurrent_vision_flags, vision_notes = _score_vision_events(concurrent_vision_events)

    # 8. STT-resilient spoken prompt-injection security scan
    injection_detected = any(p.search(safe_text) for p in INJECTION_PATTERNS)

    # -----------------------------------------------------------------------
    # Channel-Partitioned Forensic Scoring + Cross-Modal Synergy Fusion
    # -----------------------------------------------------------------------
    linguistic_risk = 0.0
    prosodic_risk = 0.0
    trigger_reasons: List[str] = []

    if tell_risk > 0.0:
        linguistic_risk += tell_risk
        if tier_a_hits > 0:
            trigger_reasons.append(f"Matched {len(ai_tell_matches)} LLM prose tells ({tier_a_hits} high-precision)")
        else:
            trigger_reasons.append(f"Matched {len(ai_tell_matches)} structured essay transitions")
        if contraction_deficit:
            trigger_reasons.append("Zero spoken contractions across 45+ word structured monologue")

    if style_shift_pts > 0.0 and style_shift_reason:
        linguistic_risk += style_shift_pts
        trigger_reasons.append(style_shift_reason)

    structural_pts = 0.0
    if has_verbal_enum and "firstly...secondly" not in ai_tell_matches:
        structural_pts += 6.0
    if essay_frame_hits >= 2:
        structural_pts += min(6.0, essay_frame_hits * 2.5)
    if sent_uniform_pts >= 2.5:
        structural_pts += sent_uniform_pts
    if structural_pts >= 5.0:
        linguistic_risk += min(structural_pts, 12.0)
        trigger_reasons.append(
            f"Written-essay syntactic uniformity (enum={has_verbal_enum}, sents={sent_count})"
        )

    if word_count >= 40 and first_person_ratio < 0.015:
        ownership_pts = (
            _linear_ramp(0.015 - first_person_ratio, low=0.0, high=0.015, max_points=7.0)
            * _length_confidence(word_count, min_words=40, full_words=80, min_scale=0.45)
        )
        if ownership_pts >= 2.0:
            linguistic_risk += ownership_pts
            trigger_reasons.append("Low personal ownership / first-person narrative markers")

    if mirroring_ratio >= 0.60 and word_count >= 35:
        linguistic_risk += 6.0
        trigger_reasons.append(f"Opening question mirroring ({int(mirroring_ratio * 100)}% keyword echo)")

    # Prosodic & Acoustic Channel
    if word_count >= 30 and disfluency_drop > 1.5:
        disfluency_pts = (
            _linear_ramp(disfluency_drop, low=1.5, high=4.5, max_points=22.0)
            * _length_confidence(word_count, min_words=30, full_words=60, min_scale=0.40)
        )
        if self_repair_count >= 2:
            disfluency_pts *= 0.5
        if disfluency_pts >= 4.0:
            prosodic_risk += disfluency_pts
            trigger_reasons.append(
                f"Disfluency collapse vs Q0 baseline (-{round(disfluency_drop, 1)}/100w)"
            )

    wpm_pts = _linear_ramp(abs(wpm_robust_zscore), low=1.8, high=3.5, max_points=14.0)
    if wpm_pts >= 4.0 and word_count >= 40:
        prosodic_risk += wpm_pts
        trigger_reasons.append(f"Significant WPM shift vs baseline (Z={round(wpm_robust_zscore, 1)})")

    if word_count >= 35 and valid_pause_count >= 5 and cadence_uniformity_cv < 0.50:
        cadence_pts = (
            _linear_ramp(0.50 - cadence_uniformity_cv, low=0.0, high=0.25, max_points=14.0)
            * _length_confidence(word_count, min_words=35, full_words=60, min_scale=0.45)
        )
        if cadence_pts >= 4.0:
            prosodic_risk += cadence_pts
            trigger_reasons.append(
                f"Uniform pause cadence (CV={round(cadence_uniformity_cv, 2)}, n={valid_pause_count})"
            )

    if word_count >= 45 and valid_pause_count >= 8 and stall_to_flow >= 2.2 and curr_disfluency_rate < 1.2:
        prosodic_risk += 6.0
        trigger_reasons.append(
            f"Stall-then-firehose delivery pattern (opening/body pause ratio={stall_to_flow:.1f}x)"
        )

    if word_count >= 45 and ttfs_excess_ms >= 1_400.0 and curr_disfluency_rate < 0.8:
        stall_pts = (
            _linear_ramp(ttfs_excess_ms, low=1_400.0, high=3_500.0, max_points=12.0)
            * _length_confidence(word_count, min_words=45, full_words=60, min_scale=0.65)
        )
        if stall_pts >= 3.0:
            prosodic_risk += stall_pts
            trigger_reasons.append(
                f"Latency stall (+{int(ttfs_excess_ms)}ms vs baseline) before polished delivery"
            )

    if vision_risk > 0.0:
        trigger_reasons.extend(vision_notes)

    # Combine channels + orthogonal cross-modal corroboration boost
    security_risk = 30.0 if injection_detected else 0.0
    if injection_detected:
        trigger_reasons.insert(0, "Security alert: spoken evaluator-manipulation phrasing detected")

    active_channels = sum((
        linguistic_risk >= 12.0,
        prosodic_risk >= 12.0,
        vision_risk >= 10.0,
    ))
    synergy_boost = 0.0
    if active_channels == 3:
        synergy_boost = 12.0
        trigger_reasons.append("Tri-modal corroboration (linguistic + prosodic + vision anomalies aligned)")
    elif active_channels == 2 and (linguistic_risk + prosodic_risk + vision_risk) >= 32.0:
        synergy_boost = 6.0
        trigger_reasons.append("Cross-modal corroboration across 2 independent forensic channels")

    # Apply bounded exculpatory discount for genuine live-engineering scars & self-repairs
    exculpatory_credit = _compute_exculpatory_credit(
        text=safe_text,
        word_count=word_count,
        self_repair_count=self_repair_count,
        vision_risk=vision_risk,
        injection_detected=injection_detected,
    )

    raw_risk = linguistic_risk + prosodic_risk + vision_risk + security_risk + synergy_boost - exculpatory_credit
    deterministic_risk = min(max(raw_risk, 0.0), 100.0)

    # Two-Tier Cost-Saving Stage 2 LLM Judge Gate
    trigger_llm_judge = False
    if injection_detected:
        trigger_llm_judge = True
    elif word_count >= 15 and (
        vision_risk >= 15.0
        or (tier_a_hits >= 1 and len(ai_tell_matches) >= 2)
        or len(ai_tell_matches) >= 3
    ):
        trigger_llm_judge = True
    elif word_count >= 35 and (
        deterministic_risk >= 40.0
        or active_channels >= 2
        or (tier_a_hits >= 1 and disfluency_drop >= 2.0)
        or (style_shift_pts >= 8.0 and disfluency_drop >= 2.0)
        or (disfluency_drop >= (2.5 if word_count >= 60 else 3.5) and curr_disfluency_rate < 1.0)
        or (ttfs_excess_ms >= 1_800.0 and curr_disfluency_rate < 0.5)
    ):
        trigger_llm_judge = True

    return HeuristicScores(
        word_count=word_count,
        wpm=round(curr_wpm, 1),
        wpm_robust_zscore=round(wpm_robust_zscore, 2),
        disfluency_rate=round(curr_disfluency_rate, 2),
        disfluency_drop=round(disfluency_drop, 2),
        ttfs_latency_ms=ttfs_latency_ms,
        cadence_uniformity_cv=round(cadence_uniformity_cv, 2),
        ai_tell_matches=ai_tell_matches,
        first_person_ratio=round(first_person_ratio, 3),
        concurrent_vision_flags=concurrent_vision_flags,
        deterministic_risk=round(deterministic_risk, 1),
        trigger_llm_judge=trigger_llm_judge,
        trigger_reasons=trigger_reasons,
    )