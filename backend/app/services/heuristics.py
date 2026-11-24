import math
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    VisionAnomalyEvent,
    VisionAnomalyType,
)

# ---------------------------------------------------------------------------
# Forensic Linguistic Regex Banks (Compiled Once at Import, <1ms per Turn)
# ---------------------------------------------------------------------------

# Pure vocal hesitations (always true spoken disfluencies)
VOCAL_HESITATION_RE = re.compile(
    r"\b(um+|uh+|er+m?|hm+m?|ah+)\b",
    re.IGNORECASE,
)

# Conversational discourse fillers (excludes grammatical "like" such as "like Kafka" or "would like")
DISCOURSE_FILLER_RE = re.compile(
    r"(?:\b(you know|i mean|sort of|kind of|basically|literally)\b"
    r"|\b(?:was|were|is|are|it'?s|that'?s)\s+like\b"
    r"|,\s*like\b|\blike,\s*)",
    re.IGNORECASE,
)

# Immediate word stutters ("the the", "we we") — strong indicator of live human speech
STUTTER_RE = re.compile(r"\b([a-z]{2,})\s+\1\b", re.IGNORECASE)
NON_STUTTER_DOUBLES = frozenset({"had", "that", "very"})

# Cognitive self-repairs — LLMs almost never generate spontaneous spoken restarts
SELF_REPAIR_RE = re.compile(
    r"\b(wait no|sorry i meant|let me rephrase|actually no|or rather|scratch that|i should say|well actually)\b",
    re.IGNORECASE,
)

# Contractions signal natural spoken English vs recited written LLM output
CONTRACTION_RE = re.compile(
    r"\b(?:i'm|i've|i'd|i'll|we're|we've|we'd|we'll|you're|you've|they're|they've|"
    r"it's|that's|there's|here's|what's|don't|doesn't|didn't|won't|wouldn't|"
    r"can't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)

# Tier A: High-precision LLM prose artifacts rarely spoken spontaneously in interviews
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
)

# Tier B: Formulaic essay signposts (mild individually; meaningful when clustered)
TIER_B_AI_TELLS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\bit'?s\s+(?:crucial|important|essential)\s+to\s+note\b", re.I), "it's crucial to note", 6.0),
    (re.compile(r"\bfirstly\b.*\bsecondly\b", re.I | re.S), "firstly...secondly", 7.0),
    (re.compile(r"\bin\s+summary\b", re.I), "in summary", 5.0),
    (re.compile(r"\bin\s+conclusion\b", re.I), "in conclusion", 6.0),
    (re.compile(r"\bkey\s+takeaways?\b", re.I), "key takeaways", 5.0),
    (re.compile(r"\b(?:furthermore|moreover|consequently|additionally)\b", re.I), "formal essay connector", 4.5),
    (re.compile(r"\bthat'?s\s+a\s+(?:great|excellent|fantastic|really\s+good)\s+question\b", re.I), "formulaic stall opener", 4.0),
)

FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
WORD_TOKEN_RE = re.compile(r"[a-z0-9']+", re.I)

# Stage 1 Security Scanner: catches spoken attempts to hijack the downstream LLM judge
INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?\b", re.I),
    re.compile(r"\b(disregard|override|bypass)\s+(your\s+)?(system|grading|scoring|proctoring)\s+(prompt|rules?|instructions?)\b", re.I),
    re.compile(r"\b(set|output|return)\s+(the\s+)?suspicion[_\s]?score\s+(to\s+)?0\b", re.I),
    re.compile(r"\bmark\s+(this\s+)?candidate\s+as\s+(honest|authentic|low\s+risk)\b", re.I),
)

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "would", "should",
    "will", "just", "you", "your", "me", "my", "we", "our", "how", "what", "why",
    "which", "who", "tell", "explain", "describe", "walk", "think", "of", "that", "this",
})

# Population priors used for Empirical Bayes shrinkage when Q0 sample is short
PRIOR_WPM = 135.0
PRIOR_MAD_WPM = 18.0
PRIOR_DISFLUENCY_RATE = 3.0
PRIOR_TTFS_MS = 1200.0
SHRINKAGE_FULL_WORDS = 60.0

# Vision anomaly severity weights (protects neurodivergent gaze variance while escalating hard proof)
VISION_BASE_WEIGHTS: Dict[str, float] = {
    "DEVICE_DETECTED": 24.0,
    "PHONE_DETECTED": 24.0,
    "MULTI_FACE": 18.0,
    "HEAD_PITCH_DOWN": 10.0,
    "GAZE_DEVIATION": 7.5,
}


def _extract_word_timing(item: Any) -> Optional[Tuple[int, int]]:
    """Supports both dict items and Pydantic WordTiming objects cleanly."""
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
    """
    Returns (total_disfluency_count, self_repair_and_stutter_count).
    Separating grammatical 'like' from true fillers prevents technical-speech bias.
    """
    vocal = len(VOCAL_HESITATION_RE.findall(text))
    discourse = len(DISCOURSE_FILLER_RE.findall(text))
    stutters = sum(
        1 for m in STUTTER_RE.finditer(text) if m.group(1).lower() not in NON_STUTTER_DOUBLES
    )
    repairs = len(SELF_REPAIR_RE.findall(text))
    return (vocal + discourse + stutters + repairs, stutters + repairs)


def _linear_ramp(value: float, low: float, high: float, max_points: float) -> float:
    """Smooth saturating ramp in [0.0, max_points] eliminating step-function cliffs."""
    if value <= low or high <= low:
        return 0.0
    if value >= high:
        return max_points
    return max_points * ((value - low) / (high - low))


def _compute_real_wpm_windows(timings: Sequence[Tuple[int, int]], window_ms: int = 10_000, step_ms: int = 5_000) -> List[float]:
    """
    Computes true time-windowed WPM from word-level timestamps using a sliding window.
    Fixes the algebraic bug where equal token slicing always returned constant overall_wpm.
    """
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


def _compute_robust_cadence_cv(inter_word_gaps: Optional[Sequence[int]]) -> Tuple[float, int]:
    """
    Computes trimmed Coefficient of Variation over meaningful spoken pauses (60ms..2500ms).
    Prevents 0ms phoneme boundaries and 4s cough/interruption outliers from corrupting CV.
    Returns (cv, valid_pause_count).
    """
    if not inter_word_gaps:
        return (0.75, 0)

    valid_pauses = sorted(g for g in inter_word_gaps if 60 <= g <= 2_500)
    n = len(valid_pauses)
    if n < 5:
        return (0.75, n)

    # 10% two-sided trim when enough pauses exist so one long breath cannot hide uniform pacing
    trim = max(1, n // 10) if n >= 10 else 0
    core = valid_pauses[trim : n - trim] if trim else valid_pauses
    mean_gap = statistics.fmean(core)
    if mean_gap <= 0.0:
        return (0.75, n)

    std_gap = statistics.pstdev(core)
    return (std_gap / mean_gap, n)


def _compute_question_mirroring_ratio(question_text: Optional[str], answer_tokens: Sequence[str]) -> float:
    """
    Measures content-word echo in the candidate's opening 25 words vs the question.
    High verbatim mirroring followed by polished delivery is a classic AI-latency stall tactic.
    """
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
    overlap = len(q_words & opening_words)
    return overlap / len(q_words)


def _score_vision_events(events: Optional[Sequence[VisionAnomalyEvent]]) -> Tuple[float, int, List[str]]:
    """
    Severity- and confidence-weighted multimodal vision risk (capped at 30.0).
    Distinguishes hard device proof from benign brief gaze shifts.
    """
    if not events:
        return (0.0, 0, [])

    total_risk = 0.0
    notes: List[str] = []
    high_severity_types: set[str] = set()

    for ev in events:
        raw_type = getattr(ev, "anomaly_type", "")
        type_name = getattr(raw_type, "value", str(raw_type)).upper()
        base_weight = VISION_BASE_WEIGHTS.get(type_name, 8.0)

        conf = max(0.4, min(1.0, float(getattr(ev, "confidence", 0.8) or 0.8)))
        duration_ms = int(getattr(ev, "duration_ms", 3_000) or 3_000)

        # Gaze/head pitch below 3s receives minimal weight to prevent neurodivergent false positives
        if type_name in ("GAZE_DEVIATION", "HEAD_PITCH_DOWN"):
            dur_scale = min(1.5, max(0.35, duration_ms / 4_000.0))
        else:
            dur_scale = 1.0
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


def compute_candidate_baseline(
    session_id: str,
    q0_text: str,
    duration_ms: int,
    words: Optional[List[Dict[str, Any]]] = None,
    ttfs_ms: float = 1200.0,
) -> CandidateBaseline:
    """
    Stage 0: Calibrate candidate baseline on Question 0 ('Tell me about yourself')
    using time-windowed median/MAD and Empirical Bayes shrinkage for short samples
    to prevent ESL and neurodivergent false positives.
    """
    raw_tokens = WORD_TOKEN_RE.findall(q0_text or "")
    word_count = len(raw_tokens)
    dur_min = max(duration_ms / 60_000.0, 0.05)
    overall_wpm = word_count / dur_min

    # Context-aware disfluency rate (fillers + stutters + self-repairs per 100 words)
    disfluency_count, _ = _count_natural_disfluencies(q0_text or "")
    raw_disfluency_rate = (disfluency_count / max(word_count, 1)) * 100.0

    # Parse word timings once for pause ratio and true time-windowed WPM
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
        # Guard against artificially narrow or explosive MAD
        calibrated_mad = max(12.0, min(raw_mad_wpm, 45.0))
    else:
        raw_median_wpm = overall_wpm
        calibrated_mad = PRIOR_MAD_WPM

    # Empirical Bayes shrinkage: prevents noisy 10-word Q0 intros from skewing the entire interview
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
    Stage 1: Pure Python deterministic forensic heuristics running in <1ms.
    Uses smooth saturating scoring, baseline-relative deltas (WPM, disfluency, TTFS),
    trimmed pause cadence CV, tiered AI tells, and severity-weighted vision correlation.
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

    # 2. Context-aware disfluency rate & baseline-relative drop
    disfluency_count, self_repair_count = _count_natural_disfluencies(safe_text)
    curr_disfluency_rate = (disfluency_count / max(word_count, 1)) * 100.0
    base_disfluency = baseline.baseline_disfluency_rate if baseline else PRIOR_DISFLUENCY_RATE
    disfluency_drop = max(base_disfluency - curr_disfluency_rate, 0.0)

    # 3. Trimmed Pause Cadence Uniformity CV (teleprompter detection)
    cadence_uniformity_cv, valid_pause_count = _compute_robust_cadence_cv(inter_word_gaps)

    # 4. Tiered Spoken LLM Tell Bank + Written-Essay Formality Check
    ai_tell_matches: List[str] = []
    tier_a_hits = 0
    tier_b_hits = 0
    tell_risk = 0.0

    for pattern, name, weight in TIER_A_AI_TELLS:
        if pattern.search(safe_text):
            ai_tell_matches.append(name)
            tier_a_hits += 1
            tell_risk += weight

    for pattern, name, weight in TIER_B_AI_TELLS:
        if pattern.search(safe_text):
            ai_tell_matches.append(name)
            tier_b_hits += 1
            tell_risk += weight

    contraction_count = len(CONTRACTION_RE.findall(safe_text))
    contraction_deficit = word_count >= 80 and contraction_count == 0 and (tier_a_hits + tier_b_hits) >= 2
    if contraction_deficit:
        tell_risk += 6.0

    # Cap linguistic tells alone at 30.0 so vocabulary alone never produces an extreme accusation
    tell_risk = min(tell_risk, 30.0)

    # 5. First-person narrative ownership density & opening question-mirroring stall
    first_person_matches = FIRST_PERSON_RE.findall(safe_text)
    first_person_ratio = len(first_person_matches) / max(word_count, 1)
    mirroring_ratio = _compute_question_mirroring_ratio(question_text, tokens)

    # 6. Baseline-relative TTFS (AI generation latency stall before polished speech)
    base_ttfs = baseline.baseline_ttfs_ms if baseline else PRIOR_TTFS_MS
    ttfs_excess_ms = max(0.0, float(ttfs_latency_ms) - max(base_ttfs, 1_000.0))

    # 7. Severity-weighted concurrent vision telemetry
    vision_risk, concurrent_vision_flags, vision_notes = _score_vision_events(concurrent_vision_events)

    # 8. Spoken prompt-injection security scan
    injection_detected = any(p.search(safe_text) for p in INJECTION_PATTERNS)

    # -----------------------------------------------------------------------
    # Smooth Deterministic Risk Aggregation (Bounded 0.0 .. 100.0)
    # -----------------------------------------------------------------------
    risk = 0.0
    trigger_reasons: List[str] = []

    if injection_detected:
        risk += 25.0
        trigger_reasons.append("Security alert: spoken evaluator-manipulation phrasing detected")

    # Disfluency collapse (smooth ramp; dampened if candidate performed genuine self-repairs)
    if word_count >= 60 and disfluency_drop > 1.5:
        disfluency_pts = _linear_ramp(disfluency_drop, low=1.5, high=4.5, max_points=22.0)
        if self_repair_count >= 2:
            disfluency_pts *= 0.5
        if disfluency_pts >= 5.0:
            risk += disfluency_pts
            trigger_reasons.append(
                f"Disfluency collapse vs Q0 baseline (-{round(disfluency_drop, 1)}/100w)"
            )

    if tell_risk > 0.0:
        risk += tell_risk
        if tier_a_hits > 0:
            trigger_reasons.append(f"Matched {len(ai_tell_matches)} LLM prose tells ({tier_a_hits} high-precision)")
        else:
            trigger_reasons.append(f"Matched {len(ai_tell_matches)} structured essay transitions")
        if contraction_deficit:
            trigger_reasons.append("Zero spoken contractions across 80+ word structured monologue")

    # Baseline-relative pacing shift
    wpm_pts = _linear_ramp(abs(wpm_robust_zscore), low=1.8, high=3.5, max_points=14.0)
    if wpm_pts >= 4.0 and word_count >= 40:
        risk += wpm_pts
        trigger_reasons.append(f"Significant WPM shift vs baseline (Z={round(wpm_robust_zscore, 1)})")

    # Teleprompter uniform pause signature
    if word_count >= 60 and valid_pause_count >= 5 and cadence_uniformity_cv < 0.50:
        cadence_pts = _linear_ramp(0.50 - cadence_uniformity_cv, low=0.0, high=0.25, max_points=14.0)
        if cadence_pts >= 4.0:
            risk += cadence_pts
            trigger_reasons.append(
                f"Uniform pause cadence (CV={round(cadence_uniformity_cv, 2)}, n={valid_pause_count})"
            )

    # Vision correlation
    if vision_risk > 0.0:
        risk += vision_risk
        trigger_reasons.extend(vision_notes)

    # AI generation stall + unusually polished delivery
    if word_count >= 50 and ttfs_excess_ms >= 1_400.0 and curr_disfluency_rate < 0.8:
        stall_pts = _linear_ramp(ttfs_excess_ms, low=1_400.0, high=3_500.0, max_points=12.0)
        if stall_pts >= 3.0:
            risk += stall_pts
            trigger_reasons.append(
                f"Latency stall (+{int(ttfs_excess_ms)}ms vs baseline) before polished delivery"
            )

    # Low personal ownership (modest weight since conceptual questions naturally lack "I/my")
    if word_count >= 80 and first_person_ratio < 0.015:
        ownership_pts = _linear_ramp(0.015 - first_person_ratio, low=0.0, high=0.015, max_points=7.0)
        if ownership_pts >= 2.5:
            risk += ownership_pts
            trigger_reasons.append("Low personal ownership / first-person narrative markers")

    # Question-mirroring stall
    if mirroring_ratio >= 0.60 and word_count >= 60:
        risk += 6.0
        trigger_reasons.append(f"Opening question mirroring ({int(mirroring_ratio * 100)}% keyword echo)")

    # Authentic live-speech credit: multiple spontaneous self-repairs dampen non-vision risk
    if self_repair_count >= 2 and vision_risk < 15.0 and not injection_detected:
        risk = max(0.0, risk - 8.0)

    deterministic_risk = min(max(risk, 0.0), 100.0)

    # -----------------------------------------------------------------------
    # Cost-Saving Stage 2 LLM Judge Pre-Filter Gate
    # Filters out ~75-85% of normal turns at $0 while escalating multi-signal
    # anomalies, confirmed device flags, and security attacks immediately.
    # -----------------------------------------------------------------------
    trigger_llm_judge = False
    if injection_detected:
        trigger_llm_judge = True
    elif word_count >= 50:
        if (
            deterministic_risk >= 40.0
            or vision_risk >= 15.0
            or (tier_a_hits >= 1 and (len(ai_tell_matches) >= 2 or disfluency_drop >= 2.0))
            or len(ai_tell_matches) >= 3
            or (disfluency_drop >= 2.5 and curr_disfluency_rate < 1.0)
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