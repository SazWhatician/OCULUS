import re
import math
from typing import List, Dict, Any, Optional
import numpy as np

from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    VisionAnomalyEvent,
    VisionAnomalyType,
)

FILLER_REGEX = re.compile(
    r"\b(um|uh|erm|like|you know|basically|actually|sort of|kind of|i mean)\b",
    re.IGNORECASE,
)

AI_TELL_PATTERNS = [
    (re.compile(r"\bdelve into\b", re.IGNORECASE), "delve into"),
    (re.compile(r"\bit'?s (crucial|important) to note\b", re.IGNORECASE), "it's crucial to note"),
    (re.compile(r"\bin summary\b", re.IGNORECASE), "in summary"),
    (re.compile(r"\bmultifaceted\b", re.IGNORECASE), "multifaceted"),
    (re.compile(r"\bcomprehensive (approach|understanding)\b", re.IGNORECASE), "comprehensive approach"),
    (re.compile(r"\bfirstly\b.*\bsecondly\b", re.IGNORECASE), "firstly...secondly"),
    (re.compile(r"\bin conclusion\b", re.IGNORECASE), "in conclusion"),
    (re.compile(r"\bfoster(s|ing)? synergy\b", re.IGNORECASE), "foster synergy"),
    (re.compile(r"\bplay(s)? a pivotal role\b", re.IGNORECASE), "plays a pivotal role"),
]

FIRST_PERSON_REGEX = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)


def compute_candidate_baseline(
    session_id: str,
    q0_text: str,
    duration_ms: int,
    words: Optional[List[Dict[str, Any]]] = None,
    ttfs_ms: float = 1200.0,
) -> CandidateBaseline:
    """
    Stage 0: Calibrate candidate baseline on Question 0 ('Tell me about yourself')
    using robust median and MAD metrics to prevent ESL/neurodivergent bias.
    """
    raw_tokens = [w for w in q0_text.strip().split() if w]
    word_count = len(raw_tokens)
    dur_min = max(duration_ms / 60000.0, 0.05)
    overall_wpm = word_count / dur_min

    # Calculate filler disfluency rate (fillers per 100 words)
    fillers = FILLER_REGEX.findall(q0_text)
    disfluency_rate = (len(fillers) / max(word_count, 1)) * 100.0

    # Calculate pauses ratio (gaps > 600ms)
    pause_ratio = 0.10
    wpm_subwindows = []

    if words and len(words) > 5:
        # Measure inter-word gaps
        gaps = []
        for i in range(1, len(words)):
            gap = words[i].get("start_ms", 0) - words[i - 1].get("end_ms", 0)
            if gap > 0:
                gaps.append(gap)
        if gaps:
            long_pauses_ms = sum(g for g in gaps if g > 600)
            pause_ratio = min(long_pauses_ms / max(duration_ms, 1), 0.95)

        # Compute per-window WPM (approx 10s windows)
        chunk_size = max(word_count // 4, 5)
        for i in range(0, word_count, chunk_size):
            chunk = raw_tokens[i : i + chunk_size]
            if chunk:
                wpm_subwindows.append(len(chunk) / (dur_min / max(len(raw_tokens) / chunk_size, 1)))

    if not wpm_subwindows:
        wpm_subwindows = [overall_wpm]

    wpm_arr = np.array(wpm_subwindows, dtype=float)
    median_wpm = float(np.median(wpm_arr))
    mad_wpm = float(np.median(np.abs(wpm_arr - median_wpm)))
    if mad_wpm < 5.0:
        mad_wpm = 18.0

    return CandidateBaseline(
        session_id=session_id,
        baseline_wpm=round(median_wpm, 1),
        mad_wpm=round(mad_wpm, 1),
        baseline_disfluency_rate=round(disfluency_rate, 2),
        baseline_pause_ratio=round(pause_ratio, 3),
        baseline_ttfs_ms=round(ttfs_ms, 1),
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
) -> HeuristicScores:
    """
    Stage 1: Pure Python deterministic heuristics running in 0ms.
    Evaluates WPM shift, relative disfluency drop, cadence uniformity, AI tells, and vision flags.
    """
    tokens = [w for w in answer_text.strip().split() if w]
    word_count = len(tokens)
    dur_min = max(duration_ms / 60000.0, 0.05)
    curr_wpm = word_count / dur_min

    # 1. Robust WPM shift
    base_wpm = baseline.baseline_wpm if baseline else 130.0
    mad = baseline.mad_wpm if baseline and baseline.mad_wpm > 0 else 18.0
    wpm_robust_zscore = 0.6745 * (curr_wpm - base_wpm) / max(mad, 18.0)

    # 2. Relative disfluency drop vs baseline
    fillers = FILLER_REGEX.findall(answer_text)
    curr_disfluency_rate = (len(fillers) / max(word_count, 1)) * 100.0
    base_disfluency = baseline.baseline_disfluency_rate if baseline else 3.5
    disfluency_drop = max(base_disfluency - curr_disfluency_rate, 0.0)

    # 3. Cadence Uniformity CV (teleprompter detection)
    cadence_uniformity_cv = 0.75
    if inter_word_gaps and len(inter_word_gaps) >= 5:
        std_gap = float(np.std(inter_word_gaps))
        mean_gap = float(np.mean(inter_word_gaps))
        if mean_gap > 0:
            cadence_uniformity_cv = std_gap / mean_gap

    # 4. Spoken LLM Tell bank
    ai_tell_matches = []
    for pattern, name in AI_TELL_PATTERNS:
        if pattern.search(answer_text):
            ai_tell_matches.append(name)

    # 5. First-person density
    first_person_matches = FIRST_PERSON_REGEX.findall(answer_text)
    first_person_ratio = len(first_person_matches) / max(word_count, 1)

    # 6. Concurrent vision flags
    concurrent_vision_flags = len(concurrent_vision_events) if concurrent_vision_events else 0

    # Deterministic Risk Calculation (clamped to 0..100)
    risk = 0.0
    trigger_reasons = []

    if disfluency_drop >= 2.5 and word_count >= 80:
        risk += 25.0
        trigger_reasons.append("Sharp disfluency drop vs candidate baseline")

    if ai_tell_matches:
        count = min(len(ai_tell_matches), 3)
        risk += 15.0 * count
        trigger_reasons.append(f"Matched {len(ai_tell_matches)} formulaic AI tell phrases")

    if abs(wpm_robust_zscore) >= 2.2:
        risk += 15.0
        trigger_reasons.append(f"Significant cadence shift (Z={round(wpm_robust_zscore, 1)})")

    if cadence_uniformity_cv < 0.45 and word_count >= 60:
        risk += 15.0
        trigger_reasons.append("Abnormally uniform pacing (teleprompter signature)")

    if concurrent_vision_flags > 0:
        count = min(concurrent_vision_flags, 2)
        risk += 15.0 * count
        trigger_reasons.append(f"{concurrent_vision_flags} concurrent off-screen gaze/device flags")

    if ttfs_latency_ms >= 2500 and curr_disfluency_rate < 0.5:
        risk += 10.0
        trigger_reasons.append("AI generation latency stall before polished delivery")

    if first_person_ratio < 0.015 and word_count >= 80:
        risk += 10.0
        trigger_reasons.append("Absence of personal narrative or experience markers")

    deterministic_risk = min(max(risk, 0.0), 100.0)

    # Cost-Saving Pre-Filter Gate for Stage 2 LLM Judge
    trigger_llm_judge = False
    if word_count >= 50:
        if (
            deterministic_risk >= 45.0
            or len(ai_tell_matches) >= 2
            or concurrent_vision_flags >= 1
            or disfluency_drop >= 2.5
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
