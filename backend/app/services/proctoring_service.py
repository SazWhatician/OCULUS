import asyncio
import hashlib
import logging
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.adapters.llm.base import FallbackLLMRouter, ForensicReasoningToolkit
from app.config import settings
from app.models.schemas import (
    AnswerProctoring,
    ProctoringFinding,
    ProctoringReport,
)
from app.schemas.contracts import (
    CandidateBaseline,
    ConfidenceBand,
    HeuristicScores,
    LLMJudgeVerdict,
)
from app.services.aggregator import (
    compute_anti_dilution_session_risk,
    compute_vision_boost,
)
from app.services.heuristics import (
    compute_candidate_baseline,
    compute_deterministic_heuristics,
)
from app.services.vision_proctoring import CATEGORY_TO_ANOMALY_TYPE, get_vision_service

logger = logging.getLogger("oculus.proctoring.service")

PROCTORING_MODEL_VERSION = "2.0.0-oculus-multimodal"
MIN_WORDS_FOR_SCORING = 10
MIN_WORDS_FOR_SUBSTANTIVE_AGG = 25
TOKEN_BUDGET_QA_LIMIT = 25
DEFAULT_MAX_LLM_TURNS_PER_SESSION = 6
DEFAULT_LLM_MAX_CONCURRENCY = 2
ESTIMATED_WPM_FALLBACK = 135.0

ADVISORY_CAVEAT = (
    "This report is advisory only. Multimodal transcript and webcam behavioral inference "
    "cannot definitively prove AI assistance and must never be used as the sole basis "
    "for hiring decisions. Use these signals and follow-up probes to guide human review."
)

# High-precision linguistic banks (removes false-positive everyday words like "so", "in order to", "certainly")
FILLER_SINGLE = frozenset({
    "um", "uh", "uhh", "umm", "er", "erm", "ah", "ahh",
    "basically", "actually", "literally", "hmm", "mhm",
})
FILLER_MULTI = (
    "you know", "i mean", "kind of", "sort of", "i guess", "you see",
)

AI_TELL_PATTERNS = (
    re.compile(r"\bdelve\s+into\b", re.I),
    re.compile(r"\bit'?s\s+(?:crucial|important|essential)\s+to\s+note\b", re.I),
    re.compile(r"\bmultifaceted\b", re.I),
    re.compile(r"\btapestry\s+of\b", re.I),
    re.compile(r"\bever[- ]evolving\s+(?:landscape|ecosystem|field)\b", re.I),
    re.compile(r"\bplays?\s+a\s+(?:crucial|vital|pivotal)\s+role\b", re.I),
    re.compile(r"\bcomprehensive\s+(?:approach|understanding|overview|framework)\b", re.I),
    re.compile(r"\bseamlessly\s+integrat(?:e|es|ed|ing)\b", re.I),
    re.compile(r"\bharness(?:ing)?\s+the\s+power\s+of\b", re.I),
    re.compile(r"\bfoster(?:s|ing)?\s+(?:a\s+sense\s+of\s+)?synergy\b", re.I),
    re.compile(r"\bholistic\s+(?:approach|view|perspective|strategy)\b", re.I),
    re.compile(r"\btestament\s+to\b", re.I),
    re.compile(r"\bin\s+conclusion\b", re.I),
    re.compile(r"\bin\s+summary\b", re.I),
    re.compile(r"\bmoreover\b", re.I),
    re.compile(r"\bfurthermore\b", re.I),
    re.compile(r"\bkey\s+takeaways?\b", re.I),
)

STRUCTURE_MARKERS = (
    re.compile(r"\bfirstly\b", re.I),
    re.compile(r"\bsecondly\b", re.I),
    re.compile(r"\bthirdly\b", re.I),
    re.compile(r"\blastly\b", re.I),
    re.compile(r"\bin\s+addition\b", re.I),
    re.compile(r"\bon\s+the\s+other\s+hand\b", re.I),
    re.compile(r"^\s*\d+[\.\)]\s", re.M),
)

SELF_REFS = frozenset({
    "i", "i'm", "im", "i've", "ive", "i'll", "i'd",
    "my", "me", "myself", "mine", "we", "our", "ours",
})

# Maps vision_proctoring.py category strings to heuristics.py canonical VisionAnomalyType names
VISION_CATEGORY_TO_HEURISTIC_TYPE: Dict[str, str] = {
    category: anomaly_type.value for category, anomaly_type in CATEGORY_TO_ANOMALY_TYPE.items()
}


def question_hash(question: str) -> str:
    normalised = re.sub(r"\s+", " ", (question or "").strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", (text or "").lower())


def disfluency_rate(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    single = sum(1 for t in tokens if t in FILLER_SINGLE)
    text_lower = text.lower()
    multi = sum(len(re.findall(rf"\b{re.escape(p)}\b", text_lower)) for p in FILLER_MULTI)
    return (single + multi) / len(tokens)


def self_reference_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in SELF_REFS) / len(tokens)


def concreteness_per_100w(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    # Count numeric metrics/years once (fixes double-counting years) + named/technical entities
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    acronyms_or_camel = len(re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b", text))
    sentences = re.split(r"[.!?]+\s*", text)
    capitalized_mid = 0
    for sent in sentences:
        words = sent.strip().split()
        for word in words[1:]:
            if len(word) > 1 and word[0].isupper() and word[1:].islower():
                capitalized_mid += 1
    return (numbers + acronyms_or_camel + capitalized_mid) / (len(tokens) / 100.0)


def ai_tell_matches(text: str) -> List[str]:
    if not text:
        return []
    matches: List[str] = []
    for pattern in AI_TELL_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(m.group(0))
    return matches


def structure_markers_per_100w(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    count = sum(len(pattern.findall(text)) for pattern in STRUCTURE_MARKERS)
    return count / (len(tokens) / 100.0)


def extract_structure_quotes(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    for pattern in STRUCTURE_MARKERS:
        for m in pattern.finditer(text):
            val = m.group(0).strip()
            if val and val not in found:
                found.append(val)
    return found[:3]


def question_mirror_score(question: str, answer: str) -> float:
    if not (question and answer):
        return 0.0
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)[:30]
    if len(q_tokens) < 3 or len(a_tokens) < 3:
        return 0.0
    q_3 = set(zip(q_tokens, q_tokens[1:], q_tokens[2:]))
    a_3 = set(zip(a_tokens, a_tokens[1:], a_tokens[2:]))
    return (len(q_3 & a_3) / len(q_3)) if q_3 else 0.0


def sentence_length_stats(text: str) -> Dict[str, float]:
    sentences = [s.strip() for s in re.split(r"[.!?]+\s*", text or "") if s.strip()]
    if not sentences:
        return {"mean": 0.0, "stdev": 0.0, "count": 0}
    lengths = [len(s.split()) for s in sentences]
    return {
        "mean": round(float(statistics.mean(lengths)), 2),
        "stdev": round(float(statistics.stdev(lengths)), 2) if len(lengths) > 1 else 0.0,
        "count": len(sentences),
    }


def compute_answer_features(question: str, answer: str) -> Dict[str, Any]:
    tells = ai_tell_matches(answer)
    return {
        "word_count": len(tokenize(answer)),
        "disfluency_rate": round(disfluency_rate(answer), 4),
        "self_reference_rate": round(self_reference_rate(answer), 4),
        "concreteness_per_100w": round(concreteness_per_100w(answer), 2),
        "ai_tell_count": len(tells),
        "ai_tells_matched": tells[:5],
        "structure_marker_per_100w": round(structure_markers_per_100w(answer), 2),
        "mirror_score": round(question_mirror_score(question, answer), 3),
        "sentence_stats": sentence_length_stats(answer),
    }


def compute_baseline(intro_qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not intro_qa_pairs:
        return {}
    combined = " ".join(qa.get("answer", "") for qa in intro_qa_pairs if qa.get("answer"))
    if not combined.strip():
        return {}
    return {
        "based_on_word_count": len(tokenize(combined)),
        "disfluency_rate": round(disfluency_rate(combined), 4),
        "self_reference_rate": round(self_reference_rate(combined), 4),
        "concreteness_per_100w": round(concreteness_per_100w(combined), 2),
        "structure_marker_per_100w": round(structure_markers_per_100w(combined), 2),
        "sentence_stats": sentence_length_stats(combined),
    }


# Retained for schema backward compatibility with external callers/tests
class JudgeFinding(BaseModel):
    name: str = Field(default="finding")
    evidence_quote: str = Field(description="Verbatim excerpt from the candidate's answer")
    explanation: str = Field(description="Why this excerpt is suspicious")


class JudgeAnswerScore(BaseModel):
    question_index: int = Field(description="0-based index into the Q&A list")
    score: float = Field(ge=0.0, le=1.0, description="0=no concern, 1=strong concern")
    risk_band: Literal["low", "medium", "high"]
    ai_reliance_percent: Optional[float] = Field(default=None)
    speech_naturalness_score: Optional[float] = Field(default=None)
    findings: List[JudgeFinding] = Field(default_factory=list)


class JudgeSessionReport(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "medium", "high"]
    ai_reliance_percent: Optional[float] = Field(default=None)
    overall_risk_score: Optional[float] = Field(default=None)
    pattern_findings: List[str] = Field(default_factory=list)
    recommendation: str
    per_answer: List[JudgeAnswerScore]


@dataclass(frozen=True)
class _BridgedVisionEvent:
    """Normalizes VisionProctoringService events into the shape heuristics.py scores."""
    anomaly_type: str
    confidence: float
    duration_ms: int
    category: str
    signal: str
    timestamp_offset: str


class ProctoringService:
    """
    OCULUS Multimodal Forensic Proctoring Orchestrator:
      - Stage 0: Calibrates candidate Q0 baseline (Empirical Bayes MAD via heuristics.py)
      - Stage 1: Runs 0ms deterministic heuristics + temporally aligned vision events on every turn
      - Stage 2: Escalates ONLY gated anomalous turns (budget-capped, bounded concurrency)
                 through FallbackLLMRouter (Groq -> Gemini Flash + zero-trust quote verification)
      - Fusion:  Anti-dilution Top-K substantive aggregation + hard multimodal vision fusion
    """

    def __init__(self, llm_router: Optional[FallbackLLMRouter] = None) -> None:
        self._llm_router: Optional[FallbackLLMRouter] = llm_router
        self.max_llm_turns_per_session: int = int(
            getattr(settings, "PROCTORING_MAX_LLM_TURNS", DEFAULT_MAX_LLM_TURNS_PER_SESSION)
        )
        self.llm_concurrency: int = int(
            getattr(settings, "PROCTORING_LLM_CONCURRENCY", DEFAULT_LLM_MAX_CONCURRENCY)
        )

    def _get_llm_router(self) -> FallbackLLMRouter:
        if self._llm_router is None:
            self._llm_router = FallbackLLMRouter()
        return self._llm_router

    async def aclose(self) -> None:
        if self._llm_router is not None:
            await self._llm_router.aclose()

    async def score_session(self, transcript: dict, job_title: str) -> ProctoringReport:
        start = time.perf_counter()
        qa_pairs: List[Dict[str, Any]] = (transcript or {}).get("qa_pairs") or []
        if not qa_pairs:
            return self._empty_report("No Q&A turns present in transcript.", elapsed_ms=0)

        session_id = self._resolve_session_id(transcript, qa_pairs)
        vision_svc = get_vision_service()
        vision_summary = vision_svc.get_session_summary(session_id, allow_demo_mock=False)
        vision_events_raw = vision_summary.get("anomaly_events", [])

        # Stage 0: Identify Q0 introduction turn(s) and build calibrated baseline
        intro_indices, intro_pairs = self._select_intro_pairs(qa_pairs)
        baseline_summary = compute_baseline(intro_pairs)
        candidate_baseline = self._build_candidate_baseline(session_id, intro_pairs)
        baseline_summary.update({
            "baseline_wpm": candidate_baseline.baseline_wpm,
            "mad_wpm": candidate_baseline.mad_wpm,
            "baseline_disfluency_rate_per_100w": candidate_baseline.baseline_disfluency_rate,
            "baseline_ttfs_ms": candidate_baseline.baseline_ttfs_ms,
        })

        # Stage 1: Deterministic 0ms forensics + temporal vision correlation on ALL turns
        legacy_features: List[Dict[str, Any]] = []
        stage1_scores: List[HeuristicScores] = []
        turn_vision_links: List[List[_BridgedVisionEvent]] = []
        turn_injection_hits: List[List[str]] = []

        for idx, qa in enumerate(qa_pairs):
            question = str(qa.get("question") or "")
            answer = str(qa.get("answer") or "")

            feats = compute_answer_features(question, answer)
            timing = self._extract_turn_timing(qa, feats["word_count"])
            bridged_vision = self._fetch_aligned_vision_events(vision_svc, session_id, timing)
            injection_hits = ForensicReasoningToolkit.detect_injection_attempts(answer)

            h_score = compute_deterministic_heuristics(
                answer_text=answer,
                duration_ms=timing["duration_ms"],
                baseline=candidate_baseline,
                inter_word_gaps=timing["inter_word_gaps"],
                ttfs_latency_ms=timing["ttfs_ms"],
                concurrent_vision_events=bridged_vision,
                question_text=question,
            )

            # Enrich feature dict cleanly for UI explainability
            feats.update({
                "question_hash": question_hash(question),
                "wpm": h_score.wpm,
                "wpm_robust_zscore": h_score.wpm_robust_zscore,
                "disfluency_drop_per_100w": h_score.disfluency_drop,
                "cadence_uniformity_cv": h_score.cadence_uniformity_cv,
                "ttfs_latency_ms": h_score.ttfs_latency_ms,
                "concurrent_vision_flags": h_score.concurrent_vision_flags,
                "deterministic_risk": h_score.deterministic_risk,
                "stage1_trigger_reasons": list(h_score.trigger_reasons),
                "timing_estimated": timing["estimated"],
                "security_injection_detected": bool(injection_hits),
            })

            legacy_features.append(feats)
            stage1_scores.append(h_score)
            turn_vision_links.append(bridged_vision)
            turn_injection_hits.append(injection_hits)

        # Stage 2: Select gated anomalous turns within per-session cost budget
        triggered_indices = self._select_stage2_candidates(
            qa_pairs=qa_pairs,
            stage1_scores=stage1_scores,
            intro_indices=intro_indices,
            turn_injection_hits=turn_injection_hits,
            turn_vision_links=turn_vision_links,
        )

        llm_verdicts, llm_failures = await self._evaluate_stage2_batch(
            indices=triggered_indices,
            qa_pairs=qa_pairs,
            stage1_scores=stage1_scores,
            baseline=candidate_baseline,
            job_title=job_title,
        )

        # Status is "degraded" ONLY if a triggered Stage 2 evaluation failed to return a verdict.
        # Clean interviews that legitimately bypass Stage 2 stay "ok".
        status: Literal["ok", "degraded", "failed"] = "degraded" if llm_failures > 0 else "ok"

        return self._aggregate(
            qa_pairs=qa_pairs,
            legacy_features=legacy_features,
            stage1_scores=stage1_scores,
            llm_verdicts=llm_verdicts,
            turn_vision_links=turn_vision_links,
            turn_injection_hits=turn_injection_hits,
            intro_indices=intro_indices,
            baseline_summary=baseline_summary,
            candidate_baseline=candidate_baseline,
            vision_summary=vision_summary,
            vision_events_raw=vision_events_raw,
            status=status,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    # -----------------------------------------------------------------------
    # Stage 0 & Stage 1 Preparation Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _resolve_session_id(transcript: dict, qa_pairs: List[Dict[str, Any]]) -> str:
        for candidate in (
            transcript.get("session_id"),
            qa_pairs[0].get("session_id") if qa_pairs else None,
        ):
            if candidate and str(candidate).strip():
                return str(candidate).strip()
        return "unassigned_session"

    @staticmethod
    def _select_intro_pairs(qa_pairs: List[Dict[str, Any]]) -> Tuple[set[int], List[Dict[str, Any]]]:
        explicit = [
            (i, qa)
            for i, qa in enumerate(qa_pairs)
            if str(qa.get("category") or "").strip().lower() in ("introduction", "intro", "q0")
        ]
        if explicit:
            return ({i for i, _ in explicit}, [qa for _, qa in explicit])

        # Avoid calibrating on a 3-word mic check ("Yes I can hear you"); pick first substantive intro
        for i in range(min(2, len(qa_pairs))):
            if len(tokenize(str(qa_pairs[i].get("answer") or ""))) >= MIN_WORDS_FOR_SUBSTANTIVE_AGG:
                return ({i}, [qa_pairs[i]])

        return ({0}, [qa_pairs[0]])

    def _build_candidate_baseline(
        self,
        session_id: str,
        intro_pairs: List[Dict[str, Any]],
    ) -> CandidateBaseline:
        combined_text = " ".join(str(qa.get("answer") or "").strip() for qa in intro_pairs).strip()
        word_count = len(tokenize(combined_text))

        total_dur_ms = 0
        all_words: List[Dict[str, Any]] = []
        ttfs_samples: List[float] = []

        for qa in intro_pairs:
            t = self._extract_turn_timing(qa, len(tokenize(str(qa.get("answer") or ""))))
            total_dur_ms += t["duration_ms"]
            if isinstance(qa.get("words"), list):
                all_words.extend(qa["words"])
            if t["ttfs_ms"] > 0:
                ttfs_samples.append(float(t["ttfs_ms"]))

        if total_dur_ms <= 0:
            total_dur_ms = max(2_000, int((word_count / ESTIMATED_WPM_FALLBACK) * 60_000))

        avg_ttfs = statistics.fmean(ttfs_samples) if ttfs_samples else 1200.0
        return compute_candidate_baseline(
            session_id=session_id,
            q0_text=combined_text,
            duration_ms=total_dur_ms,
            words=all_words or None,
            ttfs_ms=avg_ttfs,
        )

    @staticmethod
    def _extract_turn_timing(qa: Dict[str, Any], word_count: int) -> Dict[str, Any]:
        timing_obj = qa.get("timing") if isinstance(qa.get("timing"), dict) else {}

        start_ms = qa.get("start_ms", timing_obj.get("start_ms"))
        end_ms = qa.get("end_ms", timing_obj.get("end_ms"))
        duration_ms = qa.get("duration_ms", timing_obj.get("duration_ms"))
        ttfs_ms = (
            qa.get("ttfs_latency_ms")
            or qa.get("pre_answer_silence_ms")
            or timing_obj.get("ttfs_latency_ms")
            or timing_obj.get("pre_answer_silence_ms")
            or 0
        )

        estimated = False
        if duration_ms is None and start_ms is not None and end_ms is not None:
            try:
                duration_ms = max(500, int(end_ms) - int(start_ms))
            except (TypeError, ValueError):
                duration_ms = None

        if duration_ms is None or int(duration_ms) <= 0:
            # Estimate at neutral prior WPM so missing timestamps never trigger false WPM Z-scores
            duration_ms = max(2_000, int((max(word_count, 1) / ESTIMATED_WPM_FALLBACK) * 60_000))
            estimated = True
        else:
            duration_ms = int(duration_ms)

        inter_word_gaps = qa.get("inter_word_gaps")
        if not inter_word_gaps and isinstance(qa.get("words"), list) and len(qa["words"]) >= 2:
            gaps: List[int] = []
            words = qa["words"]
            for i in range(1, len(words)):
                prev_end = words[i - 1].get("end_ms") if isinstance(words[i - 1], dict) else getattr(words[i - 1], "end_ms", None)
                curr_start = words[i].get("start_ms") if isinstance(words[i], dict) else getattr(words[i], "start_ms", None)
                if prev_end is not None and curr_start is not None:
                    gap = int(curr_start) - int(prev_end)
                    if gap >= 0:
                        gaps.append(gap)
            inter_word_gaps = gaps or None

        return {
            "start_ms": int(start_ms) if start_ms is not None else None,
            "end_ms": int(end_ms) if end_ms is not None else None,
            "duration_ms": duration_ms,
            "ttfs_ms": max(0, int(ttfs_ms or 0)),
            "inter_word_gaps": inter_word_gaps,
            "estimated": estimated,
        }

    @staticmethod
    def _fetch_aligned_vision_events(
        vision_svc: Any,
        session_id: str,
        timing: Dict[str, Any],
    ) -> List[_BridgedVisionEvent]:
        start_ms = timing.get("start_ms")
        end_ms = timing.get("end_ms")
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            return []

        # Include a 2-second pre-answer buffer to capture question-reading / phone-typing glances
        raw_events = vision_svc.get_events_in_window(session_id, max(0, start_ms - 2_000), end_ms)
        bridged: List[_BridgedVisionEvent] = []
        for ev in raw_events:
            cat = str(getattr(ev, "category", ""))
            mapped_type = VISION_CATEGORY_TO_HEURISTIC_TYPE.get(cat, "GAZE_DEVIATION")
            bridged.append(
                _BridgedVisionEvent(
                    anomaly_type=mapped_type,
                    confidence=float(getattr(ev, "confidence", 0.8) or 0.8),
                    duration_ms=int(getattr(ev, "duration_ms", 3_000) or 3_000),
                    category=cat,
                    signal=str(getattr(ev, "signal", "")),
                    timestamp_offset=str(getattr(ev, "timestamp_offset", "00:00:00")),
                )
            )
        return bridged

    # -----------------------------------------------------------------------
    # Stage 2 Gated & Budget-Capped Concurrent LLM Evaluation
    # -----------------------------------------------------------------------
    def _select_stage2_candidates(
        self,
        qa_pairs: List[Dict[str, Any]],
        stage1_scores: List[HeuristicScores],
        intro_indices: set[int],
        turn_injection_hits: List[List[str]],
        turn_vision_links: List[List[_BridgedVisionEvent]],
    ) -> List[int]:
        candidates: List[Tuple[float, int]] = []
        for idx, h in enumerate(stage1_scores):
            has_injection = bool(turn_injection_hits[idx])
            has_hard_vision = any(
                ev.anomaly_type in ("DEVICE_DETECTED", "MULTI_FACE") for ev in turn_vision_links[idx]
            )
            # Do not burn LLM quota on Q0 intro unless tampered or confirmed secondary device
            if idx in intro_indices and not (has_injection or has_hard_vision):
                continue
            if h.trigger_llm_judge or has_injection or (has_hard_vision and h.word_count >= MIN_WORDS_FOR_SCORING):
                priority = h.deterministic_risk + (50.0 if has_injection else 0.0) + (25.0 if has_hard_vision else 0.0)
                candidates.append((priority, idx))

        # Prioritize highest-risk turns up to per-session budget cap
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [idx for _, idx in candidates[: self.max_llm_turns_per_session]]
        selected.sort()
        return selected

    async def _evaluate_stage2_batch(
        self,
        indices: List[int],
        qa_pairs: List[Dict[str, Any]],
        stage1_scores: List[HeuristicScores],
        baseline: CandidateBaseline,
        job_title: str,
    ) -> Tuple[Dict[int, LLMJudgeVerdict], int]:
        if not indices:
            return ({}, 0)

        router = self._get_llm_router()
        sem = asyncio.Semaphore(max(1, self.llm_concurrency))
        verdicts: Dict[int, LLMJudgeVerdict] = {}
        failures = 0

        async def _eval_one(idx: int) -> Tuple[int, Optional[LLMJudgeVerdict]]:
            qa = qa_pairs[idx]
            async with sem:
                try:
                    verdict = await router.evaluate_turn(
                        question_text=str(qa.get("question") or ""),
                        answer_text=str(qa.get("answer") or ""),
                        job_title=job_title,
                        baseline=baseline,
                        heuristics=stage1_scores[idx],
                    )
                    return (idx, verdict)
                except Exception as exc:
                    logger.warning("Stage 2 router error on turn %d: %s", idx, exc)
                    return (idx, None)

        results = await asyncio.gather(*(_eval_one(i) for i in indices))
        for idx, verdict in results:
            if verdict is not None:
                verdicts[idx] = verdict
            else:
                failures += 1

        return (verdicts, failures)

    # -----------------------------------------------------------------------
    # Multimodal Fusion & Anti-Dilution Session Aggregation
    # -----------------------------------------------------------------------
    def _aggregate(
        self,
        qa_pairs: List[Dict[str, Any]],
        legacy_features: List[Dict[str, Any]],
        stage1_scores: List[HeuristicScores],
        llm_verdicts: Dict[int, LLMJudgeVerdict],
        turn_vision_links: List[List[_BridgedVisionEvent]],
        turn_injection_hits: List[List[str]],
        intro_indices: set[int],
        baseline_summary: Dict[str, Any],
        candidate_baseline: CandidateBaseline,
        vision_summary: Dict[str, Any],
        vision_events_raw: List[Dict[str, Any]],
        status: Literal["ok", "degraded", "failed"],
        elapsed_ms: int,
    ) -> ProctoringReport:
        per_answer: List[AnswerProctoring] = []
        turn_metrics: List[Tuple[float, int, bool, bool]] = []  # (risk_0_100, word_count, is_intro, force_include)
        advisory_probes: List[str] = []
        only_turn = len(qa_pairs) == 1

        for i, (qa, feats, h) in enumerate(zip(qa_pairs, legacy_features, stage1_scores)):
            question = str(qa.get("question") or "")
            answer = str(qa.get("answer") or "")
            verdict = llm_verdicts.get(i)
            vision_links = turn_vision_links[i]
            injection_hits = turn_injection_hits[i]

            if verdict is not None:
                # Calibrated Stage 1 + Stage 2 fusion (dampens LLM weight if LLM confidence is LOW)
                if verdict.confidence_band == ConfidenceBand.LOW:
                    fused_risk = (0.55 * h.deterministic_risk) + (0.45 * verdict.suspicion_score)
                else:
                    fused_risk = (0.35 * h.deterministic_risk) + (0.65 * verdict.suspicion_score)
                ai_rel = round(verdict.suspicion_score, 1)
                findings = self._build_verdict_findings(verdict, vision_links, injection_hits)
                feats["llm_provider_used"] = verdict.provider_used
                feats["llm_confidence_band"] = verdict.confidence_band.value
                feats["llm_assessment"] = verdict.recitation_vs_live_assessment
                feats["advisory_follow_up_question"] = verdict.advisory_follow_up_question
                if verdict.advisory_follow_up_question and fused_risk >= 45.0:
                    advisory_probes.append(f"Q{i + 1} Probe: {verdict.advisory_follow_up_question}")
            else:
                fused_risk = h.deterministic_risk
                ai_rel = round(h.deterministic_risk, 1)
                findings = self._build_heuristic_findings(answer, feats, h, vision_links, injection_hits)
                feats["llm_provider_used"] = "stage1_gated_skip" if not h.trigger_llm_judge else "stage1_degraded"

            # Short-answer safeguard (unless explicit spoken prompt injection occurred)
            if feats["word_count"] < MIN_WORDS_FOR_SCORING and not injection_hits:
                fused_risk = min(fused_risk, 15.0)
                ai_rel = min(ai_rel, 15.0)

            fused_risk = round(max(0.0, min(100.0, fused_risk)), 1)
            score_unit = round(fused_risk / 100.0, 3)
            band = self._risk_band(fused_risk)
            nat_score = self._compute_speech_naturalness(feats, h, candidate_baseline)

            per_answer.append(
                AnswerProctoring(
                    question_index=i,
                    question_preview=question[:100],
                    score=score_unit,
                    risk_band=band,
                    ai_reliance_percent=round(max(0.0, min(100.0, ai_rel)), 1),
                    speech_naturalness_score=nat_score,
                    findings=findings,
                    heuristic_features=feats,
                )
            )

            has_hard_vision = any(
                ev.anomaly_type in ("DEVICE_DETECTED", "PHONE_DETECTED", "MULTI_FACE", "LIVENESS_SPOOF")
                for ev in vision_links
            )
            turn_metrics.append(
                (fused_risk, feats["word_count"], (i in intro_indices) and not only_turn, bool(injection_hits or has_hard_vision))
            )

        # Shared Anti-Dilution Aggregation + Multimodal Vision Fusion
        vision_risk_boost, vision_risk_index = self._compute_vision_fusion_boost(vision_summary)
        overall_risk = compute_anti_dilution_session_risk(turn_metrics, vision_boost=vision_risk_boost)
        overall_score = round(overall_risk / 100.0, 3)
        overall_band = self._risk_band(overall_risk)

        overall_ai_reliance = round(
            statistics.fmean([a.ai_reliance_percent for a in per_answer]) if per_answer else 0.0,
            1,
        )
        share_high_risk = round(
            sum(1 for a in per_answer if a.risk_band == "high") / max(len(per_answer), 1),
            3,
        )
        avg_nat_score = round(
            statistics.fmean([a.speech_naturalness_score for a in per_answer]) if per_answer else 100.0,
            1,
        )

        risk_factors_breakdown = self._build_risk_breakdown(
            legacy_features=legacy_features,
            stage1_scores=stage1_scores,
            avg_nat_score=avg_nat_score,
            vision_risk_index=vision_risk_index,
        )

        pattern_findings = self._synthesize_pattern_findings(
            per_answer=per_answer,
            stage1_scores=stage1_scores,
            turn_injection_hits=turn_injection_hits,
            vision_summary=vision_summary,
            llm_count=len(llm_verdicts),
            advisory_probes=advisory_probes,
        )
        recommendation = self._build_recommendation(overall_band, overall_risk, vision_summary, turn_injection_hits)

        return ProctoringReport(
            overall_score=overall_score,
            overall_risk_score=overall_risk,
            ai_reliance_percent=overall_ai_reliance,
            speech_naturalness_score=avg_nat_score,
            risk_band=overall_band,
            share_high_risk=share_high_risk,
            risk_factors_breakdown=risk_factors_breakdown,
            vision_summary=vision_summary,
            vision_events=vision_events_raw,
            pattern_findings=pattern_findings,
            recommendation=recommendation,
            caveat=ADVISORY_CAVEAT,
            per_answer=per_answer,
            baseline_summary=baseline_summary,
            model_version=PROCTORING_MODEL_VERSION,
            status=status,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _build_verdict_findings(
        verdict: LLMJudgeVerdict,
        vision_links: List[_BridgedVisionEvent],
        injection_hits: List[str],
    ) -> List[ProctoringFinding]:
        findings: List[ProctoringFinding] = []
        quotes = verdict.verbatim_evidence_quotes
        signals = verdict.reasoning_signals

        if injection_hits:
            findings.append(
                ProctoringFinding(
                    name="prompt_injection_attempt",
                    evidence_quote=injection_hits[0][:80],
                    explanation="Spoken transcript contained direct instructions attempting to manipulate automated scoring.",
                )
            )

        for idx, signal in enumerate(signals[:4]):
            quote = quotes[idx] if idx < len(quotes) else (quotes[0] if quotes else "Telemetry / structural signal")
            findings.append(
                ProctoringFinding(
                    name="llm_forensic_signal",
                    evidence_quote=quote[:120],
                    explanation=signal[:180],
                )
            )

        for ev in vision_links[:2]:
            findings.append(
                ProctoringFinding(
                    name="concurrent_vision_anomaly",
                    evidence_quote=f"[{ev.timestamp_offset}] {ev.category}",
                    explanation=ev.signal[:180],
                )
            )

        return findings

    @staticmethod
    def _build_heuristic_findings(
        answer_text: str,
        feats: Dict[str, Any],
        h: HeuristicScores,
        vision_links: List[_BridgedVisionEvent],
        injection_hits: List[str],
    ) -> List[ProctoringFinding]:
        """Emits ONLY grounded findings backed by real matched substrings or telemetry—never fabricated quotes."""
        findings: List[ProctoringFinding] = []

        if injection_hits:
            findings.append(
                ProctoringFinding(
                    name="prompt_injection_attempt",
                    evidence_quote=injection_hits[0][:80],
                    explanation="Spoken transcript contained evaluator-manipulation phrasing.",
                )
            )

        if h.ai_tell_matches:
            findings.append(
                ProctoringFinding(
                    name="ai_tell_pattern",
                    evidence_quote=", ".join(h.ai_tell_matches[:3]),
                    explanation=f"Matched {len(h.ai_tell_matches)} LLM prose transition artifact(s).",
                )
            )

        if feats.get("structure_marker_per_100w", 0.0) > 1.8:
            real_markers = extract_structure_quotes(answer_text)
            if real_markers:
                findings.append(
                    ProctoringFinding(
                        name="formulaic_enumeration",
                        evidence_quote=" / ".join(real_markers),
                        explanation="Dense written-style structural signposting inside spoken answer.",
                    )
                )

        if h.disfluency_drop >= 2.2 and h.word_count >= 60:
            findings.append(
                ProctoringFinding(
                    name="baseline_disfluency_drop",
                    evidence_quote=f"Disfluency -{h.disfluency_drop:.1f}/100w vs Q0",
                    explanation="Sharp drop in natural spoken hesitations relative to candidate's own intro baseline.",
                )
            )

        if h.cadence_uniformity_cv < 0.45 and h.word_count >= 60:
            findings.append(
                ProctoringFinding(
                    name="uniform_teleprompter_cadence",
                    evidence_quote=f"Pause CV={h.cadence_uniformity_cv:.2f}",
                    explanation="Abnormally uniform inter-word pause variance consistent with reading text aloud.",
                )
            )

        for ev in vision_links[:2]:
            findings.append(
                ProctoringFinding(
                    name="concurrent_vision_anomaly",
                    evidence_quote=f"[{ev.timestamp_offset}] {ev.category}",
                    explanation=ev.signal[:180],
                )
            )

        return findings

    @staticmethod
    def _compute_speech_naturalness(
        feats: Dict[str, Any],
        h: HeuristicScores,
        baseline: CandidateBaseline,
    ) -> float:
        """Measures genuine spoken naturalness from prosody & disfluency rather than inverting suspicion."""
        nat = 100.0
        nat -= min(h.disfluency_drop * 6.5, 25.0)
        nat -= min(len(h.ai_tell_matches) * 7.0, 25.0)
        if not feats.get("timing_estimated", True):
            nat -= min(max(0.0, abs(h.wpm_robust_zscore) - 1.5) * 8.0, 20.0)
        if h.cadence_uniformity_cv < 0.50:
            nat -= min((0.50 - h.cadence_uniformity_cv) * 80.0, 20.0)
        if feats.get("structure_marker_per_100w", 0.0) > 2.0:
            nat -= 10.0
        return round(max(0.0, min(100.0, nat)), 1)

    @staticmethod
    def _compute_vision_fusion_boost(vision_summary: Dict[str, Any]) -> Tuple[float, float]:
        if vision_summary.get("data_status") != "OK":
            return (0.0, 0.0)

        phone_n = int(vision_summary.get("phone_events_count") or 0)
        multi_n = int(vision_summary.get("multi_face_events_count") or 0)
        liveness_n = int(vision_summary.get("liveness_risk_events_count") or 0)
        teleprompter_n = int(vision_summary.get("teleprompter_scan_events_count") or 0)
        gaze_n = int(vision_summary.get("gaze_deviation_events_count") or 0)

        capped_boost = compute_vision_boost(
            phone_count=phone_n,
            multi_face_count=multi_n,
            liveness_count=liveness_n,
            teleprompter_count=teleprompter_n,
            gaze_count=gaze_n,
        )
        visual_attention = vision_summary.get("visual_attention_score")
        vision_risk_index = round(100.0 - float(visual_attention), 1) if visual_attention is not None else 0.0
        return (capped_boost, max(0.0, min(100.0, vision_risk_index)))

    @staticmethod
    def _build_risk_breakdown(
        legacy_features: List[Dict[str, Any]],
        stage1_scores: List[HeuristicScores],
        avg_nat_score: float,
        vision_risk_index: float,
    ) -> Dict[str, float]:
        n = max(len(stage1_scores), 1)
        tell_score = min(100.0, round((sum(len(h.ai_tell_matches) for h in stage1_scores) / n) * 30.0, 1))
        mirror_score = min(
            100.0,
            round((sum(f.get("mirror_score", 0.0) for f in legacy_features) / n) * 100.0, 1),
        )
        disfluency_deficit = min(
            100.0,
            round((sum(h.disfluency_drop for h in stage1_scores) / n) * 22.0, 1),
        )
        return {
            "ai_tell_patterns": tell_score,
            "speech_naturalness": avg_nat_score,
            "disfluency_deficit": disfluency_deficit,
            "question_mirroring": mirror_score,
            "vision_anomaly_risk": vision_risk_index,
        }

    @staticmethod
    def _synthesize_pattern_findings(
        per_answer: List[AnswerProctoring],
        stage1_scores: List[HeuristicScores],
        turn_injection_hits: List[List[str]],
        vision_summary: Dict[str, Any],
        llm_count: int,
        advisory_probes: List[str],
    ) -> List[str]:
        notes: List[str] = []

        if any(turn_injection_hits):
            notes.append(
                "Security Alert: Candidate transcript contained spoken instructions attempting to override automated scoring."
            )

        high_turns = [a.question_index + 1 for a in per_answer if a.risk_band == "high"]
        if high_turns:
            notes.append(
                f"Elevated behavioral anomaly signals concentrated on turn(s): {', '.join(f'Q{t}' for t in high_turns)}."
            )

        if vision_summary.get("data_status") == "OK":
            if (vision_summary.get("phone_events_count") or 0) > 0:
                notes.append(
                    f"Visual telemetry recorded {vision_summary['phone_events_count']} secondary device event(s)."
                )
            if (vision_summary.get("multi_face_events_count") or 0) > 0:
                notes.append(
                    f"Visual telemetry recorded {vision_summary['multi_face_events_count']} multi-face occurrence(s)."
                )
            if (vision_summary.get("liveness_risk_events_count") or 0) > 0:
                notes.append("Anti-spoof monitor flagged sustained liveness or frozen-feed anomalies.")
            if (vision_summary.get("teleprompter_scan_events_count") or 0) > 0:
                notes.append("Gaze saccade analyzer detected rhythmic horizontal reading patterns.")
        else:
            notes.append("No webcam telemetry ingested; evaluation is based solely on audio/transcript signals.")

        notes.append(
            f"Two-stage gating evaluated {len(per_answer)} turn(s) deterministically and escalated {llm_count} to LLM reasoning."
        )

        # Surface up to 2 sharpest live follow-up probes for the human interviewer
        notes.extend(advisory_probes[:2])
        return notes

    @staticmethod
    def _build_recommendation(
        overall_band: Literal["low", "medium", "high"],
        overall_risk: float,
        vision_summary: Dict[str, Any],
        turn_injection_hits: List[List[str]],
    ) -> str:
        if any(turn_injection_hits):
            return (
                "Escalate for human integrity review: spoken evaluator-manipulation phrasing detected; "
                "verify candidate ownership via live whiteboard or follow-up deep-dive."
            )
        if overall_band == "high":
            return (
                f"Elevated advisory risk ({overall_risk:.0f}/100): conduct a brief live follow-up "
                "probing specific implementation trade-offs and edge cases from flagged turns."
            )
        if overall_band == "medium":
            return (
                f"Moderate advisory signals ({overall_risk:.0f}/100): review highlighted timestamps "
                "and confirm practical depth during the next technical round."
            )
        if vision_summary.get("data_status") != "OK":
            return "Low transcript risk; note that webcam telemetry was not available for this session."
        return "Low advisory risk: responses and visual telemetry are consistent with authentic live construction."

    @staticmethod
    def _risk_band(risk_0_100: float) -> Literal["low", "medium", "high"]:
        if risk_0_100 >= 65.0:
            return "high"
        if risk_0_100 >= 35.0:
            return "medium"
        return "low"

    def _empty_report(self, reason: str, elapsed_ms: int) -> ProctoringReport:
        return ProctoringReport(
            overall_score=0.0,
            overall_risk_score=0.0,
            ai_reliance_percent=0.0,
            speech_naturalness_score=100.0,
            risk_band="low",
            share_high_risk=0.0,
            risk_factors_breakdown={
                "ai_tell_patterns": 0.0,
                "speech_naturalness": 100.0,
                "disfluency_deficit": 0.0,
                "question_mirroring": 0.0,
                "vision_anomaly_risk": 0.0,
            },
            vision_summary={"data_status": "NO_TELEMETRY", "anomaly_events": []},
            vision_events=[],
            pattern_findings=[reason],
            recommendation="No scorable candidate responses were captured.",
            caveat=ADVISORY_CAVEAT,
            per_answer=[],
            baseline_summary={},
            model_version=PROCTORING_MODEL_VERSION,
            status="failed",
            elapsed_ms=elapsed_ms,
        )


_proctoring_service: Optional[ProctoringService] = None


def get_proctoring_service() -> ProctoringService:
    global _proctoring_service
    if _proctoring_service is None:
        _proctoring_service = ProctoringService()
    return _proctoring_service