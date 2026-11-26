from dataclasses import dataclass
from datetime import datetime, timezone
import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from app.schemas.contracts import (
    CandidateBaseline,
    ConfidenceBand,
    HeuristicScores,
    IntegrityReport,
    LLMJudgeVerdict,
    QAPairEvaluation,
    RiskBand,
    VisionAnomalyEvent,
)

# ---------------------------------------------------------------------------
# Forensic Aggregation Constants
# ---------------------------------------------------------------------------
MIN_WORDS_FOR_SUBSTANTIVE_AGG = 25
MAX_TURN_WORD_WEIGHT = 250
NLSE_TEMPERATURE_BETA = 0.08          # Controls tail sensitivity in Boltzmann anti-dilution pooling

VISION_BOOST_CAP = 25.0
VISION_BURST_WINDOW_MS = 10_000       # Same-type alerts within 10s belong to one continuous episode
VISION_BURST_DECAY = 0.45             # Overlapping burst frames add damped weight instead of 1.0
SOFT_VISION_ONLY_CAP = 52.0           # Pure gaze/head-pose variance alone cannot reach ELEVATED

# Per-episode Noisy-OR hazard probabilities (used for vision-only & monotonic floor)
VISION_EPISODE_HAZARDS: Dict[str, float] = {
    "phone": 0.56,
    "liveness": 0.50,
    "multi_face": 0.45,
    "teleprompter": 0.28,
    "occlusion": 0.24,
    "gaze": 0.13,
    "head_pose": 0.10,
}

# Smooth saturating boost curves: (asymptotic_cap, decay_rate, free_grace_allowance)
VISION_BOOST_CURVES: Dict[str, Tuple[float, float, float]] = {
    "phone": (22.0, 1.02, 0.0),
    "multi_face": (16.0, 0.98, 0.0),
    "liveness": (20.0, 0.92, 0.0),
    "teleprompter": (14.0, 0.56, 0.0),
    "occlusion": (10.0, 0.52, 0.0),
    "gaze": (8.0, 0.45, 1.0),         # First isolated gaze shift adds +0.0 (neurodiversity shield)
    "head_pose": (5.0, 0.45, 1.0),    # First isolated posture shift adds +0.0
}

TELEPROMPTER_MARKERS = (
    "teleprompter",
    "eye-scanning",
    "peek pattern",
    "peek cycle",
    "split-angle",
)
LIVENESS_MARKERS = (
    "liveness",
    "spoof",
    "frozen",
    "substitution",
    "identity",
    "virtual camera",
)
OCCLUSION_MARKERS = (
    "occlusion",
    "absent",
    "disabled",
    "covered",
)


@dataclass(frozen=True)
class VisionEvidenceSummary:
    """Temporally de-clustered effective counts and raw counts across all 7 vision channels."""
    effective: Dict[str, float]
    raw_counts: Dict[str, int]
    total_events: int


def _classify_risk_band(score: float) -> RiskBand:
    if score < 35.0:
        return RiskBand.LOW
    if score < 65.0:
        return RiskBand.MODERATE
    return RiskBand.ELEVATED


def _normalized_log_sum_exp(
    risks: Sequence[float],
    weights: Sequence[float],
    beta: float = NLSE_TEMPERATURE_BETA,
) -> float:
    """
    Numerically stable weighted Boltzmann / Normalized Log-Sum-Exp aggregator.
    Properties:
      - Strictly idempotent: NLSE(c, c, ..., c) == c (zero clean-candidate inflation)
      - Strictly monotonic & smooth: resists dilution without step-function cliffs
    """
    if not risks:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0.0:
        return float(statistics.fmean(risks))

    m = max(risks)
    weighted_exp = sum(w * math.exp(beta * (r - m)) for r, w in zip(risks, weights))
    ratio = max(weighted_exp / total_w, 1e-12)
    return m + (math.log(ratio) / beta)


def _classify_vision_event(ev: VisionAnomalyEvent) -> str:
    """Maps both contracts.VisionAnomalyEvent and vision_proctoring events into 7 canonical buckets."""
    raw_type = getattr(ev, "anomaly_type", None)
    t_name = getattr(raw_type, "value", str(raw_type)).strip().upper() if raw_type is not None else ""

    meta_text = " ".join(
        str(getattr(ev, attr, "") or "")
        for attr in ("category", "signal", "vlm_verification_note")
    ).lower()
    duration_ms = int(getattr(ev, "duration_ms", 0) or 0)

    if t_name in ("DEVICE_DETECTED", "PHONE_DETECTED") or "phone" in meta_text or "tablet" in meta_text:
        return "phone"
    if t_name == "MULTI_FACE" or "multi-face" in meta_text:
        return "multi_face"
    if t_name == "LIVENESS_SPOOF" or any(tok in meta_text for tok in LIVENESS_MARKERS):
        return "liveness"
    if any(tok in meta_text for tok in TELEPROMPTER_MARKERS):
        return "teleprompter"
    if t_name == "CAMERA_OCCLUSION" or any(tok in meta_text for tok in OCCLUSION_MARKERS):
        return "occlusion"
    if t_name == "HEAD_PITCH_DOWN" or "head pose" in meta_text:
        return "head_pose"
    # Sustained >= 7.5s gaze anomalies emitted by VisionWorker carry teleprompter/peek-loop duration
    if t_name == "GAZE_DEVIATION" and duration_ms >= 7_500:
        return "teleprompter"
    return "gaze"


def _summarize_vision_timeline(
    vision_timeline: Sequence[VisionAnomalyEvent],
) -> VisionEvidenceSummary:
    """
    Aggregates timeline anomalies with confidence scaling, duration scaling,
    and 10-second same-category burst de-clustering.
    """
    keys = ("phone", "multi_face", "liveness", "teleprompter", "occlusion", "gaze", "head_pose")
    effective: Dict[str, float] = {k: 0.0 for k in keys}
    raw_counts: Dict[str, int] = {k: 0 for k in keys}

    if not vision_timeline:
        return VisionEvidenceSummary(effective=effective, raw_counts=raw_counts, total_events=0)

    ordered = sorted(
        vision_timeline,
        key=lambda e: int(getattr(e, "timestamp_ms", 0) or 0),
    )
    last_ts_by_bucket: Dict[str, int] = {}

    for ev in ordered:
        bucket = _classify_vision_event(ev)
        raw_counts[bucket] += 1

        raw_conf = getattr(ev, "confidence", 0.85)
        conf = float(raw_conf) if isinstance(raw_conf, (int, float)) and math.isfinite(raw_conf) else 0.85
        conf_w = max(0.45, min(1.0, conf))

        dur_ms = max(0, int(getattr(ev, "duration_ms", 0) or 0))
        if bucket in ("gaze", "head_pose"):
            dur_w = min(1.35, max(0.65, (dur_ms or 3_500) / 4_000.0))
        elif bucket == "teleprompter":
            dur_w = min(1.25, max(0.85, (dur_ms or 8_000) / 8_000.0))
        else:
            dur_w = 1.0

        ts_ms = int(getattr(ev, "timestamp_ms", 0) or 0)
        prev_ts = last_ts_by_bucket.get(bucket)
        burst_w = VISION_BURST_DECAY if (prev_ts is not None and (ts_ms - prev_ts) <= VISION_BURST_WINDOW_MS) else 1.0
        last_ts_by_bucket[bucket] = ts_ms

        effective[bucket] += conf_w * dur_w * burst_w

    return VisionEvidenceSummary(
        effective=effective,
        raw_counts=raw_counts,
        total_events=len(ordered),
    )


def calculate_qa_combined_score(
    heuristics: HeuristicScores,
    llm_verdict: Optional[LLMJudgeVerdict] = None,
) -> float:
    """
    Precision-weighted per-turn fusion between Stage 1 heuristics and Stage 2 LLM verdict:
      - Scales LLM authority by epistemic confidence band & verified verbatim quote count
      - Dampens uncorroborated LLM spikes (>40pt gap with zero quotes & zero vision flags)
      - Protects hard Stage 1 security/vision proof from being washed out by a lenient LLM
    """
    h_risk = max(0.0, min(100.0, float(heuristics.deterministic_risk)))
    if llm_verdict is None:
        return round(h_risk, 1)

    l_risk = max(0.0, min(100.0, float(llm_verdict.suspicion_score)))
    verified_quotes = len(llm_verdict.verbatim_evidence_quotes or [])
    conf = llm_verdict.confidence_band

    if conf == ConfidenceBand.LOW:
        w_llm = 0.42
    elif conf == ConfidenceBand.HIGH:
        w_llm = 0.76 if verified_quotes >= 1 else 0.68
    else:
        w_llm = 0.68 if verified_quotes >= 1 else 0.60

    # Epistemic disagreement guard: dampen unverified LLM spike when Stage 1 & vision are quiet
    if (l_risk - h_risk) > 40.0 and verified_quotes == 0 and heuristics.concurrent_vision_flags == 0:
        w_llm *= 0.75

    score = ((1.0 - w_llm) * h_risk) + (w_llm * l_risk)

    # Cross-modal corroboration lift when Stage 1 + Stage 2 + independent evidence align
    if (
        h_risk >= 45.0
        and l_risk >= 65.0
        and (heuristics.concurrent_vision_flags > 0 or verified_quotes >= 2)
    ):
        score += 4.0

    # Hard Stage 1 safety floor (prompt injection or high-severity concurrent vision proof)
    reasons_lower = " ".join(heuristics.trigger_reasons or []).lower()
    has_hard_stage1 = (
        "security alert" in reasons_lower
        or "high-severity concurrent vision" in reasons_lower
    )
    if has_hard_stage1:
        score = max(score, 0.85 * h_risk)

    return round(min(max(score, 0.0), 100.0), 1)


def compute_vision_boost(
    phone_count: float = 0.0,
    multi_face_count: float = 0.0,
    liveness_count: float = 0.0,
    teleprompter_count: float = 0.0,
    gaze_count: float = 0.0,
    occlusion_count: float = 0.0,
    head_pose_count: float = 0.0,
) -> float:
    """
    Smooth saturating multimodal vision risk boost (capped at +25.0).
    Accepts either integer raw counts or de-clustered effective weights.
    """
    counts = {
        "phone": max(0.0, float(phone_count)),
        "multi_face": max(0.0, float(multi_face_count)),
        "liveness": max(0.0, float(liveness_count)),
        "teleprompter": max(0.0, float(teleprompter_count)),
        "occlusion": max(0.0, float(occlusion_count)),
        "gaze": max(0.0, float(gaze_count)),
        "head_pose": max(0.0, float(head_pose_count)),
    }

    boost = 0.0
    for key, n_val in counts.items():
        cap, rate, grace = VISION_BOOST_CURVES[key]
        excess = max(0.0, n_val - grace)
        if excess > 0.0:
            boost += cap * (1.0 - math.exp(-rate * excess))

    return round(min(boost, VISION_BOOST_CAP), 1)


def compute_standalone_vision_risk(summary: VisionEvidenceSummary) -> float:
    """
    Computes standalone session risk [0.0..100.0] from visual telemetry via Noisy-OR hazard fusion.
    Guarantees:
      - Video-only / pre-Q&A sessions with severe visual anomalies are scored accurately (never 0.0).
      - Pure soft-behavioral gaze/head-pose variance is fairness-capped at 52.0 (MODERATE max).
    """
    if summary.total_events == 0:
        return 0.0

    hard_survival = 1.0
    for key in ("phone", "liveness", "multi_face", "teleprompter", "occlusion"):
        n_eff = max(0.0, summary.effective.get(key, 0.0))
        if n_eff > 0.0:
            hard_survival *= (1.0 - VISION_EPISODE_HAZARDS[key]) ** n_eff
    hard_prob = 1.0 - hard_survival

    soft_survival = 1.0
    for key in ("gaze", "head_pose"):
        n_eff = max(0.0, summary.effective.get(key, 0.0))
        if n_eff > 0.0:
            soft_survival *= (1.0 - VISION_EPISODE_HAZARDS[key]) ** n_eff
    soft_prob = min(1.0 - soft_survival, SOFT_VISION_ONLY_CAP / 100.0)

    combined_prob = 1.0 - ((1.0 - hard_prob) * (1.0 - soft_prob))
    return round(min(max(combined_prob * 100.0, 0.0), 100.0), 1)


def compute_anti_dilution_session_risk(
    turn_metrics: Sequence[Tuple[float, int, bool, bool]],
    vision_boost: float = 0.0,
) -> float:
    """
    Shared anti-dilution session risk calculation used by both live worker and batch reports.
    turn_metrics items: (risk_0_100, word_count, is_intro, force_include)
    """
    safe_boost = max(0.0, float(vision_boost))
    if not turn_metrics:
        return round(min(safe_boost, 100.0), 1)

    substantive = [
        (max(0.0, min(100.0, float(risk))), int(wc))
        for risk, wc, is_intro, force_include in turn_metrics
        if force_include or (not is_intro and wc >= MIN_WORDS_FOR_SUBSTANTIVE_AGG)
    ]
    if not substantive:
        substantive = [
            (max(0.0, min(100.0, float(risk))), int(wc))
            for risk, wc, _, _ in turn_metrics
        ]

    risks = [r for r, _ in substantive]
    weights = [float(min(max(wc, 1), MAX_TURN_WORD_WEIGHT)) for _, wc in substantive]
    total_weight = sum(weights)

    weighted_mean = sum(r * w for r, w in zip(risks, weights)) / max(total_weight, 1.0)
    nlse_score = _normalized_log_sum_exp(risks, weights, beta=NLSE_TEMPERATURE_BETA)

    risks_desc = sorted(risks, reverse=True)
    top_2_mean = statistics.fmean(risks_desc[:2])
    max_risk = risks_desc[0]

    # Convex idempotent blend: clean uniform turns stay exact; concise high-risk turns resist dilution
    text_session_risk = (
        (0.25 * weighted_mean)
        + (0.30 * nlse_score)
        + (0.25 * top_2_mean)
        + (0.20 * max_risk)
    )

    # Chronic multi-turn persistence lift (activates only when >= 3 distinct turns independently exceed 60)
    elevated_turns = sum(1 for r in risks if r >= 60.0)
    if elevated_turns >= 3:
        text_session_risk += min(6.0, 3.0 * (elevated_turns - 2))

    return round(min(max(text_session_risk + safe_boost, 0.0), 100.0), 1)


def generate_integrity_report(
    session_id: str,
    candidate_name: str,
    job_title: str,
    qa_evaluations: List[QAPairEvaluation],
    vision_timeline: List[VisionAnomalyEvent],
    baseline: Optional[CandidateBaseline] = None,
) -> IntegrityReport:
    """
    Aggregates overall session integrity score using:
      1. Full 7-channel burst-declustered vision telemetry (including teleprompter, gaze, occlusion)
      2. Boltzmann anti-dilution pooling across substantive Q&A turns
      3. Monotonic cross-modal envelope so empty-Q&A or short-Q&A sessions never mask visual violations
    """
    vision_summary = _summarize_vision_timeline(vision_timeline)
    standalone_vision_risk = compute_standalone_vision_risk(vision_summary)

    if not qa_evaluations:
        final_score = standalone_vision_risk
        return IntegrityReport(
            session_id=session_id,
            candidate_name=candidate_name,
            job_title=job_title,
            overall_risk_score=final_score,
            risk_band=_classify_risk_band(final_score),
            baseline=baseline,
            qa_evaluations=qa_evaluations,
            vision_timeline=vision_timeline,
            generated_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    only_turn = len(qa_evaluations) == 1
    turn_metrics: List[Tuple[float, int, bool, bool]] = []
    for qa in qa_evaluations:
        is_intro = (qa.qa_index == 0) and not only_turn
        has_security_flag = any(
            "security alert" in r.lower() for r in (qa.heuristics.trigger_reasons or [])
        ) or (
            qa.llm_verdict is not None
            and any("security alert" in s.lower() for s in (qa.llm_verdict.reasoning_signals or []))
        )
        has_vision_flag = qa.heuristics.concurrent_vision_flags > 0
        turn_metrics.append(
            (
                qa.combined_risk_score,
                qa.heuristics.word_count,
                is_intro,
                has_security_flag or has_vision_flag,
            )
        )

    eff = vision_summary.effective
    vision_boost = compute_vision_boost(
        phone_count=eff["phone"],
        multi_face_count=eff["multi_face"],
        liveness_count=eff["liveness"],
        teleprompter_count=eff["teleprompter"],
        gaze_count=eff["gaze"],
        occlusion_count=eff["occlusion"],
        head_pose_count=eff["head_pose"],
    )

    text_plus_boost = compute_anti_dilution_session_risk(
        turn_metrics,
        vision_boost=vision_boost,
    )
    # Monotonic cross-modal envelope: adding a clean greeting turn can never hide confirmed vision violations
    final_score = round(min(100.0, max(text_plus_boost, standalone_vision_risk)), 1)

    return IntegrityReport(
        session_id=session_id,
        candidate_name=candidate_name,
        job_title=job_title,
        overall_risk_score=final_score,
        risk_band=_classify_risk_band(final_score),
        baseline=baseline,
        qa_evaluations=qa_evaluations,
        vision_timeline=vision_timeline,
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
    )