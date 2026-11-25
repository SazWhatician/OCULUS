from datetime import datetime, timezone
import statistics
from typing import List, Optional, Sequence, Tuple
from app.schemas.contracts import (
    CandidateBaseline,
    ConfidenceBand,
    HeuristicScores,
    LLMJudgeVerdict,
    QAPairEvaluation,
    IntegrityReport,
    RiskBand,
    VisionAnomalyEvent,
    VisionAnomalyType,
)

MIN_WORDS_FOR_SUBSTANTIVE_AGG = 25
MAX_TURN_WORD_WEIGHT = 250


def calculate_qa_combined_score(
    heuristics: HeuristicScores,
    llm_verdict: Optional[LLMJudgeVerdict] = None,
) -> float:
    """
    Computes per-answer combined risk score, dampening Stage 2 weight when
    LLM confidence is LOW.
    """
    if llm_verdict is not None:
        if llm_verdict.confidence_band == ConfidenceBand.LOW:
            score = (0.55 * heuristics.deterministic_risk) + (0.45 * llm_verdict.suspicion_score)
        else:
            score = (0.35 * heuristics.deterministic_risk) + (0.65 * llm_verdict.suspicion_score)
    else:
        score = heuristics.deterministic_risk

    return round(min(max(score, 0.0), 100.0), 1)


def compute_vision_boost(
    phone_count: int = 0,
    multi_face_count: int = 0,
    liveness_count: int = 0,
    teleprompter_count: int = 0,
    gaze_count: int = 0,
) -> float:
    """Bounded multimodal vision risk boost (capped at +25.0)."""
    boost = (
        min(phone_count * 14.0, 20.0)
        + min(multi_face_count * 10.0, 15.0)
        + min(liveness_count * 12.0, 18.0)
        + min(teleprompter_count * 6.0, 12.0)
        + min(max(0, gaze_count - 1) * 2.0, 6.0)
    )
    return round(min(boost, 25.0), 1)


def compute_anti_dilution_session_risk(
    turn_metrics: Sequence[Tuple[float, int, bool, bool]],
    vision_boost: float = 0.0,
) -> float:
    """
    Shared anti-dilution session risk calculation used by both live worker and batch reports.
    turn_metrics items: (risk_0_100, word_count, is_intro, force_include)
    """
    if not turn_metrics:
        return 0.0

    substantive = [
        (risk, wc)
        for risk, wc, is_intro, force_include in turn_metrics
        if force_include or (not is_intro and wc >= MIN_WORDS_FOR_SUBSTANTIVE_AGG)
    ]
    if not substantive:
        substantive = [(risk, wc) for risk, wc, _, _ in turn_metrics]

    weights = [min(max(wc, 1), MAX_TURN_WORD_WEIGHT) for _, wc in substantive]
    total_weight = sum(weights)
    weighted_mean = sum(r * w for (r, _), w in zip(substantive, weights)) / max(total_weight, 1)

    risks_desc = sorted((r for r, _ in substantive), reverse=True)
    top_2_mean = statistics.fmean(risks_desc[:2])
    max_risk = risks_desc[0]

    text_session_risk = (0.35 * weighted_mean) + (0.35 * top_2_mean) + (0.30 * max_risk)
    return round(min(max(text_session_risk + max(0.0, vision_boost), 0.0), 100.0), 1)


def generate_integrity_report(
    session_id: str,
    candidate_name: str,
    job_title: str,
    qa_evaluations: List[QAPairEvaluation],
    vision_timeline: List[VisionAnomalyEvent],
    baseline: Optional[CandidateBaseline] = None,
) -> IntegrityReport:
    """
    Aggregates overall session score using anti-dilution pooling across substantive
    Q&A pairs plus bounded multimodal vision boost.
    """
    if not qa_evaluations:
        return IntegrityReport(
            session_id=session_id,
            candidate_name=candidate_name,
            job_title=job_title,
            overall_risk_score=0.0,
            risk_band=RiskBand.LOW,
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
            (qa.combined_risk_score, qa.heuristics.word_count, is_intro, has_security_flag or has_vision_flag)
        )

    phone_n = 0
    multi_n = 0
    liveness_n = 0
    for v in vision_timeline:
        t_name = getattr(v.anomaly_type, "value", str(v.anomaly_type)).upper()
        if t_name in ("DEVICE_DETECTED", "PHONE_DETECTED"):
            phone_n += 1
        elif t_name == "MULTI_FACE":
            multi_n += 1
        elif t_name == "LIVENESS_SPOOF":
            liveness_n += 1

    vision_boost = compute_vision_boost(
        phone_count=phone_n,
        multi_face_count=multi_n,
        liveness_count=liveness_n,
    )
    final_score = compute_anti_dilution_session_risk(turn_metrics, vision_boost=vision_boost)

    if final_score < 35.0:
        risk_band = RiskBand.LOW
    elif final_score < 65.0:
        risk_band = RiskBand.MODERATE
    else:
        risk_band = RiskBand.ELEVATED

    return IntegrityReport(
        session_id=session_id,
        candidate_name=candidate_name,
        job_title=job_title,
        overall_risk_score=final_score,
        risk_band=risk_band,
        baseline=baseline,
        qa_evaluations=qa_evaluations,
        vision_timeline=vision_timeline,
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
    )
