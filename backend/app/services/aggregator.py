from datetime import datetime, timezone
from typing import List, Optional
from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    LLMJudgeVerdict,
    QAPairEvaluation,
    IntegrityReport,
    RiskBand,
    VisionAnomalyEvent,
    VisionAnomalyType,
)


def calculate_qa_combined_score(
    heuristics: HeuristicScores,
    llm_verdict: Optional[LLMJudgeVerdict] = None,
) -> float:
    """
    Computes per-answer combined risk score:
    0.35 * R_det + 0.65 * R_llm if Stage 2 ran, otherwise R_det.
    """
    if llm_verdict is not None:
        score = 0.35 * heuristics.deterministic_risk + 0.65 * llm_verdict.suspicion_score
    else:
        score = heuristics.deterministic_risk

    return round(min(max(score, 0.0), 100.0), 1)


def generate_integrity_report(
    session_id: str,
    candidate_name: str,
    job_title: str,
    qa_evaluations: List[QAPairEvaluation],
    vision_timeline: List[VisionAnomalyEvent],
    baseline: Optional[CandidateBaseline] = None,
) -> IntegrityReport:
    """
    Aggregates overall session score from all evaluated Q&A pairs (qa_index >= 1)
    weighted by candidate word count, with capped vision anomaly adjustment.
    """
    evaluated_turns = [qa for qa in qa_evaluations if qa.qa_index >= 1]

    if not evaluated_turns:
        # Default empty/minimal session
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

    total_words = sum(qa.heuristics.word_count for qa in evaluated_turns)
    if total_words > 0:
        weighted_sum = sum(
            qa.combined_risk_score * qa.heuristics.word_count for qa in evaluated_turns
        )
        base_score = weighted_sum / total_words
    else:
        base_score = sum(qa.combined_risk_score for qa in evaluated_turns) / len(evaluated_turns)

    # Capped vision bonus (+15 max for confirmed device or multi-face)
    vision_bonus = 0.0
    for v in vision_timeline:
        if v.anomaly_type == VisionAnomalyType.DEVICE_DETECTED:
            vision_bonus = max(vision_bonus, 15.0)
        elif v.anomaly_type == VisionAnomalyType.MULTI_FACE:
            vision_bonus = max(vision_bonus, 10.0)

    final_score = min(max(base_score + vision_bonus, 0.0), 100.0)

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
        overall_risk_score=round(final_score, 1),
        risk_band=risk_band,
        baseline=baseline,
        qa_evaluations=qa_evaluations,
        vision_timeline=vision_timeline,
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
    )
