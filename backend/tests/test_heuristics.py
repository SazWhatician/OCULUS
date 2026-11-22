from app.services.vocab_extractor import extract_custom_vocab
from app.services.heuristics import (
    compute_candidate_baseline,
    compute_deterministic_heuristics,
)
from app.services.aggregator import (
    calculate_qa_combined_score,
    generate_integrity_report,
)
from app.schemas.contracts import (
    LLMJudgeVerdict,
    QAPairEvaluation,
    RiskBand,
    VisionAnomalyEvent,
    VisionAnomalyType,
)


def test_vocab_extractor():
    vocab = extract_custom_vocab(
        job_title="Senior Python & Distributed Systems Engineer",
        job_description="Experience with Kafka, Redis, Docker, gRPC, and Microservices required.",
    )
    assert "Python" in vocab
    assert "Kafka" in vocab
    assert "Redis" in vocab
    assert "Docker" in vocab
    assert "gRPC" in vocab


def test_stage0_baseline_calculation():
    q0_text = (
        "So um, I have been working as a full stack engineer for about four years now. "
        "Basically, I started with Python and React, and uh, you know, I love building distributed backends. "
        "Actually, in my last role, we deployed FastAPI services on Kubernetes."
    )
    baseline = compute_candidate_baseline(
        session_id="s1",
        q0_text=q0_text,
        duration_ms=25000,
    )
    assert baseline.baseline_wpm > 80.0
    assert baseline.baseline_disfluency_rate > 3.0  # Had um, basically, uh, you know, actually
    assert baseline.sample_word_count > 30


def test_heuristics_normal_esl_candidate_skips_llm():
    # Candidate speaks naturally with personal pronouns and some hesitation
    answer = (
        "Yeah, in my previous project, we faced a similar database bottleneck with PostgreSQL. "
        "So, um, I looked into our slow queries and basically added composite indexes. "
        "It took us about two weeks, and like, our latency dropped by 40 percent."
    )
    baseline = compute_candidate_baseline(
        session_id="s1",
        q0_text="Hi, I am an engineer from India, basically worked on backend APIs.",
        duration_ms=8000,
    )
    scores = compute_deterministic_heuristics(
        answer_text=answer,
        duration_ms=20000,
        baseline=baseline,
    )
    assert scores.deterministic_risk < 40.0
    assert scores.trigger_llm_judge is False  # Gate correctly skips LLM


def test_heuristics_ai_recitation_triggers_llm():
    # Highly polished recitation devoid of personal pronouns with multiple AI tells
    ai_answer = (
        "It is crucial to note that optimizing distributed database queries requires a multifaceted approach. "
        "Firstly, one must delve into execution plans using EXPLAIN ANALYZE to identify slow table scans. "
        "Secondly, caching architectures play a pivotal role in decoupling read-heavy traffic. "
        "In summary, a comprehensive approach ensures optimal throughput and fosters synergy across services."
    )
    baseline = compute_candidate_baseline(
        session_id="s1",
        q0_text="Um, yeah, so I am a software engineer, you know, worked for 3 years.",
        duration_ms=10000,
    )
    scores = compute_deterministic_heuristics(
        answer_text=ai_answer,
        duration_ms=25000,
        baseline=baseline,
        ttfs_latency_ms=4500,
    )
    assert len(scores.ai_tell_matches) >= 3
    assert scores.deterministic_risk >= 45.0
    assert scores.trigger_llm_judge is True  # Pre-filter gate correctly opened


def test_aggregator_report_generation():
    h_normal = compute_deterministic_heuristics("Normal answer with good details.", 5000)
    h_suspicious = compute_deterministic_heuristics(
        "It is crucial to note that one must delve into multifaceted architectures in summary.",
        15000,
    )
    verdict = LLMJudgeVerdict(
        answer_id="ans_1",
        suspicion_score=85.0,
        reasoning_signals=["Question mirroring", "Textbook structure"],
    )

    qa1 = QAPairEvaluation(
        qa_index=1,
        question_text="Tell me about architecture.",
        answer_text="Normal answer with good details.",
        heuristics=h_normal,
        combined_risk_score=calculate_qa_combined_score(h_normal),
    )
    qa2 = QAPairEvaluation(
        qa_index=2,
        question_text="How do you scale?",
        answer_text="It is crucial to note that one must delve into multifaceted architectures in summary.",
        heuristics=h_suspicious,
        llm_verdict=verdict,
        combined_risk_score=calculate_qa_combined_score(h_suspicious, verdict),
    )

    device_event = VisionAnomalyEvent(
        event_id="v1",
        session_id="s1",
        timestamp_ms=12000,
        anomaly_type=VisionAnomalyType.DEVICE_DETECTED,
        duration_ms=4000,
        confidence=0.92,
    )

    report = generate_integrity_report(
        session_id="s1",
        candidate_name="John Doe",
        job_title="Software Architect",
        qa_evaluations=[qa1, qa2],
        vision_timeline=[device_event],
    )

    assert report.overall_risk_score > 50.0
    assert report.risk_band in [RiskBand.MODERATE, RiskBand.ELEVATED]
    assert len(report.qa_evaluations) == 2
    assert len(report.vision_timeline) == 1
