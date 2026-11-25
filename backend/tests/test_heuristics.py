from app.services.vocab_extractor import extract_custom_vocab
from app.services.heuristics import (
    compute_candidate_baseline,
    compute_deterministic_heuristics,
)
from app.services.aggregator import (
    calculate_qa_combined_score,
    generate_integrity_report,
)
from app.services.vision_proctoring import (
    VisionAnomalyEvent as ServiceVisionEvent,
    VisionProctoringService,
    VisionTelemetryFrame,
)
from app.schemas.contracts import (
    CandidateBaseline,
    ConfidenceBand,
    HeuristicScores,
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


def test_vision_service_event_heuristics_interop():
    svc = VisionProctoringService()
    emitted = []
    for i in range(3):
        emitted.extend(
            svc.ingest_frame_all(
                VisionTelemetryFrame(
                    session_id="s_interop",
                    timestamp_ms=1_000 + i * 700,
                    phone_detected=True,
                )
            )
        )
    assert emitted
    assert isinstance(emitted[0], ServiceVisionEvent)
    assert emitted[0].anomaly_type == VisionAnomalyType.DEVICE_DETECTED

    scores = compute_deterministic_heuristics(
        answer_text="I implemented the service using standard queues.",
        duration_ms=4_000,
        concurrent_vision_events=emitted,
    )
    assert scores.concurrent_vision_flags >= 1
    assert scores.deterministic_risk > 15.0
    assert any("DEVICE_DETECTED" in reason for reason in scores.trigger_reasons)
    assert not any("()" in reason for reason in scores.trigger_reasons)


def test_aggregator_anti_dilution_parity():
    def _mk_qa(idx: int, words: int, score: float) -> QAPairEvaluation:
        return QAPairEvaluation(
            qa_index=idx,
            question_text=f"Q{idx}",
            answer_text="word " * words,
            heuristics=HeuristicScores(
                word_count=words,
                wpm=130.0,
                wpm_robust_zscore=0.0,
                disfluency_rate=2.0,
                disfluency_drop=0.0,
                deterministic_risk=score,
            ),
            combined_risk_score=score,
        )

    # 3 verbose honest turns (400 words total, risk=15) + 1 concise AI turn (40 words, risk=85)
    diluted_report = generate_integrity_report(
        session_id="s_dilute",
        candidate_name="Test Candidate",
        job_title="Backend Engineer",
        qa_evaluations=[
            _mk_qa(1, 133, 15.0),
            _mk_qa(2, 133, 15.0),
            _mk_qa(3, 134, 15.0),
            _mk_qa(4, 40, 85.0),
        ],
        vision_timeline=[],
    )
    assert diluted_report.overall_risk_score >= 48.0
    assert diluted_report.risk_band == RiskBand.MODERATE

    # 4 honest turns (risk=15 each) stay at 15.0 LOW without false-positive inflation
    clean_report = generate_integrity_report(
        session_id="s_clean",
        candidate_name="Test Candidate",
        job_title="Backend Engineer",
        qa_evaluations=[_mk_qa(i, 80, 15.0) for i in range(1, 5)],
        vision_timeline=[],
    )
    assert clean_report.overall_risk_score == 15.0
    assert clean_report.risk_band == RiskBand.LOW


def test_heuristics_length_confidence_scaling_and_fairness():
    # Case 2: Concise 38-word AI recitation with 3 tells (1 Tier A) & 0 fillers
    concise_ai = (
        "It's crucial to note that one must delve into distributed consensus carefully "
        "when designing replicated logs. Moreover, partition tolerance guarantees consistent state transitions "
        "across leader failover boundaries while preserving durable transactional semantics across all three regional production clusters."
    )
    assert len(concise_ai.split()) == 38
    concise_scores = compute_deterministic_heuristics(
        answer_text=concise_ai,
        duration_ms=16_000,
    )
    assert concise_scores.deterministic_risk > 24.0
    assert any("Disfluency collapse" in r for r in concise_scores.trigger_reasons)
    assert concise_scores.trigger_llm_judge is True

    # Case 3: Articulate 45-word honest conceptual answer (0 tells, 0 fillers vs 3.0 baseline, 0 first-person)
    clean_conceptual = (
        "Consistent hashing maps cluster nodes and object keys onto the same logical hash ring "
        "using deterministic functions. Virtual nodes distribute storage partitions evenly across active workers "
        "so scaling out shifts only neighboring shard ranges rather than forcing a full cluster cache rebalance during traffic spikes."
    )
    assert len(clean_conceptual.split()) == 45
    baseline_3 = CandidateBaseline(
        session_id="s_fair",
        baseline_wpm=135.0,
        mad_wpm=18.0,
        baseline_disfluency_rate=3.0,
        baseline_pause_ratio=0.1,
        baseline_ttfs_ms=1200.0,
        sample_word_count=60,
    )
    fair_scores = compute_deterministic_heuristics(
        answer_text=clean_conceptual,
        duration_ms=20_000,
        baseline=baseline_3,
    )
    assert fair_scores.deterministic_risk < 20.0
    assert fair_scores.trigger_llm_judge is False

    # Case 4: Trivial 15-word clean short answer
    short_clean = "We used Redis Streams for async decoupling and Postgres for durable session integrity audit reports."
    assert len(short_clean.split()) == 15
    short_scores = compute_deterministic_heuristics(
        answer_text=short_clean,
        duration_ms=6_000,
    )
    assert short_scores.deterministic_risk < 10.0
    assert short_scores.trigger_llm_judge is False
