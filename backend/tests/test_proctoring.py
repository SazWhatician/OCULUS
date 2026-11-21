import asyncio
from app.services.proctoring_service import get_proctoring_service

async def test_proctoring_ai_reliance():
    svc = get_proctoring_service()
    suspicious_transcript = {
        "qa_pairs": [
            {
                "category": "introduction",
                "question": "Tell me about yourself.",
                "answer": "Hi, um, I am a software dev working with Python."
            },
            {
                "category": "technical",
                "question": "How do you optimize a database query?",
                "answer": "It is important to note that optimizing a database query requires a multifaceted approach. Firstly, one must delve into the execution plan. Secondly, there are several indexing strategies, such as composite indexes. Furthermore, in today's fast-paced world, Redis plays a pivotal role."
            }
        ]
    }
    report = await svc.score_session(suspicious_transcript, job_title="Backend Engineer")
    print("=== Test Proctoring AI Reliance ===")
    print(f"Overall Risk Score: {report.overall_risk_score}")
    print(f"AI Reliance Percent: {report.ai_reliance_percent}%")
    print(f"Risk Band: {report.risk_band}")
    print(f"Risk Factors Breakdown: {report.risk_factors_breakdown}")
    assert report.overall_risk_score > 0
    assert report.ai_reliance_percent > 0
    assert "ai_tell_patterns" in report.risk_factors_breakdown
    print("[PASS] Proctoring AI Reliance Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_proctoring_ai_reliance())
