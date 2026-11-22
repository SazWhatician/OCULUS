import asyncio
from app.speech.enhancer import TranscriptEnhancer, TECHNICAL_TERMS_MAP
from app.speech.sarvam import SarvamSTTService

def test_entity_normalization():
    enhancer = TranscriptEnhancer()
    raw = "So, um, I built micro services using fast api, postgre sql, and kube netes with sarvam stt."
    enhanced = enhancer.enhance_text(raw, clean_fillers=True)
    assert "FastAPI" in enhanced
    assert "PostgreSQL" in enhanced
    assert "Kubernetes" in enhanced
    assert "Sarvam AI" in enhanced
    assert "um" not in enhanced.lower().split()

def test_utterance_stitching():
    enhancer = TranscriptEnhancer()
    blocks = [
        {"speaker": "Candidate", "text": "I used fast api", "timestamp_ms": 1000},
        {"speaker": "Candidate", "text": "and postgre sql for backend", "timestamp_ms": 2200},
        {"speaker": "Interviewer", "text": "What about deployment?", "timestamp_ms": 6000},
        {"speaker": "Candidate", "text": "We deployed on kube netes", "timestamp_ms": 7000},
        {"speaker": "Candidate", "text": "with Docker containers", "timestamp_ms": 8000},
    ]
    stitched = enhancer.stitch_utterances(blocks, max_gap_ms=2500)
    assert len(stitched) == 3
    assert stitched[0]["text"] == "I used FastAPI and PostgreSQL for backend"
    assert stitched[2]["text"] == "We deployed on Kubernetes with Docker containers"

import pytest

@pytest.mark.asyncio
async def test_sarvam_service_mock():
    stt = SarvamSTTService()
    res = await stt.transcribe_audio_bytes(b"dummy_audio", language_code="en-IN")
    assert res["status"] == "success"
    assert "transcript" in res
    assert "confidence_info" in res

if __name__ == "__main__":
    test_entity_normalization()
    test_utterance_stitching()
    asyncio.run(test_sarvam_service_mock())
    print("All enhancer unit tests passed successfully!")
