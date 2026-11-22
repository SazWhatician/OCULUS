import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.db.postgres import init_db


@pytest.mark.asyncio
async def test_api_health_and_sessions():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res_h = await client.get("/healthz")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "healthy"

        # 2. Metrics probe
        res_m = await client.get("/api/v1/metrics")
        assert res_m.status_code == 200
        assert "counters" in res_m.json()

        # 3. Create session with tech vocab extraction
        payload = {
            "candidate_name": "Bob Smith",
            "interviewer_name": "Alice Lead",
            "job_title": "Senior Cloud Infrastructure Engineer",
            "job_description": "Must have hands-on experience with Kubernetes, Docker, and AWS.",
            "ingestion_mode": "demo_file",
        }
        res_create = await client.post("/api/v1/sessions", json=payload)
        assert res_create.status_code == 200
        data = res_create.json()
        assert "session_id" in data
        assert "Kubernetes" in data["custom_vocab"]
        session_id = data["session_id"]

        # 4. Get session
        res_get = await client.get(f"/api/v1/sessions/{session_id}")
        assert res_get.status_code == 200
        assert res_get.json()["candidate_name"] == "Bob Smith"

        # 5. Swap speakers
        res_swap = await client.patch(
            f"/api/v1/sessions/{session_id}/speakers",
            json={"speaker_role_map": {"S1": "candidate", "S2": "interviewer"}},
        )
        assert res_swap.status_code == 200
        assert res_swap.json()["speaker_role_map"]["S1"] == "candidate"

        # 6. List sessions
        res_list = await client.get("/api/v1/sessions")
        assert res_list.status_code == 200
        assert len(res_list.json()["sessions"]) >= 1
