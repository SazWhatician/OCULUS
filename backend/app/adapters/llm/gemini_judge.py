import base64
import json
import logging
from typing import Optional, Dict, Any
import httpx
from app.config import settings
from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    LLMJudgeVerdict,
    ConfidenceBand,
)
from app.adapters.llm.base import LLMJudgeAdapter

logger = logging.getLogger("oculus.llm.gemini")


class GeminiJudgeAdapter(LLMJudgeAdapter):
    """
    Google AI Studio Gemini Flash adapter:
    1. Long-context semantic evaluation fallback when Groq hits 429 rate limits.
    2. Budget-capped Tier 2 Keyframe VLM Verifier (detects secondary devices/phones).
    """

    @property
    def provider_name(self) -> str:
        return "gemini_flash"

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.is_available = bool(self.api_key)

    async def judge_turn(
        self,
        question_text: str,
        answer_text: str,
        job_title: str,
        baseline: Optional[CandidateBaseline] = None,
        heuristics: Optional[HeuristicScores] = None,
    ) -> LLMJudgeVerdict:
        if not self.is_available:
            return self._mock_verdict(answer_text)

        prompt = (
            f"You are evaluating interview integrity for a {job_title} role.\n"
            f"Question: {question_text}\n"
            f"Answer: {answer_text}\n"
            "Return JSON matching: {suspicion_score: float (0-100), confidence_band: 'LOW'|'MEDIUM'|'HIGH', "
            "reasoning_signals: [str], verbatim_evidence_quotes: [str], recitation_vs_live_assessment: str, advisory_follow_up_question: str}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2},
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw_text)

            return LLMJudgeVerdict(
                answer_id=f"ans_gem_{hash(answer_text) % 100000}",
                provider_used="gemini_2.0_flash",
                suspicion_score=float(parsed.get("suspicion_score", 50.0)),
                confidence_band=ConfidenceBand(parsed.get("confidence_band", "MEDIUM")),
                reasoning_signals=parsed.get("reasoning_signals", []),
                verbatim_evidence_quotes=parsed.get("verbatim_evidence_quotes", []),
                recitation_vs_live_assessment=parsed.get("recitation_vs_live_assessment", ""),
                advisory_follow_up_question=parsed.get("advisory_follow_up_question", ""),
            )

    async def verify_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """
        Tier 2 VLM verification of a single flagged anomaly frame.
        Checks for phone, tablet, or off-screen reading posture.
        """
        if not self.is_available:
            return {
                "phone_or_tablet_visible": False,
                "secondary_screen_reflection": False,
                "note": "VLM verification skipped (API key not configured).",
            }

        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        prompt = (
            "Inspect this interview video frame. Detect if a secondary phone, mobile device, or tablet is visible "
            "in the candidate's hands or lap area, or if candidate is reading off an off-screen device. "
            "Respond strictly in JSON: {\"phone_or_tablet_visible\": bool, \"secondary_screen_reflection\": bool, \"note\": string}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                    ]
                }
            ],
            "generationConfig": {"response_mime_type": "application/json"},
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini VLM verification error: {e}")
            return {
                "phone_or_tablet_visible": False,
                "secondary_screen_reflection": False,
                "note": f"VLM call failed: {e}",
            }

    def _mock_verdict(self, answer_text: str) -> LLMJudgeVerdict:
        is_suspicious = "multifaceted" in answer_text.lower()
        return LLMJudgeVerdict(
            answer_id=f"ans_gem_mock_{hash(answer_text) % 100000}",
            provider_used="gemini_mock",
            suspicion_score=75.0 if is_suspicious else 25.0,
            confidence_band=ConfidenceBand.MEDIUM,
            reasoning_signals=["Analyzed semantic depth"],
            verbatim_evidence_quotes=[],
            recitation_vs_live_assessment="Mock Gemini evaluation complete.",
            advisory_follow_up_question="Can you walk me through your personal contribution?",
        )
