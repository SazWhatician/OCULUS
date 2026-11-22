import json
import logging
from typing import Optional
import httpx
from app.config import settings
from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    LLMJudgeVerdict,
    ConfidenceBand,
)
from app.adapters.llm.base import LLMJudgeAdapter

logger = logging.getLogger("oculus.llm.groq")


class GroqJudgeAdapter(LLMJudgeAdapter):
    """
    Evaluates candidate response authenticity using high-speed reasoning models on Groq
    (llama-3.3-70b-versatile or qwen-2.5-32b) with native JSON output enforcement.
    """

    @property
    def provider_name(self) -> str:
        return "groq"

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
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

        system_prompt = (
            "You are an expert technical interviewer and interview integrity evaluator. "
            "Analyze whether the candidate's answer was constructed live or recited from an AI generation/teleprompter. "
            "Do NOT penalize non-native English grammar, informal phrasing, or accent pauses. "
            "Focus on: textbook abstractions without personal ownership, formulaic tripartite structuring, and question mirroring. "
            "Output strictly valid JSON with keys: suspicion_score (0-100 float), confidence_band ('LOW'|'MEDIUM'|'HIGH'), "
            "reasoning_signals (list of strings), verbatim_evidence_quotes (list of exact quotes from answer), "
            "recitation_vs_live_assessment (string summary), advisory_follow_up_question (string probe question for the interviewer)."
        )

        user_content = json.dumps({
            "job_title": job_title,
            "question": question_text,
            "answer": answer_text,
            "deterministic_heuristics": heuristics.model_dump() if heuristics else {},
            "candidate_baseline": baseline.model_dump() if baseline else {},
        })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 800,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            return LLMJudgeVerdict(
                answer_id=f"ans_{hash(answer_text) % 100000}",
                provider_used="groq_llama70b",
                suspicion_score=float(parsed.get("suspicion_score", 50.0)),
                confidence_band=ConfidenceBand(parsed.get("confidence_band", "MEDIUM")),
                reasoning_signals=parsed.get("reasoning_signals", []),
                verbatim_evidence_quotes=parsed.get("verbatim_evidence_quotes", []),
                recitation_vs_live_assessment=parsed.get("recitation_vs_live_assessment", ""),
                advisory_follow_up_question=parsed.get("advisory_follow_up_question", ""),
            )

    def _mock_verdict(self, answer_text: str) -> LLMJudgeVerdict:
        is_suspicious = "crucial to note" in answer_text.lower() or "delve into" in answer_text.lower()
        score = 80.0 if is_suspicious else 20.0
        return LLMJudgeVerdict(
            answer_id=f"ans_mock_{hash(answer_text) % 100000}",
            provider_used="groq_mock",
            suspicion_score=score,
            confidence_band=ConfidenceBand.HIGH if is_suspicious else ConfidenceBand.LOW,
            reasoning_signals=["Detected formulaic transitions", "Structured enumeration"] if is_suspicious else ["Natural human conversational pauses"],
            verbatim_evidence_quotes=["It is crucial to note"] if is_suspicious else [],
            recitation_vs_live_assessment="Answer exhibits synthetic textbook cadence." if is_suspicious else "Candidate demonstrates authentic live construction.",
            advisory_follow_up_question="Can you dive into a specific failure case you encountered while implementing that?" if is_suspicious else "What trade-offs did you consider?",
        )
