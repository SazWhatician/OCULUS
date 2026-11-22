from abc import ABC, abstractmethod
from typing import Optional
from app.schemas.contracts import (
    CandidateBaseline,
    HeuristicScores,
    LLMJudgeVerdict,
)
from app.services.resilience import CircuitBreaker


class LLMJudgeAdapter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def judge_turn(
        self,
        question_text: str,
        answer_text: str,
        job_title: str,
        baseline: Optional[CandidateBaseline] = None,
        heuristics: Optional[HeuristicScores] = None,
    ) -> LLMJudgeVerdict:
        pass


class FallbackLLMRouter:
    """
    Tries primary LLM (Groq) with circuit breaker protection.
    On HTTP 429/timeout/circuit open, automatically falls over to Gemini Flash.
    On complete failure, returns None so the pipeline safely falls back to Stage 1.
    """

    def __init__(self):
        from app.adapters.llm.groq_judge import GroqJudgeAdapter
        from app.adapters.llm.gemini_judge import GeminiJudgeAdapter

        self.groq_judge = GroqJudgeAdapter()
        self.gemini_judge = GeminiJudgeAdapter()
        self.cb_groq = CircuitBreaker("groq_judge", failure_threshold=3, cooldown_seconds=30)
        self.cb_gemini = CircuitBreaker("gemini_judge", failure_threshold=3, cooldown_seconds=30)

    async def evaluate_turn(
        self,
        question_text: str,
        answer_text: str,
        job_title: str,
        baseline: Optional[CandidateBaseline] = None,
        heuristics: Optional[HeuristicScores] = None,
    ) -> Optional[LLMJudgeVerdict]:
        # Step 1: Try Groq
        if self.groq_judge.is_available and self.cb_groq.can_execute():
            try:
                verdict = await self.groq_judge.judge_turn(
                    question_text=question_text,
                    answer_text=answer_text,
                    job_title=job_title,
                    baseline=baseline,
                    heuristics=heuristics,
                )
                self.cb_groq.record_success()
                return verdict
            except Exception as e:
                self.cb_groq.record_failure(e)

        # Step 2: Fall back to Gemini Flash
        if self.gemini_judge.is_available and self.cb_gemini.can_execute():
            try:
                verdict = await self.gemini_judge.judge_turn(
                    question_text=question_text,
                    answer_text=answer_text,
                    job_title=job_title,
                    baseline=baseline,
                    heuristics=heuristics,
                )
                self.cb_gemini.record_success()
                return verdict
            except Exception as e:
                self.cb_gemini.record_failure(e)

        # Step 3: Mock verdict if no keys configured
        if not self.groq_judge.is_available and not self.gemini_judge.is_available:
            return self.groq_judge._mock_verdict(answer_text)

        return None
