import hashlib
import json
import re
import statistics
import time
import os
import logging
from datetime import datetime
from typing import Any, Literal, List, Dict, Tuple, Optional

from pydantic import BaseModel, Field
from app.models.schemas import (
    AnswerProctoring,
    ProctoringFinding,
    ProctoringReport,
)

logger = logging.getLogger("verimeet-proctoring")

PROCTORING_MODEL_VERSION = "1.0.0"
MIN_WORDS_FOR_SCORING = 10
TOKEN_BUDGET_QA_LIMIT = 25

def question_hash(question: str) -> str:
    normalised = re.sub(r"\s+", " ", (question or "").strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]

ADVISORY_CAVEAT = (
    "This report is advisory only. Transcript-based authenticity scoring cannot "
    "definitively detect AI assistance and should never be used as the sole basis "
    "for hiring decisions. Use it to inform human review, not replace it."
)

FILLER_SINGLE = {
    "um", "uh", "uhh", "umm", "er", "erm", "ah", "ahh",
    "like", "basically", "actually", "literally", "honestly",
    "right", "okay", "well", "so", "hmm", "mhm", "yeah",
}
FILLER_MULTI = [
    "you know", "i mean", "kind of", "sort of",
    "i guess", "i think", "you see",
]

AI_TELL_PATTERNS = [
    r"\bit'?s important to note\b",
    r"\bthere are several\b",
    r"\bthere are many\b",
    r"\blet me break (this|that|it) down\b",
    r"\bcertainly\b",
    r"\bas we can see\b",
    r"\bin conclusion\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\badditionally\b",
    r"\bin summary\b",
    r"\bto summarize\b",
    r"\bdelve into\b",
    r"\btapestry of\b",
    r"\brealm of\b",
    r"\blandscape of\b",
    r"\bplays a (crucial|vital|key|significant|pivotal) role\b",
    r"\bit'?s worth noting\b",
    r"\bthat'?s a (great|excellent|good) question\b",
    r"\bcomprehensive (approach|understanding|overview)\b",
    r"\bmultifaceted\b",
    r"\bnavigate the\b",
    r"\bseamlessly integrate\b",
    r"\bparadigm shift\b",
    r"\bcutting-edge\b",
    r"\bstate-of-the-art\b",
    r"\bgame changer\b",
    r"\bsynergy\b",
    r"\bholistic\b",
    r"\brobust and scalable\b",
    r"\bharness the power\b",
    r"\btestament to\b",
    r"\bbeacon of\b",
    r"\bin order to\b",
    r"\bkey takeaway\b",
]

STRUCTURE_MARKERS = [
    r"\bfirst(?:ly)?\b",
    r"\bsecond(?:ly)?\b",
    r"\bthird(?:ly)?\b",
    r"\blastly\b",
    r"\bfinally\b",
    r"\bin addition\b",
    r"\bon the other hand\b",
    r"^\s*\d+[\.\)]\s",
]

SELF_REFS = {
    "i", "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "id",
    "my", "me", "myself", "mine", "we", "our",
}

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())

def disfluency_rate(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    single = sum(1 for t in tokens if t in FILLER_SINGLE)
    text_lower = text.lower()
    multi = 0
    for phrase in FILLER_MULTI:
        multi += len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))
    return (single + multi) / len(tokens)

def self_reference_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    refs = sum(1 for t in tokens if t in SELF_REFS)
    return refs / len(tokens)

def concreteness_per_100w(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    sentences = re.split(r"[.!?]+\s*", text)
    capitalized_mid = 0
    for sent in sentences:
        words = sent.strip().split()
        for word in words[1:]:
            if word and len(word) > 1 and word[0].isupper() and word[1:].islower():
                capitalized_mid += 1
    return (numbers + years + capitalized_mid) / (len(tokens) / 100)

def ai_tell_matches(text: str) -> List[str]:
    if not text:
        return []
    text_lower = text.lower()
    matches: List[str] = []
    for pattern in AI_TELL_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            matches.append(m.group(0))
    return matches

def structure_markers_per_100w(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    count = 0
    for pattern in STRUCTURE_MARKERS:
        count += len(re.findall(pattern, text_lower, flags=re.MULTILINE))
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return count / (len(tokens) / 100)

def question_mirror_score(question: str, answer: str) -> float:
    if not (question and answer):
        return 0.0
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)[:30]
    if len(q_tokens) < 3 or len(a_tokens) < 3:
        return 0.0
    q_3 = set(zip(q_tokens, q_tokens[1:], q_tokens[2:]))
    a_3 = set(zip(a_tokens, a_tokens[1:], a_tokens[2:]))
    if not q_3:
        return 0.0
    return len(q_3 & a_3) / len(q_3)

def sentence_length_stats(text: str) -> Dict[str, float]:
    sentences = [s.strip() for s in re.split(r"[.!?]+\s*", text) if s.strip()]
    if not sentences:
        return {"mean": 0.0, "stdev": 0.0, "count": 0}
    lengths = [len(s.split()) for s in sentences]
    return {
        "mean": float(statistics.mean(lengths)),
        "stdev": float(statistics.stdev(lengths)) if len(lengths) > 1 else 0.0,
        "count": len(sentences),
    }

def compute_answer_features(question: str, answer: str) -> Dict[str, Any]:
    tells = ai_tell_matches(answer)
    return {
        "word_count": len(tokenize(answer)),
        "disfluency_rate": disfluency_rate(answer),
        "self_reference_rate": self_reference_rate(answer),
        "concreteness_per_100w": concreteness_per_100w(answer),
        "ai_tell_count": len(tells),
        "ai_tells_matched": tells[:5],
        "structure_marker_per_100w": structure_markers_per_100w(answer),
        "mirror_score": question_mirror_score(question, answer),
        "sentence_stats": sentence_length_stats(answer),
    }

def compute_baseline(intro_qa_pairs: List[Dict]) -> Dict[str, Any]:
    if not intro_qa_pairs:
        return {}
    combined = " ".join(qa.get("answer", "") for qa in intro_qa_pairs if qa.get("answer"))
    if not combined.strip():
        return {}
    return {
        "based_on_word_count": len(tokenize(combined)),
        "disfluency_rate": disfluency_rate(combined),
        "self_reference_rate": self_reference_rate(combined),
        "concreteness_per_100w": concreteness_per_100w(combined),
        "structure_marker_per_100w": structure_markers_per_100w(combined),
        "sentence_stats": sentence_length_stats(combined),
    }

class JudgeFinding(BaseModel):
    name: str = Field(default="finding")
    evidence_quote: str = Field(description="Verbatim excerpt from the candidate's answer")
    explanation: str = Field(description="Why this excerpt is suspicious")

class JudgeAnswerScore(BaseModel):
    question_index: int = Field(description="0-based index into the Q&A list")
    score: float = Field(ge=0.0, le=1.0, description="0=no concern, 1=strong concern")
    risk_band: Literal["low", "medium", "high"]
    ai_reliance_percent: Optional[float] = Field(default=None, description="Estimated AI reliance percentage 0 to 100")
    speech_naturalness_score: Optional[float] = Field(default=None, description="Speech naturalness rating 0 to 100")
    findings: List[JudgeFinding] = Field(default_factory=list)

class JudgeSessionReport(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "medium", "high"]
    ai_reliance_percent: Optional[float] = Field(default=None, description="Overall estimated AI reliance percentage")
    overall_risk_score: Optional[float] = Field(default=None, description="Overall session risk index 0 to 100")
    pattern_findings: List[str] = Field(default_factory=list)
    recommendation: str
    per_answer: List[JudgeAnswerScore]

class ProctoringService:
    def __init__(self) -> None:
        self._llm = None
        self._llm_provider = None
        self._llm_init_attempted = False

    def _ensure_llm(self) -> None:
        if self._llm_init_attempted:
            return
        self._llm_init_attempted = True
        sarvam_key = os.getenv("SARVAM_API_KEY")
        if not sarvam_key:
            logger.warning("SARVAM_API_KEY not found; heuristic-only reports")
            return
        try:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model="sarvam-105b",
                api_key=sarvam_key,
                base_url="https://api.sarvam.ai/v1",
                temperature=0.3,
                max_tokens=4096,
            )
            self._llm_provider = "sarvam"
            logger.info("ProctoringService initialised with Sarvam LLM judge")
        except Exception as e:
            logger.warning("Sarvam LLM init raised; heuristic-only reports", exc_info=True)

    async def score_session(self, transcript: dict, job_title: str) -> ProctoringReport:
        start = time.time()
        qa_pairs = transcript.get("qa_pairs", []) or []

        if not qa_pairs:
            return self._empty_report("no_qa_pairs", elapsed_ms=0)

        self._ensure_llm()

        features = [
            compute_answer_features(qa.get("question", ""), qa.get("answer", ""))
            for qa in qa_pairs
        ]

        intro_pairs = [qa for qa in qa_pairs if qa.get("category") == "introduction"]
        if not intro_pairs:
            intro_pairs = [qa_pairs[0]]
        baseline = compute_baseline(intro_pairs)

        judge_report: JudgeSessionReport | None = None
        status: Literal["ok", "degraded", "failed"] = "ok"

        if self._llm is None:
            status = "degraded"
        else:
            try:
                judge_report = await self._call_judge(qa_pairs, features, baseline, job_title)
            except Exception as e:
                logger.warning("Judge call failed; degrading to heuristics", exc_info=True)
                status = "degraded"

        report = self._aggregate(
            qa_pairs=qa_pairs,
            features=features,
            baseline=baseline,
            judge=judge_report,
            status=status,
            elapsed_ms=int((time.time() - start) * 1000),
        )
        return report

    async def _call_judge(
        self,
        qa_pairs: List[Dict],
        features: List[Dict],
        baseline: Dict,
        job_title: str,
    ) -> JudgeSessionReport:
        if len(qa_pairs) > TOKEN_BUDGET_QA_LIMIT:
            qa_pairs = qa_pairs[:TOKEN_BUDGET_QA_LIMIT]
            features = features[:TOKEN_BUDGET_QA_LIMIT]

        prompt = self._build_judge_prompt(qa_pairs, features, baseline, job_title)
        
        try:
            raw_res = await self._llm.ainvoke(prompt)
            content = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            
            cleaned = content.strip()
            if "```" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1:
                    cleaned = cleaned[start:end + 1]
            
            try:
                json_obj = json.loads(cleaned)
            except Exception:
                cleaned_fixed = re.sub(r'(?<=: ")(.*?)(?=",\n|"\n|\"\})', lambda m: m.group(1).replace('\n', ' '), cleaned, flags=re.DOTALL)
                try:
                    json_obj = json.loads(cleaned_fixed)
                except Exception:
                    end_pos = cleaned.rfind("}")
                    if end_pos != -1:
                        json_obj = json.loads(cleaned[:end_pos + 1])
                    else:
                        raise

            return JudgeSessionReport.model_validate(json_obj)
        except Exception as e:
            logger.error(f"Sarvam 105B judge parsing error: {e}")
            raise

    def _build_judge_prompt(
        self,
        qa_pairs: List[Dict],
        features: List[Dict],
        baseline: Dict,
        job_title: str,
    ) -> str:
        baseline_block = ""
        if baseline:
            baseline_block = (
                "\nCANDIDATE NATURAL BASELINE (from their introduction speech):\n"
                f"- Computed from {baseline.get('based_on_word_count', 0)} words\n"
                f"- Disfluency rate: {baseline.get('disfluency_rate', 0):.3f}\n"
                f"- Self-reference rate: {baseline.get('self_reference_rate', 0):.3f}\n"
                "EVALUATE SUBSEQUENT ANSWERS AS DELTAS FROM THIS BASELINE.\n\n"
            )

        body_parts = []
        for i, (qa, f) in enumerate(zip(qa_pairs, features)):
            tells = ", ".join(f.get("ai_tells_matched", [])) or "none"
            block = (
                f"\n--- Answer {i + 1} ---\n"
                f"Question: {qa.get('question', '')}\n"
                f"Answer: {qa.get('answer', '')}\n"
                f"Features: words={f.get('word_count', 0)}, "
                f"disfluency={f.get('disfluency_rate', 0.0):.3f}, "
                f"self_ref={f.get('self_reference_rate', 0.0):.3f}, "
                f"ai_tells={f.get('ai_tell_count', 0)} ({tells})"
            )
            body_parts.append(block)

        return (
            "You are an authenticity reviewer for AI-assisted job interviews. Your job is to "
            "flag ANSWERS where the candidate may have used an AI tool (like ChatGPT) to "
            "generate the response, so a human reviewer can investigate.\n\n"
            f"Position being interviewed for: {job_title}\n"
            f"{baseline_block}"
            "INTERVIEW TRANSCRIPT:\n"
            f"{''.join(body_parts)}\n\n"
            "WHAT TO LOOK FOR:\n"
            "1. Generic AI structure: balanced enumerated lists, formulaic transitions.\n"
            "2. Mode switching: sudden shifts in vocabulary, tone, or formality.\n"
            "3. Suspicious concreteness gaps: confident claims with no names/dates.\n"
            "4. AI tells: 'delve into', 'it's important to note', 'landscape of'.\n\n"
            "RULES FOR OUTPUT:\n"
            "- Include overall_score (0-1), risk_band ('low'|'medium'|'high'), ai_reliance_percent (0-100), overall_risk_score (0-100).\n"
            "- Be concise. Keep evidence_quote under 8 words and explanation under 15 words.\n"
            "- Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "overall_score": 0.85,\n'
            '  "overall_risk_score": 85.0,\n'
            '  "ai_reliance_percent": 78.0,\n'
            '  "risk_band": "high",\n'
            '  "pattern_findings": ["Short pattern finding 1"],\n'
            '  "recommendation": "Short recommendation",\n'
            '  "per_answer": [\n'
            '    {\n'
            '      "question_index": 0,\n'
            '      "score": 0.85,\n'
            '      "risk_band": "high",\n'
            '      "ai_reliance_percent": 80.0,\n'
            '      "speech_naturalness_score": 35.0,\n'
            '      "findings": [{"name": "ai_tell", "evidence_quote": "short quote", "explanation": "short explanation"}]\n'
            '    }\n'
            '  ]\n'
            "}"
        )

    def _aggregate(
        self,
        qa_pairs: List[Dict],
        features: List[Dict],
        baseline: Dict,
        judge: JudgeSessionReport | None,
        status: Literal["ok", "degraded", "failed"],
        elapsed_ms: int,
    ) -> ProctoringReport:
        per_answer: List[AnswerProctoring] = []

        if judge is not None:
            judge_by_idx = {j.question_index: j for j in judge.per_answer}
            for i, (qa, f) in enumerate(zip(qa_pairs, features)):
                j = judge_by_idx.get(i)
                if j is None:
                    score, band, findings = 0.0, "low", []
                    ai_rel = 0.0
                    nat_score = 100.0
                else:
                    score, band = j.score, j.risk_band
                    ai_rel = j.ai_reliance_percent if j.ai_reliance_percent is not None else round(score * 100, 1)
                    nat_score = j.speech_naturalness_score if j.speech_naturalness_score is not None else round(max(0.0, 100.0 - (ai_rel * 0.7)), 1)
                    findings = [
                        ProctoringFinding(
                            name=ff.name or "finding",
                            evidence_quote=ff.evidence_quote,
                            explanation=ff.explanation,
                        )
                        for ff in j.findings
                    ]
                if f["word_count"] < MIN_WORDS_FOR_SCORING:
                    score = min(score, 0.2)
                    ai_rel = min(ai_rel, 20.0)
                    band = "low"
                per_answer.append(
                    AnswerProctoring(
                        question_index=i,
                        question_preview=qa.get("question", "")[:100],
                        score=score,
                        risk_band=band,
                        ai_reliance_percent=ai_rel,
                        speech_naturalness_score=nat_score,
                        findings=findings,
                        heuristic_features=f,
                    )
                )
            overall_score = judge.overall_score
            overall_band = judge.risk_band
            overall_ai_reliance = judge.ai_reliance_percent if judge.ai_reliance_percent is not None else (
                round(sum(a.ai_reliance_percent for a in per_answer) / max(len(per_answer), 1), 1)
            )
            overall_risk = judge.overall_risk_score if judge.overall_risk_score is not None else (
                round(min(100.0, (overall_score * 70.0) + (overall_ai_reliance * 0.3)), 1)
            )
            pattern_findings = list(judge.pattern_findings) if judge.pattern_findings else []
            recommendation = judge.recommendation
        else:
            for i, (qa, f) in enumerate(zip(qa_pairs, features)):
                hscore = self._heuristic_score(f)
                if f["word_count"] < MIN_WORDS_FOR_SCORING:
                    hscore = min(hscore, 0.2)
                band = "high" if hscore >= 0.6 else "medium" if hscore >= 0.3 else "low"
                ai_rel = round(hscore * 100, 1)
                nat_score = round(max(0.0, 100.0 - (ai_rel * 0.75)), 1)

                h_findings: List[ProctoringFinding] = []
                if f.get("ai_tell_count", 0) > 0:
                    matched_tells = f.get("ai_tells_matched", [])
                    quote = matched_tells[0] if matched_tells else "formulaic phrase"
                    h_findings.append(ProctoringFinding(
                        name="ai_tell_pattern",
                        evidence_quote=quote,
                        explanation="Detected formulaic LLM transition phrase in candidate response."
                    ))
                if f.get("structure_marker_per_100w", 0.0) > 1.5:
                    h_findings.append(ProctoringFinding(
                        name="formulaic_enumeration",
                        evidence_quote="Firstly / Secondly...",
                        explanation="Structured list enumeration characteristic of AI-generated answers."
                    ))
                if f.get("disfluency_rate", 0.0) < 0.005 and f.get("word_count", 0) > 40:
                    h_findings.append(ProctoringFinding(
                        name="unnatural_cadence",
                        evidence_quote=qa.get("answer", "")[:35] + "...",
                        explanation="Abnormal absence of natural human speech fillers and pauses."
                    ))
                if f.get("mirror_score", 0.0) > 0.15:
                    h_findings.append(ProctoringFinding(
                        name="question_mirroring",
                        evidence_quote=qa.get("answer", "")[:30] + "...",
                        explanation="Candidate repeated exact phrasing from the interviewer prompt."
                    ))

                per_answer.append(
                    AnswerProctoring(
                        question_index=i,
                        question_preview=qa.get("question", "")[:100],
                        score=hscore,
                        risk_band=band,
                        ai_reliance_percent=ai_rel,
                        speech_naturalness_score=nat_score,
                        findings=h_findings,
                        heuristic_features=f,
                    )
                )
            overall_score = sum(a.score for a in per_answer) / max(len(per_answer), 1) if per_answer else 0.0
            overall_band = "high" if overall_score >= 0.6 else "medium" if overall_score >= 0.3 else "low"
            overall_ai_reliance = round(sum(a.ai_reliance_percent for a in per_answer) / max(len(per_answer), 1), 1) if per_answer else 0.0
            overall_risk = round(min(100.0, (overall_score * 65.0) + (overall_ai_reliance * 0.35)), 1)
            pattern_findings = ["AI reliance calculated via pattern heuristic & baseline prosody delta."]
            recommendation = "Review flagged technical answers for AI assistance." if overall_risk > 50 else "Acceptable response authenticity."

        share_high_risk = sum(1 for a in per_answer if a.risk_band == "high") / max(len(per_answer), 1)
        avg_nat_score = round(sum(a.speech_naturalness_score for a in per_answer) / max(len(per_answer), 1), 1) if per_answer else 100.0

        # Calculate micro risk factors breakdown
        total_tells = sum(f.get("ai_tell_count", 0) for f in features)
        tell_score = min(100.0, round((total_tells / max(len(features), 1)) * 35.0, 1))
        mirror_score = min(100.0, round(sum(f.get("mirror_score", 0.0) for f in features) / max(len(features), 1) * 100.0, 1))
        disfluency_deficit = min(100.0, round(max(0.0, (0.04 - (sum(f.get("disfluency_rate", 0.0) for f in features) / max(len(features), 1))) * 2000.0), 1))

        risk_factors_breakdown = {
            "ai_tell_patterns": tell_score,
            "speech_naturalness": avg_nat_score,
            "disfluency_deficit": disfluency_deficit,
            "question_mirroring": mirror_score
        }

        # Fetch vision proctoring metrics
        from app.services.vision_proctoring import get_vision_service
        vision_svc = get_vision_service()
        session_id = qa_pairs[0].get("session_id", "demo_session") if qa_pairs else "demo_session"
        vision_summary = vision_svc.get_session_summary(session_id)
        vision_events = vision_summary.get("anomaly_events", [])

        return ProctoringReport(
            overall_score=overall_score,
            overall_risk_score=overall_risk,
            ai_reliance_percent=overall_ai_reliance,
            speech_naturalness_score=avg_nat_score,
            risk_band=overall_band,
            share_high_risk=share_high_risk,
            risk_factors_breakdown=risk_factors_breakdown,
            vision_summary=vision_summary,
            vision_events=vision_events,
            pattern_findings=pattern_findings,
            recommendation=recommendation,
            caveat=ADVISORY_CAVEAT,
            per_answer=per_answer,
            baseline_summary=baseline,
            model_version=PROCTORING_MODEL_VERSION,
            status=status,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _heuristic_score(feats: Dict) -> float:
        score = 0.0
        # Each AI buzzword/tell phrase contributes +0.30 (max 0.85)
        score += min(feats.get("ai_tell_count", 0) * 0.30, 0.85)
        # Structural markers (firstly, secondly, in addition) contribute up to 0.25
        score += min(feats.get("structure_marker_per_100w", 0.0) / 4.0, 0.25)
        # Question prompt mirroring contributes up to 0.25
        score += min(feats.get("mirror_score", 0.0) * 1.2, 0.25)
        # Complete absence of natural fillers in answers >20 words contributes +0.20
        if feats.get("disfluency_rate", 0.0) < 0.005 and feats.get("word_count", 0) > 20:
            score += 0.20
        # Pre-answer silence pause (>5s formulation pause typed into ChatGPT/teleprompter)
        pre_silence = feats.get("pre_answer_silence_ms") or feats.get("timing", {}).get("pre_answer_silence_ms")
        if pre_silence and pre_silence > 5000:
            score += min((pre_silence - 5000) / 10000.0, 0.20)
        return min(score, 1.0)

    def _empty_report(self, reason: str, elapsed_ms: int) -> ProctoringReport:
        return ProctoringReport(
            overall_score=0.0,
            risk_band="low",
            share_high_risk=0.0,
            pattern_findings=[reason],
            recommendation="no action",
            caveat=ADVISORY_CAVEAT,
            per_answer=[],
            baseline_summary={},
            model_version=PROCTORING_MODEL_VERSION,
            status="failed",
            elapsed_ms=elapsed_ms,
        )

_proctoring_service: ProctoringService | None = None

def get_proctoring_service() -> ProctoringService:
    global _proctoring_service
    if _proctoring_service is None:
        _proctoring_service = ProctoringService()
    return _proctoring_service
