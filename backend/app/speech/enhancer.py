import re
from typing import List, Dict, Any, Optional

# Comprehensive technical terms & domain entity dictionary for precision speech normalization
TECHNICAL_TERMS_MAP = {
    # Programming Languages & Frameworks
    r"\breact js\b": "React.js",
    r"\breactjs\b": "React",
    r"\bnext js\b": "Next.js",
    r"\bnextjs\b": "Next.js",
    r"\bnode js\b": "Node.js",
    r"\bnodejs\b": "Node.js",
    r"\bfast api\b": "FastAPI",
    r"\bfastapi\b": "FastAPI",
    r"\bvue js\b": "Vue.js",
    r"\bexpress js\b": "Express.js",
    r"\btype script\b": "TypeScript",
    r"\bjava script\b": "JavaScript",
    r"\bpython three\b": "Python 3",
    r"\bpy torch\b": "PyTorch",
    r"\btensor flow\b": "TensorFlow",

    # Databases & Caching
    r"\bpostgre sql\b": "PostgreSQL",
    r"\bpostgres sql\b": "PostgreSQL",
    r"\bpost gres\b": "PostgreSQL",
    r"\bmy sql\b": "MySQL",
    r"\bmongo db\b": "MongoDB",
    r"\bre dis\b": "Redis",
    r"\belastic search\b": "Elasticsearch",

    # DevOps, Cloud & Architecture
    r"\bkube netes\b": "Kubernetes",
    r"\bk8s\b": "Kubernetes",
    r"\bdocker\b": "Docker",
    r"\baws\b": "AWS",
    r"\bgcp\b": "GCP",
    r"\bazure\b": "Azure",
    r"\bci cd\b": "CI/CD",
    r"\bci\/cd\b": "CI/CD",
    r"\brest api\b": "REST API",
    r"\brestful api\b": "RESTful API",
    r"\bgraph ql\b": "GraphQL",
    r"\bg rpc\b": "gRPC",
    r"\bweb socket\b": "WebSocket",
    r"\bweb sockets\b": "WebSockets",
    r"\bmicro service\b": "Microservice",
    r"\bmicro services\b": "Microservices",
    r"\bka fka\b": "Kafka",
    r"\bra bbit mq\b": "RabbitMQ",

    # AI & Speech Systems
    r"\bsarvam\b": "Sarvam AI",
    r"\bsarvam ai\b": "Sarvam AI",
    r"\bsaaras\b": "Saaras",
    r"\bdeep gram\b": "Deepgram",
    r"\bopen ai\b": "OpenAI",
    r"\bwhisper\b": "Whisper",
    r"\bllm\b": "LLM",
    r"\bllms\b": "LLMs",
    r"\brag\b": "RAG",
    r"\bstt\b": "STT",
    r"\btts\b": "TTS",

    # Institutions & Local Entities
    r"\bBCRMSI\b": "VSSUT Burla",
    r"\bVCRMSI\b": "VSSUT Burla",
    r"\bvssut\b": "VSSUT Burla",
    r"\bbput\b": "BPUT",
    r"\biit\b": "IIT",
    r"\bnit\b": "NIT",
    r"\biiit\b": "IIIT",
}

DISFLUENCY_PATTERNS = [
    r"\b(um+|uh+|er+|ah+|hmm+)\b",
    r"\b(you know|like|i mean|sort of|kind of)\b(?=[\s,.]|\b)",
    r"\b(\w+)\s+\1\b"  # Repeated contiguous words e.g. "the the" or "I I"
]


class TranscriptEnhancer:
    """
    Advanced speech post-processing engine providing domain entity normalization,
    disfluency/filler cleanup, temporal utterance stitching, and confidence scoring.
    """

    def __init__(self, terms_map: Optional[Dict[str, str]] = None):
        self.terms_map = terms_map or TECHNICAL_TERMS_MAP

    def normalize_entities(self, text: str) -> str:
        """Replace phonetic STT misrecognitions with standard technical terminology."""
        if not text:
            return text
        normalized = text
        for pattern, replacement in self.terms_map.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Clean up double spaces or loose punctuation spacing
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def remove_disfluencies(self, text: str) -> str:
        """Strip filler words and hesitations ('um', 'uh', repeated words) while preserving sentence flow."""
        if not text:
            return text
        cleaned = text
        for pattern in DISFLUENCY_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        # Clean up stray commas or multiple punctuation left after filler removal
        cleaned = re.sub(r",\s*,", ",", cleaned)
        cleaned = re.sub(r"^\s*,\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Fix comma before space or capitalization
        cleaned = re.sub(r"\s+,", ",", cleaned)
        # Restore sentence capitalization
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def annotate_confidence(self, text: str, raw_confidence: float = 0.95) -> Dict[str, Any]:
        """
        Classifies confidence into qualitative bands ('high', 'medium', 'low')
        and identifies words that may need reviewer inspection.
        """
        band = "high"
        if raw_confidence < 0.70:
            band = "low"
        elif raw_confidence < 0.85:
            band = "medium"

        words = text.split()
        flagged_words = []
        # Flag unusually short words or unknown acronyms if overall confidence is low/medium
        if band != "high":
            flagged_words = [w for w in words if len(w) <= 3 and w.isalpha()]

        return {
            "score": round(raw_confidence, 3),
            "band": band,
            "is_flagged": band != "high",
            "flagged_words": flagged_words
        }

    def enhance_text(self, text: str, clean_fillers: bool = True) -> str:
        """Apply full normalization and optional filler cleanup."""
        if not text:
            return ""
        result = self.normalize_entities(text)
        if clean_fillers:
            result = self.remove_disfluencies(result)
        return result

    def stitch_utterances(
        self,
        utterances: List[Dict[str, Any]],
        max_gap_ms: int = 2500,
        clean_fillers: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Stitches fragmented utterances from the same speaker if the timestamp gap is within `max_gap_ms`.
        Combines raw text before enhancing so sentence capitalization and entity normalization work seamlessly.
        """
        if not utterances:
            return []

        grouped_blocks: List[Dict[str, Any]] = []

        for block in utterances:
            speaker = block.get("speaker_name") or block.get("speaker") or "Participant"
            transcription = block.get("transcription") or block.get("text") or ""
            ts_ms = block.get("timestamp_ms", 0)
            confidence = block.get("confidence", 0.95)

            if isinstance(transcription, dict):
                raw_text = transcription.get("transcript", "").strip()
            else:
                raw_text = str(transcription).strip()

            if not raw_text:
                continue

            if grouped_blocks and grouped_blocks[-1]["speaker"] == speaker:
                prev_ts = grouped_blocks[-1].get("timestamp_ms", 0)
                duration = grouped_blocks[-1].get("duration_ms", 0)
                end_time = prev_ts + duration if duration else prev_ts + 2000

                if ts_ms == 0 or (ts_ms - end_time) <= max_gap_ms:
                    grouped_blocks[-1]["raw_text"] += f" {raw_text}"
                    grouped_blocks[-1]["confidences"].append(confidence)
                    continue

            grouped_blocks.append({
                "speaker": speaker,
                "raw_text": raw_text,
                "timestamp_ms": ts_ms,
                "confidences": [confidence]
            })

        stitched: List[Dict[str, Any]] = []
        for g in grouped_blocks:
            combined_raw = g["raw_text"]
            enhanced_text = self.enhance_text(combined_raw, clean_fillers=clean_fillers)
            avg_conf = round(sum(g["confidences"]) / len(g["confidences"]), 3)
            conf_info = self.annotate_confidence(enhanced_text, avg_conf)

            stitched.append({
                "speaker": g["speaker"],
                "speaker_name": g["speaker"],
                "text": enhanced_text,
                "raw_text": combined_raw,
                "timestamp_ms": g["timestamp_ms"],
                "confidence": avg_conf,
                "confidence_info": conf_info
            })

        return stitched
