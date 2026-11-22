import re
from typing import List, Set

COMMON_TECH_TERMS: Set[str] = {
    "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "FastAPI",
    "Docker", "Kubernetes", "Redis", "PostgreSQL", "MySQL", "MongoDB", "Kafka",
    "RabbitMQ", "GraphQL", "REST", "gRPC", "AWS", "GCP", "Azure", "Cloudflare",
    "Terraform", "PyTorch", "TensorFlow", "Pandas", "NumPy", "OpenCV", "MediaPipe",
    "WebRTC", "WebSocket", "Microservices", "CI/CD", "Git", "OAuth", "JWT",
    "Elasticsearch", "Prometheus", "Grafana", "SQLAlchemy", "Pydantic", "Celery",
}


def extract_custom_vocab(job_title: str, job_description: str = "", max_terms: int = 50) -> List[str]:
    """
    Extracts high-value technical domain keywords from job title and description
    to feed speech-to-text language model prompts (Speechmatics / Groq Whisper).
    """
    combined_text = f"{job_title} {job_description}"
    vocab: Set[str] = set()

    # 1. Match against curated engineering terms (case-insensitive search, store canonical)
    for term in COMMON_TECH_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", combined_text, re.IGNORECASE):
            vocab.add(term)

    # 2. Extract capitalized acronyms (2-6 letters like AWS, SQL, gRPC, API, HTTP)
    acronyms = re.findall(r"\b[A-Z0-9]{2,6}\b", combined_text)
    for acr in acronyms:
        if len(acr) >= 2 and not acr.isnumeric():
            vocab.add(acr)

    # 3. Extract PascalCase/CamelCase tokens (e.g. DynamoDB, Fastify, NextJS)
    camel_cases = re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b", combined_text)
    for cc in camel_cases:
        vocab.add(cc)

    # 4. Extract hyphenated tech identifiers (e.g. CI-CD, full-stack, cloud-native)
    hyphenated = re.findall(r"\b[a-zA-Z0-9]+-[a-zA-Z0-9]+\b", combined_text)
    for hyp in hyphenated:
        vocab.add(hyp)

    # Sort deterministically and limit to max_terms
    return sorted(list(vocab))[:max_terms]
