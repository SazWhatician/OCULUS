import os
from typing import Literal, Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime & Deployment Mode
    DEPLOY_MODE: Literal["single_node", "distributed"] = "single_node"
    APP_ENV: str = "development"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "t", "yes", "debug")
        return bool(v)

    # Provider Switches
    INGESTION_PROVIDER: Literal["demo_file", "browser_webrtc", "attendee_bot"] = "demo_file"
    STT_PROVIDER: Literal["hybrid", "speechmatics", "groq"] = "hybrid"
    LLM_PRIMARY_PROVIDER: Literal["groq", "gemini"] = "groq"
    STORAGE_PROVIDER: Literal["local", "r2"] = "local"

    # Database & Redis Bus
    # Default SQLite aiosqlite for zero-dependency local dev/demo, seamlessly switches to Postgres
    DATABASE_URL: str = "sqlite+aiosqlite:///./oculus.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # API Keys & Endpoints (All optional with graceful mock fallback)
    SPEECHMATICS_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ATTENDEE_API_KEY: str = ""
    ATTENDEE_WEBHOOK_SECRET: str = ""
    ATTENDEE_BASE_URL: str = "https://app.attendee.dev"
    PUBLIC_API_URL: str = "http://localhost:8000"

    # Cloudflare R2 Object Storage
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "oculus-media"
    LOCAL_STORAGE_DIR: str = "./tmp_storage"

    # Numeric Engineering Thresholds
    STITCH_GAP_MAX_MS: int = 2500
    STITCH_HARD_GAP_MAX_MS: int = 6500
    STT_IDLE_FLUSH_SECONDS: float = 8.0
    MEDIAPIPE_FACE_LANDMARKER_MODEL: Optional[str] = None
    OPENCV_YUNET_MODEL: Optional[str] = None
    ENABLE_VISION_MODEL_AUTO_DOWNLOAD: bool = True
    VISION_FPS: int = 1
    VISION_ANOMALY_DURATION_MS: int = 3000
    VISION_NORMAL_RESET_MS: int = 1000
    PRE_FILTER_SCORE_CUTOFF: float = 45.0
    MAX_VLM_CALLS_PER_SESSION: int = 12
    VLM_COOLDOWN_SEC: int = 30
    STREAM_MAXLEN_AUDIO: int = 5000
    STREAM_MAXLEN_VIDEO: int = 300

    # LLM, STT, Proctoring & Webhook Tuning Overrides
    LLM_ALLOW_MOCK_FALLBACK: bool = True
    LLM_PROVIDER_TIMEOUT_SECONDS: float = 22.0
    LLM_HIGH_RISK_CONSENSUS: bool = False
    GROQ_PRIMARY_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FALLBACK_MODELS: str = "openai/gpt-oss-20b,qwen/qwen3.8-27b"
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
    GROQ_MAX_RETRIES: int = 2
    GEMINI_TEXT_MODEL: str = "gemini-3.8-flash"
    GEMINI_TEXT_FALLBACK_MODELS: str = "gemini-3.5-flash-lite,gemini-2.5-flash-lite"
    GEMINI_VLM_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_VLM_FALLBACK_MODELS: str = "gemini-3.8-flash,gemini-2.5-flash-lite"
    GEMINI_MAX_RETRIES: int = 2
    GEMINI_MAX_VLM_CALLS_PER_SESSION: int = 12
    STT_ALLOW_MOCK_FALLBACK: bool = True
    STT_PROVIDER_TIMEOUT_SECONDS: float = 60.0
    SPEECHMATICS_BASE_URL: str = "https://asr.api.speechmatics.com/v2"
    SPEECHMATICS_OPERATING_POINT: str = "enhanced"
    SPEECHMATICS_LANGUAGE: str = "en"
    SPEECHMATICS_POLL_TIMEOUT_SEC: float = 45.0
    SPEECHMATICS_MAX_RETRIES: int = 2
    ATTENDEE_WEBHOOK_ALLOW_INSECURE_DEV: bool = False
    PROCTORING_MAX_LLM_TURNS: int = 8
    PROCTORING_LLM_CONCURRENCY: int = 2
    SCORING_MAX_LLM_TURNS: int = 8
    SCORING_LLM_CONCURRENCY: int = 1
    SCORING_IDLE_FLUSH_SECONDS: float = 18.0
    SCORING_SESSION_TTL_SECONDS: float = 900.0


settings = Settings()
