import os
from typing import Literal, Optional
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
    VISION_FPS: int = 1
    VISION_ANOMALY_DURATION_MS: int = 3000
    VISION_NORMAL_RESET_MS: int = 1000
    PRE_FILTER_SCORE_CUTOFF: float = 45.0
    MAX_VLM_CALLS_PER_SESSION: int = 12
    VLM_COOLDOWN_SEC: int = 30
    STREAM_MAXLEN_AUDIO: int = 5000
    STREAM_MAXLEN_VIDEO: int = 300


settings = Settings()
