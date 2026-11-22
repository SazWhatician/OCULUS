import json
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

Base = declarative_base()


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)
    candidate_name = Column(String(255), nullable=False)
    interviewer_name = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=False)
    job_description = Column(Text, nullable=True)
    custom_vocab = Column(JSON, default=list)
    ingestion_mode = Column(String(50), default="demo_file")
    status = Column(String(50), default="CREATED")
    bot_external_id = Column(String(128), nullable=True, index=True)
    speaker_role_map = Column(JSON, default=dict)  # e.g. {"S1": "interviewer", "S2": "candidate"}
    baseline_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    utterances = relationship("UtteranceModel", back_populates="session", cascade="all, delete-orphan")
    vision_events = relationship("VisionEventModel", back_populates="session", cascade="all, delete-orphan")
    qa_evaluations = relationship("QAEvaluationModel", back_populates="session", cascade="all, delete-orphan")


class UtteranceModel(Base):
    __tablename__ = "utterances"

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    speaker = Column(String(50), nullable=False)
    speaker_label = Column(String(50), nullable=False)
    text = Column(Text, nullable=False)
    start_ms = Column(Integer, nullable=False, index=True)
    end_ms = Column(Integer, nullable=False)
    words_json = Column(JSON, default=list)

    session = relationship("SessionModel", back_populates="utterances")


class VisionEventModel(Base):
    __tablename__ = "vision_events"

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp_ms = Column(Integer, nullable=False, index=True)
    anomaly_type = Column(String(50), nullable=False)
    duration_ms = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    pitch_deg = Column(Float, nullable=True)
    yaw_deg = Column(Float, nullable=True)
    evidence_frame_key = Column(String(255), nullable=True)
    vlm_verification_note = Column(Text, nullable=True)

    session = relationship("SessionModel", back_populates="vision_events")


class QAEvaluationModel(Base):
    __tablename__ = "qa_evaluations"

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    qa_index = Column(Integer, nullable=False)
    combined_risk_score = Column(Float, nullable=False)
    payload_json = Column(JSON, nullable=False)

    session = relationship("SessionModel", back_populates="qa_evaluations")


class IntegrityReportModel(Base):
    __tablename__ = "integrity_reports"

    session_id = Column(String(64), primary_key=True)
    overall_risk_score = Column(Float, nullable=False)
    risk_band = Column(String(50), nullable=False)
    report_json = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# Engine and Session initialization
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
