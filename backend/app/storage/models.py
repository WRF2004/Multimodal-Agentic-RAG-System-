"""
Database ORM models for persistent state.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer,
    Boolean, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from app.storage.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    title = Column(String(256), default="New Chat")
    config_overrides = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)
    tool_calls = Column(JSON, nullable=True)
    tool_call_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="messages")


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending | processing | completed | failed
    error_message = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    dataset_name = Column(String(256), nullable=False)
    config_snapshot = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)