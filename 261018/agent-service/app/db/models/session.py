"""会话表（Session）。

对应 LangGraph 的 thread_id，用于隔离多用户多会话的 checkpoint 记忆。
"""
from __future__ import annotations

from sqlalchemy import Column, String, DateTime, func

from app.db.base import Base


class Session(Base):
    __tablename__ = "sessions"

    thread_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    # active / human / closed
    status = Column(String(16), nullable=False, default="active")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
