"""审计日志表（AuditLog）。

记录工具调用（尤其写操作）的入参/出参/改前值/改后值，供追溯与合规审计。
"""
from __future__ import annotations

from sqlalchemy import Column, BigInteger, String, DateTime, JSON, func

from app.db.base import Base


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id = Column(String(64), nullable=False, index=True)
    tool_name = Column(String(64), nullable=False, index=True)
    action = Column(String(64), nullable=False)
    params = Column(JSON, nullable=False)  # 入参
    result = Column(JSON, nullable=False)  # 出参
    before_value = Column(JSON, nullable=True)  # 改前值（写操作）
    after_value = Column(JSON, nullable=True)  # 改后值（写操作）
    created_at = Column(DateTime, server_default=func.now())
