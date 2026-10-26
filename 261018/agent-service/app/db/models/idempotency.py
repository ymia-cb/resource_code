"""幂等记录表（IdempotencyRecord）。

写操作（cancel_order / apply_refund / apply_return / update_address）执行前写入幂等键，
并发/重复请求命中同一键时返回已有结果，不重复执行（防重复扣款/重复取消）。
"""
from __future__ import annotations

from sqlalchemy import Column, String, BigInteger, JSON, DateTime, func, Index

from app.db.base import Base


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        Index("ix_idem_key", "idempotency_key", unique=True),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    idempotency_key = Column(String(128), nullable=False)  # 幂等键（tool_name + thread_id + 业务键哈希）
    tool_name = Column(String(64), nullable=False)
    thread_id = Column(String(64), nullable=False, index=True)
    result = Column(JSON, nullable=False)  # 首次执行结果（重复请求直接返回）
    created_at = Column(DateTime, server_default=func.now())
