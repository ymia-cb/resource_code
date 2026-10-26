"""聊天消息持久化模型（异步 SSE 流式的落库真源）。

设计：
- 一条用户消息 = 一条 ``ChatMessage``（message_id 主键，thread_id 定位会话）。
- ``final_answer`` / ``sources`` / ``intent`` 等为 Agent 跑完后的**最终完整结果**，
  是前端「刷新页面 / 断网恢复后请求一次拉取」的真源。
- Redis 里的 chunk 仅为实时推流的临时态，不落库、可丢失。
"""
from __future__ import annotations

from enum import Enum

from sqlalchemy import Column, String, Text, Integer, JSON, DateTime, func, Index

from app.db.base import Base


class ChatMessageStatus(str, Enum):
    RUNNING = "running"  # Agent 后台异步执行中
    DONE = "done"        # 已完成，final_answer 就绪
    FAILED = "failed"    # 执行失败，error_code 就绪


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_thread", "thread_id"),
    )

    message_id = Column(String(64), primary_key=True)
    thread_id = Column(String(64), nullable=False)
    role = Column(String(16), nullable=False, default="user")  # user / assistant
    user_text = Column(Text, nullable=True)  # 用户原始输入（user 消息）
    status = Column(String(16), nullable=False, default=ChatMessageStatus.RUNNING.value)

    # 最终完整结果（Agent 跑完后写入）
    final_answer = Column(Text, nullable=True)
    sources = Column(JSON, nullable=True)  # [{chunk_id, text, source}]
    intent = Column(String(32), nullable=True)  # simple / complex / human / unknown
    needs_human = Column(String(1), nullable=True)  # "1" / "0"
    degraded = Column(String(1), nullable=True)  # "1" / "0"
    error_code = Column(String(64), nullable=True)  # 稳定错误码（失败时）

    # 流式元信息
    chunk_count = Column(Integer, nullable=False, default=0)  # 已产出 chunk 数
    run_id = Column(String(64), nullable=True)  # 关联 AgentRun

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
