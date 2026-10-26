"""Agent Run 追踪 + 工具执行状态（轻量版 Run 级追踪）。

设计（轻量版，不做全节点断点续跑）：
- ``AgentRun``：一次 /chat 请求 = 一个 run，记录 run_id / thread_id / 状态 / 错误码。
- ``ToolExecution``：每个工具调用记录一次执行状态（关联 run_id + tool_name + 结果）。

「从持久化状态恢复、不从头重放」的实现：写操作（cancel/refund/return/address）
执行后结果已持久化在 ``idempotency_records``（含 result），Run 重试时靠幂等键
``DuplicateRequest`` 命中已有结果、跳过已执行写操作，天然避免重放副作用。
AgentRun / ToolExecution 提供 Run 级追踪维度（run_id 关联工具调用），供指标/日志。
"""
from __future__ import annotations

from enum import Enum

from sqlalchemy import Column, String, BigInteger, JSON, DateTime, func, Index

from app.db.base import Base


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"  # 无法可靠完成，需重试或转人工


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_thread", "thread_id"),
    )

    run_id = Column(String(64), primary_key=True)
    thread_id = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default=RunStatus.RUNNING.value)
    error_code = Column(String(64), nullable=True)  # 稳定错误码（失败时）
    error_category = Column(String(32), nullable=True)  # 错误分类
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ToolExecution(Base):
    __tablename__ = "tool_executions"
    __table_args__ = (
        Index("ix_tool_exec_run", "run_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    tool_name = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)  # success / error / need_approval / need_confirm
    result = Column(JSON, nullable=True)  # 工具返回结果
    error = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
