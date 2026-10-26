"""命令执行 + 步骤状态持久化（供审计/恢复/排查）。

对应 grill 会话收敛的「聚合命令要有幂等键 + 持久化命令及其步骤状态」：

- ``CommandExecution``：一次命令编排执行（可含多个步骤），记录命令级幂等键、
  整体状态、结果。超时重试靠命令级幂等键命中，跳过已执行命令。
- ``CommandStep``：命令内的每个子步骤（单操作/聚合子写），记录 pending/success/
  failed 状态，供「部分完成时逐项告知」与「恢复时跳过已完成步骤」。

弱关联批量命令允许部分完成：命令级状态可为 partial，步骤级各自记录成败。
"""
from __future__ import annotations

from sqlalchemy import Column, String, BigInteger, JSON, DateTime, func, Index

from app.db.base import Base


class CommandStatus(str):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"  # 部分完成（弱关联批量，允许部分成功）
    FAILED = "failed"


class StepStatus(str):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class CommandExecution(Base):
    __tablename__ = "command_executions"
    __table_args__ = (
        Index("ix_cmd_exec_run", "run_id"),
        Index("ix_cmd_exec_key", "command_key", unique=True),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    command_key = Column(String(128), nullable=False)  # 命令级幂等键（防超时重试重复）
    command_name = Column(String(64), nullable=False)  # 命令名（如 cancel_order）
    status = Column(String(16), nullable=False, default=CommandStatus.RUNNING)
    result = Column(JSON, nullable=True)  # 命令整体结果
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class CommandStep(Base):
    __tablename__ = "command_steps"
    __table_args__ = (
        Index("ix_cmd_step_exec", "command_execution_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    command_execution_id = Column(BigInteger, nullable=False)  # 关联 CommandExecution.id
    step_index = Column(BigInteger, nullable=False)  # 步骤序号
    tool_name = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default=StepStatus.PENDING)
    result = Column(JSON, nullable=True)
    error = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
