"""ORM 数据模型统一导出。

Alembic autogenerate 依赖 ``Base.metadata`` 收集所有模型，因此必须在 ``models/__init__.py``
里导入全部模型，确保 metadata 完整。
"""
from app.db.models.session import Session
from app.db.models.audit import AuditLog
from app.db.models.checkpoint import Checkpoint, CheckpointWrite
from app.db.models.order import (
    Order,
    OrderItem,
    OrderStatus,
    RefundStatus,
    ReturnStatus,
)
from app.db.models.idempotency import IdempotencyRecord
from app.db.models.knowledge import KnowledgeDoc, KnowledgeDocStatus
from app.db.models.agent_run import AgentRun, ToolExecution, RunStatus
from app.db.models.command import (
    CommandExecution,
    CommandStep,
    CommandStatus,
    StepStatus,
)
from app.db.models.chat_message import ChatMessage, ChatMessageStatus

__all__ = [
    "Session",
    "AuditLog",
    "Checkpoint",
    "CheckpointWrite",
    "Order",
    "OrderItem",
    "OrderStatus",
    "RefundStatus",
    "ReturnStatus",
    "IdempotencyRecord",
    "KnowledgeDoc",
    "KnowledgeDocStatus",
    "AgentRun",
    "ToolExecution",
    "RunStatus",
    "CommandExecution",
    "CommandStep",
    "CommandStatus",
    "StepStatus",
    "ChatMessage",
    "ChatMessageStatus",
]
