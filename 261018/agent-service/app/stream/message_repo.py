"""消息仓库：chat_messages 表的持久化读写。

异步 SSE 流式链路里，这是「最终完整结果」的落库真源：
- 提交消息 → 插入 running 记录
- Agent 跑完 → 更新为 done + final_answer/sources 等
- 失败 → 更新为 failed + error_code
- 前端刷新/断线 → 按 message_id 拉取完整结果
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.chat_message import ChatMessage, ChatMessageStatus


def new_message_id() -> str:
    return uuid.uuid4().hex


def create_message(
    session: Session,
    thread_id: str,
    user_text: str,
    message_id: str | None = None,
    role: str = "user",
) -> str:
    """插入一条消息（初始 running）。返回 message_id。"""
    mid = message_id or new_message_id()
    msg = ChatMessage(
        message_id=mid,
        thread_id=thread_id,
        role=role,
        user_text=user_text,
        status=ChatMessageStatus.RUNNING.value,
        chunk_count=0,
    )
    session.add(msg)
    session.commit()
    return mid


def update_message_done(
    session: Session,
    message_id: str,
    final_answer: str,
    sources: list[dict] | None = None,
    intent: str | None = None,
    needs_human: bool = False,
    degraded: bool = False,
    chunk_count: int = 0,
) -> None:
    """Agent 跑完 → 更新为 done。"""
    msg = session.get(ChatMessage, message_id)
    if msg is None:
        return
    msg.status = ChatMessageStatus.DONE.value
    msg.final_answer = final_answer
    msg.sources = sources or []
    msg.intent = intent
    msg.needs_human = "1" if needs_human else "0"
    msg.degraded = "1" if degraded else "0"
    msg.chunk_count = chunk_count
    session.commit()


def update_message_failed(
    session: Session,
    message_id: str,
    error_code: str,
) -> None:
    """Agent 失败 → 更新为 failed。"""
    msg = session.get(ChatMessage, message_id)
    if msg is None:
        return
    msg.status = ChatMessageStatus.FAILED.value
    msg.error_code = error_code
    session.commit()


def get_message(session: Session, message_id: str) -> ChatMessage | None:
    return session.get(ChatMessage, message_id)


def list_messages(session: Session, thread_id: str) -> list[ChatMessage]:
    """按 thread_id 列出该会话所有消息（按创建时间升序）。"""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return list(session.scalars(stmt).all())


def to_dict(msg: ChatMessage) -> dict:
    """ORM → 前端友好的 dict。"""
    return {
        "message_id": msg.message_id,
        "thread_id": msg.thread_id,
        "role": msg.role,
        "user_text": msg.user_text,
        "status": msg.status,
        "final_answer": msg.final_answer,
        "sources": msg.sources or [],
        "intent": msg.intent,
        "needs_human": msg.needs_human == "1",
        "degraded": msg.degraded == "1",
        "error_code": msg.error_code,
        "chunk_count": msg.chunk_count,
        "run_id": msg.run_id,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }
