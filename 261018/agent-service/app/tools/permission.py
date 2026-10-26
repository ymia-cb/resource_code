"""权限校验：thread_id → user_id 映射（Spec 4.3 闸门③）。

从 sessions 表读取 thread 归属的 user，强制租户隔离。
"""
from __future__ import annotations

from sqlalchemy import select

from app.db.models.session import Session
from app.db.session import get_session
from app.tools.errors import PermissionDenied


def resolve_user(thread_id: str) -> str:
    """由 thread_id 解析 user_id。

    未找到会话（thread 不存在）→ 抛 PermissionDenied。
    """
    with get_session() as session:
        row = session.execute(
            select(Session.user_id).where(Session.thread_id == thread_id)
        ).scalar_one_or_none()
    if row is None:
        raise PermissionDenied(f"无法解析 thread_id 归属: {thread_id}")
    return row


def ensure_session(thread_id: str, user_id: str) -> None:
    """确保会话存在且归属正确（用于 HTTP 层 / 写操作前置校验）。

    不存在则创建，存在但 user_id 不一致则抛 PermissionDenied。
    """
    with get_session() as session:
        row = session.execute(
            select(Session).where(Session.thread_id == thread_id)
        ).scalar_one_or_none()
        if row is None:
            session.add(Session(thread_id=thread_id, user_id=user_id, status="active"))
        elif row.user_id != user_id:
            raise PermissionDenied(
                f"thread_id {thread_id} 归属 {row.user_id}，与请求 {user_id} 不符"
            )
