"""幂等校验：写操作防重复执行（Spec 4.3 闸门④）。

- ``make_idempotency_key``：由 tool_name + thread_id + 业务键哈希生成唯一键。
  业务键由 ToolSpec.idempotent_keys 声明（如 update_address 含 address，
  实现「同地址去重、异地址重写」）。
- ``check_idempotency``：命中已有键 → 返回已有结果（幂等命中是正常业务结果，
  不再抛异常；由调用方判断是否跳过执行）。
- ``record_idempotency``：执行成功后写入结果（供重复请求复用）。

并发安全：依赖 ``idempotency_records.ix_idem_key`` 唯一索引，并发插入同一键时
后者触发 IntegrityError，回读已有结果返回（不静默吞掉）。
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models.idempotency import IdempotencyRecord
from app.db.session import get_session


def make_idempotency_key(
    tool_name: str, thread_id: str, params: dict, key_fields: tuple[str, ...] | None = None
) -> str:
    """生成幂等键：工具名 + 会话 + 业务参数哈希。

    Args:
        key_fields: 参与幂等键的业务字段白名单（来自 ToolSpec.idempotent_keys）。
            默认取 order_id 等稳定标识。排除易变字段（如 reason）以允许重试。
    """
    if key_fields is None:
        key_fields = ("order_id", "user_id", "address_id", "coupon_id", "product_id", "sku")
    stable = {k: v for k, v in params.items() if k in key_fields}
    payload = json.dumps(
        {"tool": tool_name, "thread": thread_id, "params": stable},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_idempotency(
    tool_name: str, thread_id: str, params: dict, key_fields: tuple[str, ...] | None = None
) -> dict | None:
    """闸门④：命中已有幂等键 → 返回已有结果（不抛异常）。

    Returns:
        None = 未命中（应继续执行）；dict = 已有结果（幂等命中，应跳过执行）。
    """
    key = make_idempotency_key(tool_name, thread_id, params, key_fields)
    with get_session() as session:
        record = session.execute(
            select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == key)
        ).scalar_one_or_none()
        if record is not None:
            return dict(record.result) if record.result is not None else None
    return None


def record_idempotency(
    tool_name: str, thread_id: str, params: dict, result: dict, key_fields: tuple[str, ...] | None = None
) -> dict | None:
    """执行成功后记录幂等键与结果。

    并发冲突（另一请求已写入同一键）→ 回读已有结果返回（不静默吞掉）。
    Returns:
        None = 正常写入；dict = 并发下回读到的已有结果（说明重复执行了）。
    """
    key = make_idempotency_key(tool_name, thread_id, params, key_fields)
    with get_session() as session:
        record = IdempotencyRecord(
            idempotency_key=key,
            tool_name=tool_name,
            thread_id=thread_id,
            result=result,
        )
        session.add(record)
        try:
            session.flush()
        except IntegrityError:
            # 并发重复：唯一索引冲突，回读已有结果返回
            session.rollback()
            existing = session.execute(
                select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == key)
            ).scalar_one_or_none()
            if existing is not None:
                return dict(existing.result) if existing.result is not None else {}
    return None

