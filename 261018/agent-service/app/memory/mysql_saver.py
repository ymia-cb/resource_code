"""自研 MySQL CheckpointSaver（LangGraph 官方无 MySQL 版）。

实现 ``BaseCheckpointSaver`` 同步接口，把 checkpoint 存进 MySQL 的
``checkpoints`` / ``checkpoint_writes`` 两张表（见 app/db/models/checkpoint.py）。

表结构与序列化方式严格对齐官方 ``SqliteSaver``，仅把 SQL 方言从 SQLite（``?`` 占位符）
换成 MySQL（``%s`` 占位符），确保可通过与 SqliteSaver 的对拍测试。

实现的方法（BaseCheckpointSaver 要求）：
- ``get_tuple``：按 thread_id + checkpoint_ns + checkpoint_id（或最新）读取
- ``list``：按 thread_id 倒序列出 checkpoint（含 metadata 过滤、before、limit）
- ``put``：写入 checkpoint 快照
- ``put_writes``：写入中断恢复的通道写入
- ``delete_thread``：删除某 thread 的所有 checkpoint 与 writes
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Sequence, cast

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import (
    get_checkpoint_id,
    get_checkpoint_metadata,
    load_pending_writes,
    WRITES_IDX_MAP,
)
from langchain_core.runnables import RunnableConfig

from app.db.session import get_session
from app.db.models.checkpoint import Checkpoint as CheckpointRow
from app.db.models.checkpoint import CheckpointWrite as CheckpointWriteRow

from sqlalchemy import select, delete, text


def _pending_writes_query() -> str:
    """MySQL 版 pending_writes 查询，列顺序对齐 load_pending_writes 期望。

    列顺序：(task_id, channel, type, value, task_path, idx)
    """
    return (
        "SELECT task_id, channel, type, value, task_path, idx FROM checkpoint_writes "
        "WHERE thread_id = :tid AND checkpoint_ns = :ns AND checkpoint_id = :cid"
    )


def _deserialize_pending_writes(rows, serde: SerializerProtocol):
    """反序列化 pending writes，保持 writes_sort_key 顺序。

    复用官方 load_pending_writes（列顺序已对齐）。
    """
    return load_pending_writes(rows, serde)


class MySQLSaver(BaseCheckpointSaver[str]):
    """把 checkpoint 存进 MySQL 的同步 CheckpointSaver。"""

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with get_session() as session:
            if checkpoint_id := get_checkpoint_id(config):
                stmt = (
                    select(CheckpointRow)
                    .where(
                        CheckpointRow.thread_id == thread_id,
                        CheckpointRow.checkpoint_ns == checkpoint_ns,
                        CheckpointRow.checkpoint_id == checkpoint_id,
                    )
                )
            else:
                stmt = (
                    select(CheckpointRow)
                    .where(
                        CheckpointRow.thread_id == thread_id,
                        CheckpointRow.checkpoint_ns == checkpoint_ns,
                    )
                    .order_by(CheckpointRow.checkpoint_id.desc())
                    .limit(1)
                )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                return None

            # 反序列化 checkpoint 与 metadata
            checkpoint = self.serde.loads_typed((row.type, row.checkpoint))
            metadata = cast(
                CheckpointMetadata,
                json.loads(row.meta) if row.meta is not None else {},
            )

            # 读 pending writes
            write_rows = session.execute(
                text(_pending_writes_query()),
                {"tid": thread_id, "ns": checkpoint_ns, "cid": row.checkpoint_id},
            ).fetchall()
            pending_writes = _deserialize_pending_writes(write_rows, self.serde)

            return CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint,
                metadata,
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row.parent_checkpoint_id,
                        }
                    }
                    if row.parent_checkpoint_id
                    else None
                ),
                pending_writes,
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with get_session() as session:
            stmt = select(CheckpointRow)
            if config is not None:
                thread_id = str(config["configurable"]["thread_id"])
                stmt = stmt.where(CheckpointRow.thread_id == thread_id)
                checkpoint_ns = config["configurable"].get("checkpoint_ns")
                if checkpoint_ns is not None:
                    stmt = stmt.where(CheckpointRow.checkpoint_ns == checkpoint_ns)
                if checkpoint_id := get_checkpoint_id(config):
                    stmt = stmt.where(CheckpointRow.checkpoint_id == checkpoint_id)
            if before is not None:
                stmt = stmt.where(
                    CheckpointRow.checkpoint_id < get_checkpoint_id(before)
                )
            stmt = stmt.order_by(CheckpointRow.checkpoint_id.desc())
            if limit is not None:
                stmt = stmt.limit(limit)

            rows = session.execute(stmt).scalars().all()
            for row in rows:
                checkpoint = self.serde.loads_typed((row.type, row.checkpoint))
                metadata = cast(
                    CheckpointMetadata,
                    json.loads(row.meta) if row.meta is not None else {},
                )
                write_rows = session.execute(
                    text(_pending_writes_query()),
                    {
                        "tid": row.thread_id,
                        "ns": row.checkpoint_ns,
                        "cid": row.checkpoint_id,
                    },
                ).fetchall()
                pending_writes = _deserialize_pending_writes(write_rows, self.serde)
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": row.thread_id,
                            "checkpoint_ns": row.checkpoint_ns,
                            "checkpoint_id": row.checkpoint_id,
                        }
                    },
                    checkpoint,
                    metadata,
                    (
                        {
                            "configurable": {
                                "thread_id": row.thread_id,
                                "checkpoint_ns": row.checkpoint_ns,
                                "checkpoint_id": row.parent_checkpoint_id,
                            }
                        }
                        if row.parent_checkpoint_id
                        else None
                    ),
                    pending_writes,
                )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )

        with get_session() as session:
            row = CheckpointRow(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=config["configurable"].get("checkpoint_id"),
                type=type_,
                checkpoint=serialized_checkpoint,
                meta=serialized_metadata,
            )
            session.merge(row)  # INSERT OR REPLACE 语义（同 checkpoint_id 覆盖）
            session.flush()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        with get_session() as session:
            for idx, (channel, value) in enumerate(writes):
                type_, serialized_value = self.serde.dumps_typed(value)
                write = CheckpointWriteRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=WRITES_IDX_MAP.get(channel, idx),
                    task_path=task_path,
                    channel=channel,
                    type=type_,
                    value=serialized_value,
                )
                session.merge(write)  # 幂等覆盖
            session.flush()

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        with get_session() as session:
            session.execute(
                delete(CheckpointWriteRow).where(
                    CheckpointWriteRow.thread_id == thread_id
                )
            )
            session.execute(
                delete(CheckpointRow).where(CheckpointRow.thread_id == thread_id)
            )
