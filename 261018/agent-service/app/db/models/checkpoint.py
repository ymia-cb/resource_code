"""checkpoint 表（自研 MySQL CheckpointSaver 专用）。

表结构对齐 LangGraph 官方 SqliteSaver 的字段约定，仅把数据库从 SQLite 换成 MySQL：

- ``checkpoints``：thread_id + checkpoint_ns + checkpoint_id 联合主键，
  存序列化 checkpoint（BLOB）+ type 标记 + metadata（JSON）+ parent_checkpoint_id。
- ``checkpoint_writes``：thread_id + checkpoint_ns + checkpoint_id + task_id + idx
  联合主键，存中断恢复的通道写入（channel + type + value + task_path）。
"""
from __future__ import annotations

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    JSON,
    LargeBinary,
    func,
    Index,
)

from app.db.base import Base


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    __table_args__ = (
        Index("ix_checkpoints_thread_created", "thread_id", "created_at"),
    )

    thread_id = Column(String(64), primary_key=True)
    checkpoint_ns = Column(String(128), primary_key=True, default="")
    checkpoint_id = Column(String(64), primary_key=True)
    parent_checkpoint_id = Column(String(64), nullable=True)  # 父 checkpoint（分支/时间旅行）
    type = Column(String(64), nullable=True)  # 序列化类型标记（serde.dumps_typed 返回）
    checkpoint = Column(LargeBinary, nullable=False)  # 序列化 State
    # 列名 metadata，属性名 meta 避开 Declarative 保留字
    meta = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class CheckpointWrite(Base):
    __tablename__ = "checkpoint_writes"

    thread_id = Column(String(64), primary_key=True)
    checkpoint_ns = Column(String(128), primary_key=True, default="")
    checkpoint_id = Column(String(64), primary_key=True)
    task_id = Column(String(64), primary_key=True)
    idx = Column(Integer, primary_key=True)  # 写入顺序（writes_sort_key 用）
    task_path = Column(String(255), nullable=False, default="")
    channel = Column(String(128), nullable=False)
    type = Column(String(64), nullable=True)  # 序列化类型标记
    value = Column(LargeBinary, nullable=True)  # 通道写入值（中断恢复用）
