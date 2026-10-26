"""checkpoint 初始化。

默认使用自研 MySQL CheckpointSaver（对齐官方 SqliteSaver 接口，已通过对拍测试）。
可通过环境变量 ``CHECKPOINT_BACKEND=sqlite`` 回退到官方 SqliteSaver。
"""
from __future__ import annotations

import os

from app.memory.mysql_saver import MySQLSaver


def get_checkpointer(db_path: str = "checkpoints.db"):
    """初始化 checkpointer。

    - 默认：MySQLSaver（存 MySQL 的 checkpoints/checkpoint_writes 表）
    - CHECKPOINT_BACKEND=sqlite 时：官方 SqliteSaver（用于对拍/本地调试）
    """
    backend = os.getenv("CHECKPOINT_BACKEND", "mysql").lower()
    if backend == "sqlite":
        import sqlite3
        from pathlib import Path
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(conn)
    return MySQLSaver()
