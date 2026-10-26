"""FastAPI 依赖注入：进程级单例 graph。

避免每次请求重建 graph / checkpointer（checkpointer 需持长连接）。
"""
from __future__ import annotations

from functools import lru_cache

from app.graph.build import build_graph
from app.memory.checkpointer import get_checkpointer


@lru_cache(maxsize=1)
def get_graph():
    """进程级单例：编译一次 graph，复用 checkpointer。"""
    return build_graph(get_checkpointer())
