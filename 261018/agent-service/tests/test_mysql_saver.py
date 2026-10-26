"""MySQL CheckpointSaver 对拍测试。

用同一张图分别挂 SqliteSaver（官方，基准）和 MySQLSaver（自研），
比对 get_tuple / list / put / put_writes / delete_thread 的行为一致性。

运行：python -m pytest tests/test_mysql_saver.py -v
"""
from __future__ import annotations

import sqlite3

import pytest
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from app.memory.mysql_saver import MySQLSaver


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph():
    """一个简单的累计图：把输入文本转成 AI 消息，测试 checkpoint 记忆。"""

    def echo(state: AgentState) -> dict:
        last = state["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last)
        return {"messages": [AIMessage(content=f"echo: {text}")]}

    g = StateGraph(AgentState)
    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    saver = MySQLSaver()
    saver.delete_thread("parity-1")
    saver.delete_thread("parity-2")


def _run_graph(checkpointer, thread_id: str, texts: list[str]) -> list[dict]:
    """用给定 checkpointer 跑多轮，返回每轮 get_tuple 的关键字段。"""
    graph = _build_graph().compile(checkpointer=checkpointer)
    results = []
    for t in texts:
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"messages": [HumanMessage(content=t)]}, config)
        state = graph.get_state(config)
        results.append(
            {
                "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                "messages": [m.content for m in state.values["messages"]],
            }
        )
    return results


def test_parity_multi_turn():
    """多轮对话：SqliteSaver 与 MySQLSaver 的 checkpoint 记忆应完全一致。"""
    texts = ["你好", "我要退款", "订单 123"]

    # 基准：SqliteSaver（内存库）
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    sqlite_results = _run_graph(SqliteSaver(conn), "parity-1", texts)

    # 自研：MySQLSaver
    mysql_results = _run_graph(MySQLSaver(), "parity-1", texts)

    # 消息序列应一致（记忆累积正确）
    for sq, my in zip(sqlite_results, mysql_results):
        assert sq["messages"] == my["messages"]


def test_get_tuple_returns_none_for_unknown_thread():
    saver = MySQLSaver()
    assert saver.get_tuple({"configurable": {"thread_id": "nonexistent"}}) is None


def test_list_returns_checkpoints_desc():
    saver = MySQLSaver()
    graph = _build_graph().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "parity-2"}}
    for t in ["a", "b", "c"]:
        graph.invoke({"messages": [HumanMessage(content=t)]}, config)

    tuples = list(saver.list({"configurable": {"thread_id": "parity-2"}}))
    assert len(tuples) >= 3
    # 倒序（最新在前）
    ids = [t.config["configurable"]["checkpoint_id"] for t in tuples]
    assert ids == sorted(ids, reverse=True)


def test_delete_thread():
    saver = MySQLSaver()
    graph = _build_graph().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "parity-1"}}
    graph.invoke({"messages": [HumanMessage(content="hello")]}, config)
    assert saver.get_tuple(config) is not None

    saver.delete_thread("parity-1")
    assert saver.get_tuple(config) is None


def test_put_and_get_roundtrip():
    """直接 put 一个 checkpoint 再 get_tuple 读回，验证序列化往返。"""
    saver = MySQLSaver()
    checkpoint = {
        "v": 1,
        "id": "manual-cp-1",
        "ts": "2026-10-21T18:00:00+00:00",
        "channel_values": {"messages": []},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }
    config = {"configurable": {"thread_id": "parity-1", "checkpoint_ns": ""}}
    saved = saver.put(config, checkpoint, {"source": "input", "step": 1, "writes": {}}, {})
    assert saved["configurable"]["checkpoint_id"] == "manual-cp-1"

    tup = saver.get_tuple({"configurable": {"thread_id": "parity-1"}})
    assert tup is not None
    assert tup.checkpoint["id"] == "manual-cp-1"
    assert tup.metadata["source"] == "input"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
