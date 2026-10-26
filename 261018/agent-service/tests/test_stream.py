"""流式链路测试：pub/sub 推流 + 消息落库 + 异步 worker 节点级流式。

运行（不依赖真实 MySQL/Redis）：
    python -m pytest tests/test_stream.py -v

说明：
- pub/sub 走内存降级路径（未配置 REDIS_URL 时）。
- 消息仓库用 SQLite 内存库隔离（不依赖真实 MySQL）。
- worker 测试用 CHECKPOINT_BACKEND=sqlite 跑 graph.astream（不依赖 MySQL checkpoint）。
"""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.stream.pubsub import channel_for, _InMemoryPubSub
from app.stream import message_repo


# ---------------------------------------------------------------------------
# pub/sub 内存降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_pubsub_roundtrip():
    ps = _InMemoryPubSub()
    received: list[str] = []

    async def consumer():
        async for msg in ps.subscribe("ch1"):
            received.append(msg)
            break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)  # 让订阅先就位
    await ps.publish("ch1", {"type": "node", "chunk_id": 1})
    await asyncio.wait_for(task, timeout=1)
    assert len(received) == 1
    assert '"chunk_id": 1' in received[0] or '"chunk_id":1' in received[0]


@pytest.mark.asyncio
async def test_inmemory_pubsub_no_cross_channel():
    ps = _InMemoryPubSub()
    got_other: list[str] = []

    async def consumer_other():
        async for msg in ps.subscribe("other"):
            got_other.append(msg)
            break

    task = asyncio.create_task(consumer_other())
    await asyncio.sleep(0.01)
    await ps.publish("ch1", {"type": "node", "chunk_id": 1})
    # ch1 的消息不应到 other 订阅者；任务应仍在等待
    await asyncio.sleep(0.02)
    assert got_other == []
    task.cancel()


def test_channel_for():
    assert channel_for("t1", "m1") == "t1:m1"


# ---------------------------------------------------------------------------
# 消息仓库（SQLite 内存）
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        yield s


@pytest.fixture()
def sqlite_session_factory():
    """返回一个 context-manager 风格的 session 工厂（每次调用新建一个 session）。"""
    from contextlib import contextmanager

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def factory():
        with Session() as s:
            yield s

    return factory


def test_message_repo_lifecycle(sqlite_session):
    s = sqlite_session
    mid = message_repo.create_message(s, thread_id="t1", user_text="退款政策是什么")
    assert mid

    # running 态
    msg = message_repo.get_message(s, mid)
    assert msg is not None
    assert msg.status == "running"
    assert msg.user_text == "退款政策是什么"

    # 更新为 done
    message_repo.update_message_done(
        s, mid, final_answer="退款政策是 7 天无理由",
        sources=[{"chunk_id": "c1", "text": "政策原文"}],
        intent="knowledge", needs_human=False, degraded=False, chunk_count=3,
    )
    msg = message_repo.get_message(s, mid)
    assert msg.status == "done"
    assert msg.final_answer == "退款政策是 7 天无理由"
    assert msg.sources == [{"chunk_id": "c1", "text": "政策原文"}]
    assert msg.chunk_count == 3

    # 列表
    msgs = message_repo.list_messages(s, "t1")
    assert len(msgs) == 1
    assert msgs[0].message_id == mid

    # to_dict
    d = message_repo.to_dict(msg)
    assert d["needs_human"] is False
    assert d["degraded"] is False
    assert d["status"] == "done"


def test_message_repo_failed(sqlite_session):
    s = sqlite_session
    mid = message_repo.create_message(s, thread_id="t2", user_text="我要取消订单")
    message_repo.update_message_failed(s, mid, error_code="E_INTERNAL")
    msg = message_repo.get_message(s, mid)
    assert msg.status == "failed"
    assert msg.error_code == "E_INTERNAL"


# ---------------------------------------------------------------------------
# worker：节点级流式（用 sqlite checkpoint + 内存 pub/sub）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stream_agent_end_to_end(monkeypatch, sqlite_session_factory):
    """端到端：后台跑 graph.astream → 逐节点推流 → 落库 done。

    用内存 pub/sub 订阅 channel，收集所有 node chunk；结束后验证落库结果。
    """
    import os
    monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")

    from app.stream.worker import run_stream_agent
    from app.stream.pubsub import get_pubsub, channel_for

    # 强制重建 graph（用 sqlite checkpointer）—— get_graph 是 lru_cache，需清缓存
    from app.api import deps
    deps.get_graph.cache_clear()

    thread_id = "stream-t1"
    message_id = "stream-m1"
    run_id = "stream-r1-" + str(abs(hash(thread_id)))

    # 先落一条 running 消息（用注入的 sqlite session factory）
    with sqlite_session_factory() as s:
        message_repo.create_message(s, thread_id, "退款政策是什么", message_id=message_id)

    pubsub = get_pubsub()
    channel = channel_for(thread_id, message_id)
    collected: list[dict] = []

    async def consumer():
        async for raw in pubsub.subscribe(channel):
            import json
            collected.append(json.loads(raw))
            if collected[-1].get("type") in ("done", "error"):
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    await run_stream_agent(thread_id, message_id, "退款政策是什么", run_id, session_factory=sqlite_session_factory)
    await asyncio.wait_for(task, timeout=5)

    # 至少一个 node chunk + 最终 done
    assert any(c["type"] == "node" for c in collected)
    assert collected[-1]["type"] == "done"

    # done chunk 带 final_answer
    done = collected[-1]
    assert "final_answer" in done

    # 落库状态应为 done
    with sqlite_session_factory() as s:
        msg = message_repo.get_message(s, message_id)
        assert msg is not None
        assert msg.status == "done"
