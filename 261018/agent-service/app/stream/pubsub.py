"""Redis 推流层（异步 SSE 流式的实时推送通道）。

定位：**Redis 只做推流，不做持久化**。大模型产出的 chunk 通过 Redis
Pub/Sub 推送给订阅中的前端；完整结果落库（``chat_messages``），前端刷新/
断线后靠请求一次拉取完整结果兜底，不依赖 Redis 补齐。

降级策略（与项目「未配置自动降级」哲学一致）：
- 配了 ``REDIS_URL`` → 用 Redis Pub/Sub（channel = ``thread_id:message_id``）。
- 未配 → 降级为进程内 ``asyncio`` 内存 pub/sub（单实例可用，离线可跑、测试不依赖真实 Redis）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


def _redis_url() -> str | None:
    return os.getenv("REDIS_URL") or None


# ---------------------------------------------------------------------------
# 进程内内存 pub/sub（降级兜底）
# ---------------------------------------------------------------------------
class _InMemoryPubSub:
    """单实例内存 pub/sub：每个 channel 一个 asyncio.Queue，订阅者消费后丢弃。"""

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}

    async def publish(self, channel: str, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        for q in list(self._subs.get(channel, ())):
            q.put_nowait(data)

    def publish_sync(self, channel: str, payload: dict) -> None:
        """同步版（供无事件循环的上下文使用）。"""
        data = json.dumps(payload, ensure_ascii=False)
        for q in list(self._subs.get(channel, ())):
            q.put_nowait(data)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(channel, set()).add(q)
        try:
            while True:
                yield await q.get()
        finally:
            subs = self._subs.get(channel)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subs.pop(channel, None)


_memory_pubsub = _InMemoryPubSub()


# ---------------------------------------------------------------------------
# Redis Pub/Sub 封装
# ---------------------------------------------------------------------------
class RedisStream:
    """Redis Pub/Sub 推流封装（channel 由调用方指定）。"""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(url, decode_responses=True)
        self._url = url

    async def publish(self, channel: str, payload: dict) -> None:
        try:
            await self._client.publish(channel, json.dumps(payload, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            # 推流失败不阻断主流程（结果仍会落库，前端靠拉取兜底）
            logger.warning("Redis publish failed channel=%s: %s", channel, e)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        import redis.asyncio as aioredis

        pubsub = self._client.pubsub()
        try:
            await pubsub.subscribe(channel)
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield msg["data"]
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
class PubSub:
    """推流统一入口：优先 Redis，未配置降级内存。"""

    def __init__(self) -> None:
        url = _redis_url()
        if url:
            self._backend: Any = RedisStream(url)
            self._is_redis = True
        else:
            self._backend = _memory_pubsub
            self._is_redis = False

    @property
    def is_redis(self) -> bool:
        return self._is_redis

    async def publish(self, channel: str, payload: dict) -> None:
        await self._backend.publish(channel, payload)

    def subscribe(self, channel: str) -> AsyncIterator[str]:
        return self._backend.subscribe(channel)


# 进程级单例
_pubsub: PubSub | None = None


def get_pubsub() -> PubSub:
    global _pubsub
    if _pubsub is None:
        _pubsub = PubSub()
    return _pubsub


def channel_for(thread_id: str, message_id: str) -> str:
    """构造推流 channel：thread_id:message_id。"""
    return f"{thread_id}:{message_id}"
