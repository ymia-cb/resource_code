"""流式聊天路由：异步提交 + SSE 订阅 + 消息拉取。

三个核心端点：
- POST /chat/stream/submit  → 提交消息，立即返回 message_id/run_id（不阻塞）
- GET  /chat/stream/{message_id} → SSE 订阅该消息的实时 chunk（Redis 推流）
- GET  /messages/{message_id} → 拉取该消息的完整结果（刷新/断线兜底）
- GET  /threads/{thread_id}/messages → 拉取整个会话的历史消息
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.schemas import MessageSubmitResponse, MessageDetail, ChatRequest
from app.db.session import get_session
from app.stream import message_repo, channel_for, get_pubsub, submit_stream_agent
from app.observability.run_tracker import new_run_id

router = APIRouter(tags=["chat-stream"])


@router.post("/chat/stream/submit", response_model=MessageSubmitResponse)
async def submit_chat(req: ChatRequest) -> MessageSubmitResponse:
    """提交消息：立即返回，Agent 在后台异步执行。

    与旧 /chat 的区别：不等待 graph 跑完，而是返回 message_id，前端随后
    通过 SSE 订阅流式 chunk；刷新/断线后靠 GET /messages/{id} 拉完整结果。

    注意：必须 async def（在事件循环内执行），否则 submit_stream_agent 里的
    asyncio.create_task 会因「无运行中的事件循环」失败。
    """
    text = req.message.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="message.text 不能为空")

    thread_id = req.thread_id
    # 落库（同步 ORM，放到线程池避免阻塞事件循环）
    message_id = await asyncio.to_thread(_create_message_sync, thread_id, text)

    run_id = submit_stream_agent(thread_id, message_id, text)

    return MessageSubmitResponse(
        message_id=message_id,
        thread_id=thread_id,
        run_id=run_id,
        status="running",
    )


def _create_message_sync(thread_id: str, text: str) -> str:
    """同步落一条 running 消息，返回 message_id。"""
    with get_session() as session:
        return message_repo.create_message(session, thread_id=thread_id, user_text=text)


@router.get("/chat/stream/{message_id}")
async def stream_chat(message_id: str) -> StreamingResponse:
    """SSE 订阅某条消息的实时 chunk。

    - 先检查消息是否已完成（done/failed）：若已完成，直接推一条 done/error 后关闭。
    - 否则订阅 Redis（或内存降级）channel，转发 chunk。
    - 客户端断线 → 生成器退出 → 取消订阅；重连后重新订阅 + 先拉库补历史。
    """
    # 先查消息当前状态，若已完成则无需订阅，直接回放最终态（DB 同步读放线程池）
    msg = await asyncio.to_thread(_get_message_sync, message_id)
    if msg is None:
        raise HTTPException(status_code=404, detail="消息不存在")

    thread_id = msg.thread_id
    channel = channel_for(thread_id, message_id)

    async def event_stream():
        # 若已完成，直接推最终态并结束
        if msg.status != "running":
            if msg.status == "failed":
                yield _sse("error", {
                    "type": "error",
                    "chunk_id": 0,
                    "error_code": msg.error_code,
                    "message": "处理失败，请稍后重试或转人工客服。",
                })
            else:
                yield _sse("done", {
                    "type": "done",
                    "chunk_id": 0,
                    "final_answer": msg.final_answer,
                    "sources": msg.sources or [],
                    "intent": msg.intent,
                    "needs_human": msg.needs_human == "1",
                    "degraded": msg.degraded == "1",
                })
            return

        # running：订阅推流 channel，转发 chunk 直到收到 done/error
        pubsub = get_pubsub()
        try:
            async for raw in pubsub.subscribe(channel):
                payload = json.loads(raw)
                ptype = payload.get("type", "node")
                yield _sse(ptype, payload)
                if ptype in ("done", "error"):
                    break
        except asyncio.CancelledError:
            # 客户端断开：正常退出，取消订阅由 pubsub.subscribe 的 finally 处理
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/messages/{message_id}", response_model=MessageDetail)
async def get_message_detail(message_id: str) -> MessageDetail:
    """拉取单条消息的完整结果（刷新/断线兜底的真源）。"""
    msg = await asyncio.to_thread(_get_message_sync, message_id)
    if msg is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    return MessageDetail(**message_repo.to_dict(msg))


@router.get("/threads/{thread_id}/messages", response_model=list[MessageDetail])
async def list_thread_messages(thread_id: str) -> list[MessageDetail]:
    """拉取整个会话的历史消息（按时间升序）。"""
    msgs = await asyncio.to_thread(_list_messages_sync, thread_id)
    return [MessageDetail(**message_repo.to_dict(m)) for m in msgs]


def _get_message_sync(message_id: str):
    with get_session() as session:
        return message_repo.get_message(session, message_id)


def _list_messages_sync(thread_id: str):
    with get_session() as session:
        return message_repo.list_messages(session, thread_id)


def _sse(event: str, payload: dict) -> str:
    """构造 SSE 帧。"""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"
