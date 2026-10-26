"""异步流式 worker：后台跑 graph.astream() 节点级流式，逐 chunk 推 Redis + 落库。

链路（方案 A「提交即返回」）：
    POST /chat → 立即返回 message_id（不阻塞）
    asyncio.create_task(run_stream_agent(...)) 后台协程：
        async for chunk in graph.astream(...)   # 节点级流式
            → 解析节点产出 → PUBLISH chunk 到 Redis（带自增 chunk_id）
        → 跑完：完整结果落库 status=done
        → 失败：落库 status=failed + error_code，并推一条 error chunk

chunk 结构（SSE 透传给前端）：
    {"type": "node", "node": "classify", "chunk_id": 1, "data": {...}}
    {"type": "token", "chunk_id": N, "text": "..."}            # 预留 token 级
    {"type": "done", "chunk_id": N, "final_answer": "...", ...}
    {"type": "error", "chunk_id": N, "error_code": "..."}
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage

from app.api.deps import get_graph
from app.stream import message_repo
from app.stream.pubsub import get_pubsub, channel_for
from app.observability import (
    classify_exception,
    category_to_code,
    ObservedError,
)
from app.observability.run_tracker import new_run_id, start_run, finish_run
from app.db.models.agent_run import RunStatus

logger = logging.getLogger(__name__)

# 节点名 → 前端友好阶段文案（节点级流式的进度提示）
NODE_STAGE_LABEL = {
    "classify": "正在理解您的问题…",
    "load_skill": "正在加载相应技能…",
    "rag": "正在检索知识库…",
    "planner": "正在拆解任务…",
    "executor": "正在查询/处理…",
    "reflector": "正在校验结果…",
    "summarize": "正在生成答案…",
    "human": "正在为您转接人工客服…",
}

# 最终答案所在的节点（astream 产出的 chunk 里包含 final_answer）
_FINAL_NODES = {"rag", "summarize", "human"}


def _extract_text(content: object) -> str:
    """把消息 content 转成纯文本（兼容 str / list 多模态块）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _node_state(node_update: dict) -> dict | None:
    """从 astream 的节点产出中提取状态片段。

    LangGraph 节点产出形如 {node_name: {field: value}}；部分实现产出
    多键（如 {node_name: {...}, "final_answer": ...}）。取 node_name 对应的 dict。
    """
    for key, value in node_update.items():
        if isinstance(value, dict) and key != "__end__":
            return value
    return None


async def _astream_in_thread(graph, user_text: str, config: dict):
    """在独立线程跑同步 graph.stream，桥接为 async generator。

    原因：自研 MySQLSaver 是同步 BaseCheckpointSaver，graph.astream 会触发
    async checkpointer（NotImplementedError）。同步 stream 放进线程后逐 chunk
    通过 asyncio.Queue 回传，事件循环不阻塞。
    """
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_text)]},
                config,
            ):
                queue.put_nowait(("chunk", chunk))
            queue.put_nowait(("done", None))
        except Exception as e:  # noqa: BLE001
            queue.put_nowait(("error", e))

    thread = asyncio.get_running_loop().run_in_executor(None, _run)
    try:
        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                yield value
            elif kind == "error":
                raise value
            else:  # done
                break
    finally:
        # 确保线程结束（run_in_executor 返回的 Future，这里不强制取消，避免残留）
        pass


async def run_stream_agent(
    thread_id: str,
    message_id: str,
    user_text: str,
    run_id: str,
    session_factory=None,
) -> None:
    """后台协程：跑 graph.astream 节点级流式，逐 chunk 推 Redis，最终落库。

    所有异常在此捕获并转为失败态落库 + 推 error chunk，绝不向上抛出导致
    create_task 未捕获告警。

    ``session_factory``：可注入的 session 上下文管理器（默认 app.db.session.get_session），
    测试中可替换为 SQLite 内存库以离线运行。
    """
    if session_factory is None:
        from app.db.session import get_session as _get_session
        session_factory = _get_session

    graph = get_graph()
    pubsub = get_pubsub()
    channel = channel_for(thread_id, message_id)

    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}}
    chunk_id = 0
    start_run(run_id, thread_id, session_factory=session_factory)

    async def _push(payload: dict) -> None:
        await pubsub.publish(channel, payload)

    try:
        final_answer = ""
        sources: list[dict] = []
        needs_human = False
        degraded = False
        intent: str | None = None
        stage_hint = "正在思考…"

        # 用 to_thread 把同步 graph.stream 包到线程执行：
        # 自研 MySQLSaver 是同步 BaseCheckpointSaver，graph.astream 会触发
        # async checkpointer 调用（NotImplementedError）。同步 stream + 线程
        # 既能复用现有 saver，又不阻塞事件循环（HTTP 层保持异步）。
        async for chunk in _astream_in_thread(graph, user_text, config):
            # 每个节点产出一个 chunk
            for node_name, node_update in chunk.items():
                if node_name == "__end__":
                    continue
                state = _node_state(node_update) if isinstance(node_update, dict) else None
                if state is None:
                    state = node_update if isinstance(node_update, dict) else {}

                # 收集最终字段（可能出现在多个节点，取最后非空）
                if state.get("intent"):
                    intent = state["intent"]
                if state.get("needs_human"):
                    needs_human = True
                if state.get("degraded"):
                    degraded = True
                if state.get("sources"):
                    sources = state["sources"]
                if state.get("final_answer"):
                    final_answer = state["final_answer"]

                # 节点消息回写（AIMessage 累加在 messages 里），提取阶段文本
                stage_hint = NODE_STAGE_LABEL.get(node_name, "正在处理…")
                text = ""
                if isinstance(state, dict) and state.get("messages"):
                    msgs = state["messages"]
                    if isinstance(msgs, list) and msgs:
                        text = _extract_text(getattr(msgs[-1], "content", ""))

                chunk_id += 1
                await _push({
                    "type": "node",
                    "node": node_name,
                    "chunk_id": chunk_id,
                    "stage": stage_hint,
                    "text": text,
                })

        # 流结束：落库 + 推 done
        finish_run(run_id, RunStatus.SUCCESS, session_factory=session_factory)
        with session_factory() as session:
            message_repo.update_message_done(
                session,
                message_id,
                final_answer=final_answer,
                sources=sources,
                intent=intent,
                needs_human=needs_human,
                degraded=degraded,
                chunk_count=chunk_id,
            )
        await _push({
            "type": "done",
            "chunk_id": chunk_id + 1,
            "final_answer": final_answer,
            "sources": sources,
            "intent": intent,
            "needs_human": needs_human,
            "degraded": degraded,
        })

    except Exception as e:  # noqa: BLE001
        category, status_code = classify_exception(e)
        code = category_to_code(category)
        ObservedError(
            category=category,
            code=code,
            run_id=run_id,
            detail=str(e),
            exception_type=type(e).__name__,
            status_code=status_code,
        ).log()
        finish_run(run_id, RunStatus.FAILED, error_code=code.value, error_category=category.value, session_factory=session_factory)
        with session_factory() as session:
            message_repo.update_message_failed(session, message_id, code.value)
        await _push({
            "type": "error",
            "chunk_id": chunk_id + 1,
            "error_code": code.value,
            "message": "处理出现问题，请稍后重试或转人工客服。",
        })


def submit_stream_agent(
    thread_id: str,
    message_id: str,
    user_text: str,
) -> str:
    """提交后台流式任务，返回 run_id。

    立即返回，不阻塞 HTTP 请求；实际执行由 asyncio.create_task 后台协程完成。
    """
    run_id = new_run_id()
    task = asyncio.create_task(
        run_stream_agent(thread_id, message_id, user_text, run_id)
    )
    # 防止协程异常未被消费（内部已 try/except 兜底，这里做二次保险）
    task.add_done_callback(_on_task_done)
    return run_id


def _on_task_done(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception as e:  # noqa: BLE001
        logger.exception("stream agent task crashed unexpectedly: %s", e)
