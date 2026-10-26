"""聊天路由：/chat + /session 系列。

对接已跑通的 graph（文本→分类→RAG直答→checkpoint记忆）。
/chat 管理 AgentRun 生命周期：生成 run_id、注入图、捕获异常返回稳定错误码。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    SessionStatus,
    CloseResponse,
)
from app.api.deps import get_graph
from app.observability import (
    ErrorCategory,
    ErrorCode,
    SAFE_MESSAGE,
    classify_exception,
    category_to_code,
    ObservedError,
)
from app.observability.run_tracker import new_run_id, start_run, finish_run
from app.db.models.agent_run import RunStatus

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, graph=Depends(get_graph)) -> ChatResponse:
    """发送消息，返回答案 + 溯源 + 是否转人工 + 稳定错误码。

    多轮记忆由 checkpointer 按 thread_id 隔离；每次请求一个 run_id 供追踪。
    """
    text = req.message.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="message.text 不能为空")

    run_id = new_run_id()
    start_run(run_id, req.thread_id)

    config = {"configurable": {"thread_id": req.thread_id, "run_id": run_id}}
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config,
        )
    except Exception as e:  # noqa: BLE001
        # 图执行异常 → 记录结构化日志（底层原因/异常类型/状态码只进日志）+ 稳定错误码返回
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
        finish_run(run_id, RunStatus.FAILED, error_code=code.value, error_category=category.value)
        return ChatResponse(
            answer=SAFE_MESSAGE[code],
            sources=[],
            needs_human=True,
            error_code=code.value,
            retryable=category in (ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMITED, ErrorCategory.NETWORK, ErrorCategory.SERVER_ERROR),
            run_id=run_id,
        )

    finish_run(run_id, RunStatus.SUCCESS)

    sources = [
        Source(chunk_id=s.get("chunk_id", ""), **{k: v for k, v in s.items() if k != "chunk_id"})
        for s in result.get("sources", [])
    ]
    return ChatResponse(
        answer=result.get("final_answer") or "",
        sources=sources,
        needs_human=bool(result.get("needs_human")),
        intent=result.get("intent"),
        run_id=run_id,
        degraded=bool(result.get("degraded")),
    )


@router.get("/session/{thread_id}", response_model=SessionStatus)
def get_session_status(thread_id: str, graph=Depends(get_graph)) -> SessionStatus:
    """查询会话历史（从 checkpoint 读取）。"""
    config = {"configurable": {"thread_id": thread_id}}
    history: list[dict] = []
    try:
        state = graph.get_state(config)
        for msg in state.values.get("messages", []):
            history.append(
                {
                    "role": getattr(msg, "type", "unknown"),
                    "text": getattr(msg, "content", ""),
                }
            )
    except Exception:
        # 无历史或读取失败时返回空
        pass
    return SessionStatus(thread_id=thread_id, status="active", history=history)


@router.post("/session/{thread_id}/close", response_model=CloseResponse)
def close_session(thread_id: str) -> CloseResponse:
    """关闭会话（M1 为逻辑关闭，不真正删 checkpoint）。"""
    return CloseResponse(thread_id=thread_id, status="closed")
