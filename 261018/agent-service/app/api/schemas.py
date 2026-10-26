"""API 层请求/响应模型（Pydantic）。

对齐 Spec 2.2 消息协议与 5.1 REST API 设计。
"""
from __future__ import annotations

from typing import Literal, Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 消息协议
# ---------------------------------------------------------------------------


class ImageRef(BaseModel):
    """消息中的图片引用（接入层已转存，Agent 不存原始二进制）。"""

    url: str
    type: Literal["product", "screenshot", "receipt"] = "product"


class MessageMeta(BaseModel):
    """消息元信息。"""

    user_id: str | None = None
    order_id: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    text: str
    images: list[ImageRef] = Field(default_factory=list)
    meta: MessageMeta = Field(default_factory=MessageMeta)


class ChatRequest(BaseModel):
    thread_id: str
    message: Message


class Source(BaseModel):
    chunk_id: str
    # 兼容 dict 来源，允许额外字段
    model_config = {"extra": "allow"}


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    needs_human: bool = False
    intent: str | None = None
    error_code: str | None = None  # 稳定错误码（失败时，不含内部堆栈）
    retryable: bool = False  # 是否可稍后重试
    run_id: str | None = None  # 本次 Run ID（供追踪）
    degraded: bool = False  # 是否发生降级（真实模型不可用 → mock 兜底）


# ---------------------------------------------------------------------------
# 会话 / 健康
# ---------------------------------------------------------------------------


class SessionStatus(BaseModel):
    thread_id: str
    status: str
    history: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class CloseResponse(BaseModel):
    thread_id: str
    status: str = "closed"


# ---------------------------------------------------------------------------
# 异步流式消息
# ---------------------------------------------------------------------------


class MessageSubmitResponse(BaseModel):
    """提交消息（异步）后立即返回的应答：携带 message_id / run_id 供订阅流式。"""

    message_id: str
    thread_id: str
    run_id: str
    status: str = "running"  # 后台执行中


class MessageDetail(BaseModel):
    """单条消息的完整结果（刷新/断线后拉取的真源）。"""

    message_id: str
    thread_id: str
    role: str
    user_text: str | None = None
    status: str
    final_answer: str | None = None
    sources: list[dict] = Field(default_factory=list)
    intent: str | None = None
    needs_human: bool = False
    degraded: bool = False
    error_code: str | None = None
    chunk_count: int = 0
    run_id: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# 订单取消
# ---------------------------------------------------------------------------


class CancelOrderRequest(BaseModel):
    order_id: str
    user_id: str
    reason: str | None = None
    # 二次确认：cancel 前必须显式 confirm
    confirm: bool = False


class CancelOrderResponse(BaseModel):
    success: bool
    message: str
    refund_amount: float | None = None
    need_refund: bool = False
    restock: bool = False


class OrderInfo(BaseModel):
    order_id: str
    user_id: str
    status: str
    amount: float
    paid: bool
    stock_deducted: bool
    coupon_code: str | None = None
