"""Clarify 节点：信息不足时反问用户补充，不终止会话。

与 human 节点的区别：
- clarify：有明确操作意图，但缺关键参数（如退款/退货/取消没给订单号），
  反问用户补充后继续，不终止会话（needs_clarify=True, needs_human=False）。
- human：无法处理 / 矛盾请求 / 转人工，终止会话（needs_human=True）。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph.state import AgentState


def _clarify_question(intent: str, sub_intent: str | None, text: str) -> str:
    """根据缺失信息生成反问句（确定性规则，真实场景换小模型）。"""
    # 缺订单号：有写/查操作意图，但文本里没提取到订单号
    has_order_id = any(ch.isdigit() for ch in text) or "订单" in text and _has_order_number(text)
    if sub_intent:
        action = {
            "cancel_order": "取消订单",
            "apply_refund": "申请退款",
            "apply_return": "申请退货",
            "update_address": "修改地址",
        }.get(sub_intent, "处理")
        if not has_order_id:
            return f"好的，为您{action}。请提供您的订单号，以便我帮您处理。"
    # 兜底澄清
    return "为了更准确地帮您处理，请补充一下相关信息（如订单号、商品信息等）。"


def _has_order_number(text: str) -> bool:
    """判断文本里是否带了一个像订单号的标识（数字/字母数字组合）。"""
    import re

    return bool(re.search(r"(订单|单号|编号)\s*[:：]?\s*[A-Za-z0-9\-]{4,}", text))


def clarify_node(state: AgentState) -> dict:
    """反问用户补充信息，标记 needs_clarify，不终止会话。"""
    last = state["messages"][-1].content if state.get("messages") else ""
    intent = state.get("intent", "clarify")
    sub_intent = state.get("sub_intent")
    question = _clarify_question(intent, sub_intent, last)

    return {
        "messages": [AIMessage(content=question)],
        "final_answer": question,
        "needs_clarify": True,
        "needs_human": False,
    }
