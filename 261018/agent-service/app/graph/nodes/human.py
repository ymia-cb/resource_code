"""转人工节点：生成兜底回复并标记 needs_human。"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph.state import AgentState


def human_node(state: AgentState) -> dict:
    answer = "已为您转接人工客服，请稍候。"
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "sources": [],
        "needs_human": True,
    }
