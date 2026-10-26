"""分类节点：判断意图（细粒度五类）+ 匹配技能 + 写操作细分。

竖切片阶段用确定性规则（mock_classify），置信度低转 human。
"""
from __future__ import annotations

from app.graph.state import AgentState
from app.llm.router import classify

# 置信度阈值：低于此值判为无法归类，转人工兜底（C5 兜底分支）
CONFIDENCE_THRESHOLD = 0.5


def classify_node(state: AgentState) -> dict:
    last = state["messages"][-1].content if state.get("messages") else ""
    intent, skill, sub_intent, confidence = classify(last)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "human",
            "sub_intent": None,
            "active_skill": None,
            "needs_human": True,
            "needs_clarify": False,
        }

    return {
        "intent": intent,
        "sub_intent": sub_intent,
        "active_skill": skill,
        "needs_human": intent == "human",
        "needs_clarify": intent == "clarify",
    }


def route_by_intent(state: AgentState) -> str:
    """条件边：按意图 + 是否已加载 skill 路由（细粒度五类）。

    - knowledge（政策/知识库问答）：直接进 rag，不需要 load_skill
      （知识库问答无需领域工具，policy skill 是 KNOWLEDGE 类型无工具）。
    - query/multi_step 但 active_skill 为空 → 先 load_skill（渐进披露，需领域工具）。
    - 其余按意图路由：query→query / multi_step→planner / clarify→clarify / human→human。
    """
    intent = state.get("intent", "human")
    if intent in ("query", "multi_step") and not state.get("active_skill"):
        return "load_skill"
    if intent in ("knowledge", "query", "multi_step", "clarify", "human"):
        return intent
    return "human"
