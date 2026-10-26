"""图装配：节点注册 + 条件边 + 绑定 checkpointer。

图结构（细粒度五类意图路由）：
    START → classify → {knowledge: rag, query: query, multi_step: planner, clarify: clarify, human: human}
    - knowledge 分支：rag（RAG + LLM 直答政策/知识库）
    - query 分支：query（一步查询物流/订单状态，轻量，不经重推理）
    - multi_step 分支：planner → executor → reflector → {继续: executor, 结束: summarize} → END
      （多步读写混合 + 顺序/条件逻辑，Plan-Execute-Reflect 重循环，支持条件分支）
    - clarify 分支：clarify（反问补充信息，不终止会话）
    - human 分支：human（转人工，终止会话）
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes.classify import classify_node, route_by_intent
from app.graph.nodes.load_skill import load_skill_node
from app.graph.nodes.rag import rag_node
from app.graph.nodes.query import query_node
from app.graph.nodes.clarify import clarify_node
from app.graph.nodes.human import human_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.executor import executor_node
from app.graph.nodes.reflector import reflector_node
from app.graph.nodes.summarize import summarize_node


def _should_continue(state: AgentState) -> str:
    """Reflector 之后的边：继续执行 or 汇总结束。"""
    if state.get("plan_done"):
        return "summarize"
    return "executor"


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("load_skill", load_skill_node)
    graph.add_node("rag", rag_node)
    graph.add_node("query", query_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("human", human_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "load_skill": "load_skill",
            "knowledge": "rag",
            "query": "query",
            "multi_step": "planner",
            "clarify": "clarify",
            "human": "human",
        },
    )
    # load_skill 之后：若成功加载 → 回到 classify 的意图路由（带 active_skill 再分流）；
    # 若仍需人工 → human。
    graph.add_conditional_edges(
        "load_skill",
        lambda s: "human" if s.get("needs_human") else "classify",
        {"classify": "classify", "human": "human"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("query", END)
    graph.add_edge("clarify", END)
    graph.add_edge("human", END)

    # multi_step 分支：Plan-Execute-Reflect 循环
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _should_continue,
        {"executor": "executor", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=checkpointer)
