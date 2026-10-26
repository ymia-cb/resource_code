"""Agent 图的状态定义。

State 是 LangGraph 图在节点间传递的唯一载体，所有字段必须在此显式声明，
避免在节点里塞入未定义的隐式状态。
"""
from typing import Annotated, TypedDict, Literal

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # 消息历史（add_messages 保证跨节点自动累加而非覆盖）
    messages: Annotated[list[BaseMessage], add_messages]

    # 分流结果（细粒度五类）：
    #   knowledge = 政策/平台规则等知识库问答 → RAG + LLM
    #   query     = 物流/订单状态等一步查询 → query 节点单步（轻量，不经 planner 重推理）
    #   multi_step = 多步读写混合 + 顺序/条件逻辑 → planner 重推理循环
    #               （Plan-Execute-Reflect：规划 + 执行 + 反思，支持条件分支）
    #   clarify   = 信息不足需澄清 → clarify 节点反问补充信息（不终止会话）
    #   human     = 无法处理/矛盾请求 → 转人工（终止会话）
    intent: Literal["knowledge", "query", "multi_step", "clarify", "human", "unknown"]

    # 写操作的细分命令（cancel/refund/return/address），供 planner 精准拆解
    # 只在 intent=multi_step 时有值，其余为 None
    sub_intent: str | None

    # 当前命中的技能名（渐进式披露，竖切片阶段暂仅透传）
    active_skill: str | None

    # 复杂推理中间草稿：[{step, tool, params, result}]
    scratchpad: list[dict]

    # Planner 产出的子任务清单：[{step, tool, params, description}]
    plan: list[dict]

    # 当前执行到的 plan 下标（-1 表示未开始）
    plan_index: int

    # Reflector 判断是否继续（True=下一步 / False=结束）
    plan_done: bool

    # 最终答案
    final_answer: str | None

    # 溯源引用：[{chunk_id, text, source}]
    sources: list[dict]

    # 是否需要转人工
    needs_human: bool

    # 是否需要澄清（信息不足，反问用户补充，不终止会话）
    needs_clarify: bool

    # 是否发生了降级（真实模型不可用 → mock/确定性规则兜底）
    # 供上层感知「当前答案是否为降级态」，可据此提示用户或转人工
    degraded: bool
