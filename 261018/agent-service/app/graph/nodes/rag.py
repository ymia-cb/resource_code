"""RAG 直答节点：检索 + 生成 + 溯源校验 + 事实校验。

生成走 router.llm（真实大模型，未配置降级 mock），提示词约束输出
「回答 + 实体」结构化格式，生成后提取实体，用知识库溯源做事实校验。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.graph.state import AgentState
from app.graph.grounding import (
    GroundedEntity,
    verify_grounding,
    build_grounding_prompt,
)
from app.knowledge.retriever import retrieve
from app.llm import router
from app.llm.providers.mock import mock_generate


def rag_node(state: AgentState) -> dict:
    last = state["messages"][-1].content if state.get("messages") else ""
    sources = retrieve(last, top_k=3)

    # 无命中 → 转人工兜底（避免无据可答时瞎编）
    if not sources:
        answer = "抱歉，我暂时无法回答这个问题，已为您转接人工客服。"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "sources": [],
            "needs_human": True,
        }

    degraded = False
    # 有真实大模型 → 走 router.llm + grounding 提示词（约束输出结构化格式）
    if router._build_provider("rag") is not None:
        answer, hallucinated, degraded = _generate_with_grounding(last, sources)
    else:
        # 无 key 降级 mock（确定性拼接，基于检索结果，安全）
        answer = mock_generate(state["messages"], sources)
        hallucinated = []
        degraded = True

    if hallucinated:
        # 回答出现不在知识库里的业务实体 → 转人工，不返回不可靠信息
        answer = "抱歉，我暂时无法可靠回答这个问题，已为您转接人工客服。"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "sources": [],
            "needs_human": True,
        }

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "sources": sources,
        "needs_human": False,
        "degraded": degraded,
    }


def _generate_with_grounding(last: str, sources: list[dict]) -> tuple[str, list, bool]:
    """用真实大模型 + grounding 提示词生成，并做事实校验。

    Returns:
        (answer, hallucinated_entities, degraded)
        degraded=True 表示真实模型调用失败、降级到确定性 mock（基于检索拼接）。
    """
    fact_source = {(s.get("category", ""), s.get("chunk_id", "")) for s in sources}
    knowledge_text = "\n".join(f"[{s['chunk_id']}] {s['text']}" for s in sources)
    prompt = build_grounding_prompt(
        f"基于以下知识库内容回答用户问题「{last}」：\n{knowledge_text}",
        fact_source,
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=last),
    ]
    raw, degraded = router.llm_with_status("rag", messages)

    # 真实模型调用失败 → 降级到确定性 mock_generate（基于检索结果拼接，安全），
    # 而不是回显用户问题（避免把用户问题当答案返回）。
    if degraded:
        answer = mock_generate(messages, sources)
        return answer, [], True

    answer, declared_entities = _parse_grounded_output(raw)
    # 分层校验：声明实体优先，仅空时兜底从正文提取
    hallucinated = verify_grounding(
        declared_entities=declared_entities,
        answer_text=answer,
        fact_source=fact_source,
    )
    return answer, hallucinated, False


def _parse_grounded_output(raw: str) -> tuple[str, list]:
    """解析大模型输出，返回 (answer, declared_entities)。

    模型按提示词应返回 JSON（answer + entities）；未按格式则 answer=原文，
    declared_entities=[]（触发兜底文本提取）。
    """
    import json

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            answer = str(data.get("answer", raw))
            entities = data.get("entities") or []
            declared = [GroundedEntity(**e) if isinstance(e, dict) else e for e in entities]
            return answer, declared
        return raw, []
    except (json.JSONDecodeError, ValueError):
        return raw, []
