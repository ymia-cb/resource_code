"""事实校验（Grounding）测试。

覆盖：JSON Schema 实体提取、事实源构建、实体校验（幻觉识别）、提示词格式、
summarize/rag 节点用工具成功结果校验回答中的业务事实。

运行：python -m pytest tests/test_grounding.py -v
"""
from __future__ import annotations

import pytest

from app.graph.grounding import (
    GroundedEntity,
    GroundedAnswer,
    extract_entities,
    build_fact_source,
    verify_entities,
    verify_grounding,
    build_grounding_prompt,
)


# ---------------------------------------------------------------------------
# 实体提取
# ---------------------------------------------------------------------------


def test_extract_entities_from_dict():
    raw = {"answer": "已退款", "entities": [{"type": "order_id", "value": "1234"}]}
    entities = extract_entities(raw)
    assert len(entities) == 1
    assert entities[0].type == "order_id"
    assert entities[0].value == "1234"


def test_extract_entities_from_grounded_answer():
    ga = GroundedAnswer(answer="x", entities=[GroundedEntity(type="amount", value=199.0)])
    entities = extract_entities(ga)
    assert len(entities) == 1
    assert entities[0].value == 199.0


def test_extract_entities_from_text():
    text = "您的订单号 12345678 已取消，退款 ¥199.00 元，券码 COUPON1 已退回。"
    entities = extract_entities(text)
    types = {(e.type, str(e.value)) for e in entities}
    assert ("order_id", "12345678") in types
    assert ("amount", "199.0") in types or ("amount", "199.00") in types
    assert ("coupon_code", "COUPON1") in types


# ---------------------------------------------------------------------------
# 事实源构建 + 校验
# ---------------------------------------------------------------------------


def test_build_fact_source_from_successful_tool():
    scratchpad = [
        {
            "result": {
                "status": "ok",
                "data": {"success": True, "order_id": "1234", "refund_amount": 199.0, "message": "已退款"},
            }
        }
    ]
    facts = build_fact_source(scratchpad)
    assert ("order_id", "1234") in facts
    assert ("refund_amount", "199.0") in facts


def test_build_fact_source_skips_failed():
    scratchpad = [
        {"result": {"status": "ok", "data": {"success": False, "order_id": "9999"}}}
    ]
    facts = build_fact_source(scratchpad)
    assert ("order_id", "9999") not in facts


def test_verify_entities_valid_and_hallucinated():
    fact_source = {("order_id", "1234")}
    entities = [
        GroundedEntity(type="order_id", value="1234"),  # 在事实源
        GroundedEntity(type="order_id", value="5678"),  # 幻觉
    ]
    valid, hallucinated = verify_entities(entities, fact_source)
    assert len(valid) == 1
    assert valid[0].value == "1234"
    assert len(hallucinated) == 1
    assert hallucinated[0].value == "5678"


def test_verify_empty_entities_no_hallucination():
    valid, hallucinated = verify_entities([], set())
    assert valid == []
    assert hallucinated == []


# ---------------------------------------------------------------------------
# 分层校验：主通道 entities 优先，仅空时兜底文本提取
# ---------------------------------------------------------------------------


def test_verify_grounding_uses_declared_when_present():
    """模型自报了 entities → 只核对声明实体，正文夹带的幻觉不被正则误报。"""
    fact_source = {("order_id", "1234")}
    declared = [GroundedEntity(type="order_id", value="1234")]
    # 正文里有个「订单 1234」正常；即使正文含其他数字，主通道只看 declared
    hallucinated = verify_grounding(declared, "您的订单 1234 已处理", fact_source)
    assert hallucinated == []


def test_verify_grounding_declared_hallucination_caught():
    """模型自报实体但不在事实源 → 主通道抓到幻觉。"""
    fact_source = {("order_id", "1234")}
    declared = [GroundedEntity(type="order_id", value="9999")]
    hallucinated = verify_grounding(declared, "订单 9999 已处理", fact_source)
    assert len(hallucinated) == 1
    assert hallucinated[0].value == "9999"


def test_verify_grounding_fallback_when_declared_empty():
    """entities 为空 → 兜底从正文提取，抓「自报干净但正文夹带幻觉」。"""
    fact_source = {("order_id", "1234")}
    # 模型没自报实体，但正文里冒出了订单 5678
    hallucinated = verify_grounding([], "您的订单 5678 已退款", fact_source)
    assert any(h.type == "order_id" and h.value == "5678" for h in hallucinated)


def test_verify_grounding_fallback_no_hallucination_when_clean():
    """entities 为空 + 正文干净（无业务实体）→ 不误判。"""
    fact_source = {("order_id", "1234")}
    hallucinated = verify_grounding([], "已为您处理完成。", fact_source)
    assert hallucinated == []


# ---------------------------------------------------------------------------
# 提示词约束
# ---------------------------------------------------------------------------


def test_build_grounding_prompt_contains_format_and_facts():
    facts = {("order_id", "1234")}
    prompt = build_grounding_prompt("回答用户", facts)
    assert "JSON" in prompt
    assert "entities" in prompt
    assert "order_id" in prompt
    assert "1234" in prompt
    assert "严禁凭空编造" in prompt


def test_build_grounding_prompt_empty_facts():
    prompt = build_grounding_prompt("回答用户", set())
    assert "（无）" in prompt


# ---------------------------------------------------------------------------
# summarize 节点：用成功工具结果校验回答
# ---------------------------------------------------------------------------


def test_summarize_passes_with_grounded_answer():
    from app.graph.nodes.summarize import summarize_node

    state = {
        "scratchpad": [
            {"result": {"status": "ok", "data": {"success": True, "message": "已取消", "order_id": "1234"}}}
        ],
        "needs_human": False,
        "sources": [],
    }
    result = summarize_node(state)
    # 回答里不含未核验实体 → 正常返回
    assert result.get("needs_human") is not True
    assert "已取消" in result["final_answer"]


def test_summarize_turns_human_on_hallucination():
    """回答里出现不在事实源里的订单号 → 转人工。"""
    from app.graph.nodes.summarize import summarize_node

    # 模拟：工具成功结果是订单 1234，但回答里冒出订单 9999（幻觉）
    state = {
        "scratchpad": [
            {"result": {"status": "ok", "data": {"success": True, "message": "已取消订单 9999", "order_id": "1234"}}}
        ],
        "needs_human": False,
        "sources": [],
    }
    result = summarize_node(state)
    assert result.get("needs_human") is True
    assert "人工" in result["final_answer"]
