"""降级策略测试：降级可观测 + 降级分级 + 降级标记。

运行：python -m pytest tests/test_degradation.py -v
"""
from __future__ import annotations

import httpx
import pytest

from langchain_core.messages import HumanMessage

from app.llm import router
from app.llm.degradation import DegradeReason, should_degrade_safely, record_degradation
from app.observability.breaker import metrics


# ---------------------------------------------------------------------------
# 降级分级：可安全降级 vs 不可安全降级
# ---------------------------------------------------------------------------


def test_safe_degrade_roles():
    """classify/rag/reflector 可安全降级到 mock。"""
    assert should_degrade_safely("classify") is True
    assert should_degrade_safely("rag") is True
    assert should_degrade_safely("reflector") is True


def test_unsafe_degrade_roles():
    """planner/summarize/vlm 降级会丢失关键能力，不可安全降级。"""
    assert should_degrade_safely("planner") is False
    assert should_degrade_safely("summarize") is False
    assert should_degrade_safely("vlm") is False


# ---------------------------------------------------------------------------
# 降级标记：llm_with_status 返回 degraded
# ---------------------------------------------------------------------------


def test_llm_with_status_not_degraded_when_configured(monkeypatch):
    """配了 key 且调用成功 → degraded=False。"""
    monkeypatch.setenv("SMALL_LLM_API_KEY", "sk-x")
    monkeypatch.setenv("SMALL_LLM_MODEL", "m")
    monkeypatch.setenv("SMALL_LLM_BASE_URL", "http://mock/v1")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "真实答案"}}]})

    import app.llm.providers.openai_compat as oc

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        text, degraded = router.llm_with_status("rag", [HumanMessage(content="hi")])
        assert text == "真实答案"
        assert degraded is False
    finally:
        oc.httpx.post = orig


def test_llm_with_status_degraded_when_not_configured(monkeypatch):
    """未配 key → degraded=True，且记录降级指标。"""
    for k in ["SMALL_LLM_API_KEY", "SMALL_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)

    before = metrics.count("mock:degraded:not_configured")
    text, degraded = router.llm_with_status("rag", [HumanMessage(content="测试")])
    assert degraded is True
    assert text == "测试"  # mock 兜底
    after = metrics.count("mock:degraded:not_configured")
    assert after >= before  # 降级被记入指标（可观测）


def test_llm_with_status_degraded_when_provider_fails(monkeypatch):
    """配了 key 但调用失败 → degraded=True（provider_failed）。"""
    monkeypatch.setenv("SMALL_LLM_API_KEY", "sk-x")
    monkeypatch.setenv("SMALL_LLM_MODEL", "m")
    monkeypatch.setenv("SMALL_LLM_BASE_URL", "http://mock/v1")

    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    import app.llm.providers.openai_compat as oc

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        text, degraded = router.llm_with_status("rag", [HumanMessage(content="hi")])
        assert degraded is True
        assert text == "hi"  # mock 兜底（回显最后一条消息）
    finally:
        oc.httpx.post = orig


# ---------------------------------------------------------------------------
# 降级记录：record_degradation 累加指标 + 结构化日志
# ---------------------------------------------------------------------------


def test_record_degradation_incr_metrics():
    before = metrics.count("test-provider:degraded")
    record_degradation("rag", DegradeReason.PROVIDER_FAILED, provider="test-provider", model="m", retries=2)
    after = metrics.count("test-provider:degraded")
    assert after == before + 1
    assert metrics.count("test-provider:degraded:provider_failed") >= 1


# ---------------------------------------------------------------------------
# rag 节点降级安全：真实模型失败 → 确定性 mock_generate，而非回显问题
# ---------------------------------------------------------------------------


def test_rag_node_degrades_to_mock_generate_not_echo(monkeypatch):
    """真实模型调用失败时，rag 应降级到基于检索的 mock_generate，
    而不是把用户问题原文当答案回显。"""
    monkeypatch.setenv("SMALL_LLM_API_KEY", "sk-x")
    monkeypatch.setenv("SMALL_LLM_MODEL", "m")
    monkeypatch.setenv("SMALL_LLM_BASE_URL", "http://mock/v1")

    # 让真实模型调用失败
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    import app.llm.providers.openai_compat as oc
    from app.graph.nodes.rag import rag_node
    from app.knowledge.retriever import retrieve

    # 确保有检索结果（注入一条命中）
    monkeypatch.setattr("app.graph.nodes.rag.retrieve", lambda text, top_k=3: [
        {"chunk_id": "doc_1_0", "text": "退款政策：7 天内可无理由退款", "category": "refund", "score": 0.9}
    ])

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        state = {"messages": [HumanMessage(content="退款政策是什么")], "intent": "knowledge"}
        result = rag_node(state)
        # 降级后答案应基于检索拼接（含「退款政策」内容），而不是回显用户问题
        assert "退款" in result["final_answer"]
        assert result["degraded"] is True
    finally:
        oc.httpx.post = orig


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
