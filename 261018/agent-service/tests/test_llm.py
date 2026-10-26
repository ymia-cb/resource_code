"""LLM 接入测试：Provider 抽象 + OpenAI 兼容 + 路由降级。

用 httpx.MockTransport 模拟 OpenAI 兼容端点，验证：
- OpenAICompatProvider 正确构造请求 + 解析响应
- 无 key 时 classify/llm 降级 mock
- 有 key 时走真实 provider
运行：python -m pytest tests/test_llm.py -v
"""
from __future__ import annotations

import json

import httpx
import pytest

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.providers.openai_compat import OpenAICompatProvider, ProviderError
from app.llm.providers.mock import mock_classify
from app.llm import router


# ---------------------------------------------------------------------------
# OpenAICompatProvider 单元测试（用 MockTransport）
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    """构造一个 httpx.Client，把请求交给 handler(request) -> Response。"""
    return httpx.MockTransport(handler)


def test_openai_compat_provider_chat():
    def handler(request: httpx.Request) -> httpx.Response:
        # 验证请求
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][-1]["role"] == "user"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "你好，我是模型"}}]},
        )

    provider = OpenAICompatProvider(
        base_url="http://mock/v1",
        api_key="test-key",
        model="test-model",
    )
    # 注入 mock transport（绕过真实网络）
    import app.llm.providers.openai_compat as oc

    original_post = oc.httpx.post
    client = httpx.Client(transport=_mock_transport(handler))
    oc.httpx.post = client.post
    try:
        result = provider.chat([HumanMessage(content="hi")])
        assert result == "你好，我是模型"
    finally:
        oc.httpx.post = original_post


def test_openai_compat_provider_raises_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    provider = OpenAICompatProvider(
        base_url="http://mock/v1",
        api_key="k",
        model="m",
        max_retries=0,  # 不重试，快速失败
    )
    import app.llm.providers.openai_compat as oc

    original_post = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=_mock_transport(handler)).post
    try:
        with pytest.raises(ProviderError):
            provider.chat([HumanMessage(content="hi")])
    finally:
        oc.httpx.post = original_post


# ---------------------------------------------------------------------------
# 路由降级测试（无 key → mock）
# ---------------------------------------------------------------------------


def test_classify_falls_back_to_mock_without_key(monkeypatch):
    # 清除 LLM 相关环境变量，强制降级 mock
    for k in ["SMALL_LLM_API_KEY", "SMALL_LLM_MODEL", "LARGE_LLM_API_KEY", "LARGE_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    # 重新加载 settings
    router.get_llm_settings.cache_clear() if hasattr(router.get_llm_settings, "cache_clear") else None

    intent, skill, sub_intent, conf = router.classify("退款政策是什么")
    assert intent == "knowledge"


def test_llm_falls_back_to_mock_without_key(monkeypatch):
    for k in ["SMALL_LLM_API_KEY", "SMALL_LLM_MODEL", "LARGE_LLM_API_KEY", "LARGE_LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)

    result = router.llm("rag", [HumanMessage(content="测试")])
    assert result == "测试"  # mock 返回最后一条消息内容


def test_role_mapping(monkeypatch):
    """classify/rag/reflector → 小模型；planner/summarize → 大模型。"""
    monkeypatch.setenv("SMALL_LLM_API_KEY", "sk-small")
    monkeypatch.setenv("SMALL_LLM_MODEL", "small-model")
    monkeypatch.setenv("SMALL_LLM_BASE_URL", "http://mock/v1")
    monkeypatch.setenv("LARGE_LLM_API_KEY", "sk-large")
    monkeypatch.setenv("LARGE_LLM_MODEL", "large-model")
    monkeypatch.setenv("LARGE_LLM_BASE_URL", "http://mock/v1")

    from app.config import get_llm_settings

    settings = get_llm_settings()
    # 小模型 role
    assert settings.provider_for("classify").model == "small-model"
    assert settings.provider_for("rag").model == "small-model"
    assert settings.provider_for("reflector").model == "small-model"
    # 大模型 role
    assert settings.provider_for("planner").model == "large-model"
    assert settings.provider_for("summarize").model == "large-model"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
