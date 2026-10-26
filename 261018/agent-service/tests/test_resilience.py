"""健壮性测试：退避重试 / 熔断 / 错误分类 / Run 追踪 / 稳定错误码。

运行：python -m pytest tests/test_resilience.py -v
"""
from __future__ import annotations

import httpx
import pytest

from langchain_core.messages import HumanMessage

from app.llm.providers.openai_compat import OpenAICompatProvider, ProviderError
from app.observability import (
    ErrorCategory,
    ErrorCode,
    classify_exception,
    category_to_code,
    SAFE_MESSAGE,
)
from app.observability.breaker import CircuitBreaker, Metrics
from app.observability.run_tracker import new_run_id, start_run, finish_run, record_tool_execution
from app.db.models.agent_run import AgentRun, ToolExecution, RunStatus
from app.db.session import get_session


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------


def test_classify_timeout():
    exc = httpx.ReadTimeout("timeout")
    assert classify_exception(exc) == (ErrorCategory.TIMEOUT, None)


def test_classify_rate_limited():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(429, request=req)
    assert classify_exception(httpx.HTTPStatusError("429", request=req, response=resp)) == (ErrorCategory.RATE_LIMITED, 429)


def test_classify_auth_error_401():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(401, request=req)
    assert classify_exception(httpx.HTTPStatusError("401", request=req, response=resp)) == (ErrorCategory.AUTH_ERROR, 401)


def test_classify_auth_error_403():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(403, request=req)
    assert classify_exception(httpx.HTTPStatusError("403", request=req, response=resp)) == (ErrorCategory.AUTH_ERROR, 403)


def test_classify_config_error_400():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(400, request=req)
    assert classify_exception(httpx.HTTPStatusError("400", request=req, response=resp)) == (ErrorCategory.CONFIG_ERROR, 400)


def test_classify_config_error_404():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(404, request=req)
    assert classify_exception(httpx.HTTPStatusError("404", request=req, response=resp)) == (ErrorCategory.CONFIG_ERROR, 404)


def test_classify_client_error_4xx_other():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(408, request=req)
    assert classify_exception(httpx.HTTPStatusError("408", request=req, response=resp)) == (ErrorCategory.CLIENT_ERROR, 408)


def test_classify_server_error_5xx():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(500, request=req)
    assert classify_exception(httpx.HTTPStatusError("500", request=req, response=resp)) == (ErrorCategory.SERVER_ERROR, 500)


def test_classify_network_error():
    assert classify_exception(httpx.ConnectError("conn refused")) == (ErrorCategory.NETWORK, None)


def test_category_to_code_mapping():
    assert category_to_code(ErrorCategory.TIMEOUT) == ErrorCode.TIMEOUT
    assert category_to_code(ErrorCategory.RATE_LIMITED) == ErrorCode.RATE_LIMITED
    assert category_to_code(ErrorCategory.SERVER_ERROR) == ErrorCode.MODEL_UNAVAILABLE
    assert category_to_code(ErrorCategory.AUTH_ERROR) == ErrorCode.AUTH_FAILED
    assert category_to_code(ErrorCategory.CONFIG_ERROR) == ErrorCode.CONFIG_ERROR
    assert category_to_code(ErrorCategory.TOOL_ERROR) == ErrorCode.TOOL_FAILED


def test_safe_message_no_internal_detail():
    """安全提示不含堆栈/内部信息。"""
    for code, msg in SAFE_MESSAGE.items():
        assert "traceback" not in msg.lower()
        assert "Exception" not in msg


# ---------------------------------------------------------------------------
# 退避重试：4xx 不重试、5xx 重试
# ---------------------------------------------------------------------------


def test_4xx_not_retried():
    """4xx 客户端错误应立即失败，不重试（重试无意义）。"""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    provider = OpenAICompatProvider(base_url="http://mock/v1", api_key="k", model="m", max_retries=2)
    import app.llm.providers.openai_compat as oc

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        with pytest.raises(ProviderError) as e:
            provider.chat([HumanMessage(content="hi")])
        assert e.value.category == ErrorCategory.AUTH_ERROR
        assert e.value.code == ErrorCode.AUTH_FAILED
        assert e.value.status_code == 401
        assert calls["n"] == 1  # 只请求一次，不重试
    finally:
        oc.httpx.post = orig


def test_5xx_retried_then_fails():
    """5xx 服务端错误会重试（带退避），耗尽后抛 ProviderError 并记录重试次数。"""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "server error"})

    provider = OpenAICompatProvider(base_url="http://mock/v1", api_key="k", model="m", max_retries=2)
    import app.llm.providers.openai_compat as oc

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        with pytest.raises(ProviderError) as e:
            provider.chat([HumanMessage(content="hi")])
        assert e.value.category == ErrorCategory.SERVER_ERROR
        assert e.value.code == ErrorCode.MODEL_UNAVAILABLE
        assert calls["n"] == 3  # 初始 1 次 + 重试 2 次
    finally:
        oc.httpx.post = orig


def test_400_config_error_not_retried():
    """400 配置错误（模型名错误等）→ CONFIG_ERROR，立即失败不重试。"""
    # 熔断器是进程级单例，前面测试可能已把它打开 → 复位，避免短路到 SERVER_ERROR
    from app.observability import breaker as _b
    _b.breaker.record_success("openai_compat")

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    provider = OpenAICompatProvider(base_url="http://mock/v1", api_key="k", model="m", max_retries=2)
    import app.llm.providers.openai_compat as oc

    orig = oc.httpx.post
    oc.httpx.post = httpx.Client(transport=httpx.MockTransport(handler)).post
    try:
        with pytest.raises(ProviderError) as e:
            provider.chat([HumanMessage(content="hi")])
        assert e.value.category == ErrorCategory.CONFIG_ERROR
        assert e.value.code == ErrorCode.CONFIG_ERROR
        assert e.value.status_code == 400
        assert calls["n"] == 1  # 配置错误不重试
    finally:
        oc.httpx.post = orig


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(threshold=2, cooldown=100.0)  # 冷却设长，避免半开干扰
    assert cb.allow("p") is True
    assert cb.record_failure("p") is False  # 第 1 次失败，未达阈值
    assert cb.record_failure("p") is True  # 第 2 次失败，触发熔断
    assert cb.allow("p") is False  # 熔断中，拒绝
    # 成功复位
    cb.record_success("p")
    assert cb.allow("p") is True


def test_metrics_counters():
    m = Metrics()
    m.incr("p:error")
    m.incr("p:error")
    assert m.count("p:error") == 2
    snap = m.snapshot()
    assert snap["p:error"] == 2


# ---------------------------------------------------------------------------
# Run 追踪
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_runs():
    yield
    with get_session() as s:
        s.query(ToolExecution).filter(ToolExecution.run_id.like("res-%")).delete(synchronize_session=False)
        s.query(AgentRun).filter(AgentRun.run_id.like("res-%")).delete(synchronize_session=False)


def test_run_lifecycle():
    run_id = "res-" + new_run_id()[:8]
    start_run(run_id, "thread-x")
    finish_run(run_id, RunStatus.SUCCESS)

    with get_session() as s:
        run = s.query(AgentRun).filter(AgentRun.run_id == run_id).first()
        assert run is not None
        assert run.status == RunStatus.SUCCESS.value
        assert run.thread_id == "thread-x"


def test_run_failure_records_error_code():
    run_id = "res-" + new_run_id()[:8]
    start_run(run_id, "thread-y")
    finish_run(run_id, RunStatus.FAILED, error_code=ErrorCode.TIMEOUT.value, error_category=ErrorCategory.TIMEOUT.value)

    with get_session() as s:
        run = s.query(AgentRun).filter(AgentRun.run_id == run_id).first()
        assert run.error_code == ErrorCode.TIMEOUT.value
        assert run.error_category == ErrorCategory.TIMEOUT.value


def test_tool_execution_recorded():
    run_id = "res-" + new_run_id()[:8]
    start_run(run_id, "thread-z")
    record_tool_execution(run_id, "cancel_order", "success", result={"success": True})

    with get_session() as s:
        execs = s.query(ToolExecution).filter(ToolExecution.run_id == run_id).all()
        assert len(execs) == 1
        assert execs[0].tool_name == "cancel_order"
        assert execs[0].status == "success"
        assert execs[0].result == {"success": True}


# ---------------------------------------------------------------------------
# /chat 稳定错误码
# ---------------------------------------------------------------------------


def test_chat_returns_run_id_and_no_error_on_success():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={
        "thread_id": "res-chat-1",
        "message": {"role": "user", "text": "退款政策是什么"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] is not None
    assert data["error_code"] is None


def test_chat_error_code_on_internal_failure(monkeypatch):
    """图执行异常 → 返回稳定错误码 + needs_human，不泄露堆栈。"""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import deps

    def boom(*a, **k):
        raise RuntimeError("内部爆炸 detail-不应该泄露")

    # 让 graph.invoke 抛异常
    client = TestClient(app)
    import app.api.routes.chat as chat_route

    monkeypatch.setattr(chat_route, "new_run_id", lambda: "res-fail-1")

    # 直接打桩 get_graph 返回一个 invoke 抛异常的假 graph
    fake_graph = type("G", (), {"invoke": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))})()

    # 用 dependency override 替换
    from app.api.deps import get_graph as _get_graph

    app.dependency_overrides[_get_graph] = lambda: fake_graph
    try:
        resp = client.post("/chat", json={
            "thread_id": "res-chat-fail",
            "message": {"role": "user", "text": "hi"},
        })
        data = resp.json()
        assert data["error_code"] is not None
        assert data["needs_human"] is True
        # 不泄露内部堆栈
        assert "boom" not in data["answer"]
        assert "RuntimeError" not in data["answer"]
    finally:
        app.dependency_overrides.clear()
