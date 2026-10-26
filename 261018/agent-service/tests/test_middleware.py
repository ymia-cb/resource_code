"""Agent 中间件测试：Middleware 基类/链编排、渐进披露、工具超时。

运行：python -m pytest tests/test_middleware.py -v
"""
from __future__ import annotations

import time

import pytest

from app.middleware import (
    Middleware,
    MiddlewareChain,
    LLMContext,
    ToolContext,
    get_default_chain,
)
from app.middleware.skill_disclosure import SkillDisclosureMiddleware
from app.middleware.tool_timeout import ToolTimeoutMiddleware, run_tool_with_timeout
from app.skills import tools_for_skill, tool_type, BusinessType
from app.tools.errors import ToolNotAllowed


# ---------------------------------------------------------------------------
# Middleware 基类 + 链编排
# ---------------------------------------------------------------------------


class _RecordMW(Middleware):
    """记录钩子调用顺序的测试中间件。"""

    def __init__(self):
        self.calls: list[str] = []

    def before_llm(self, ctx: LLMContext) -> LLMContext:
        self.calls.append("before_llm")
        return ctx

    def after_llm(self, ctx: LLMContext, result: str) -> str:
        self.calls.append("after_llm")
        return result + "!"

    def before_tool(self, ctx: ToolContext) -> ToolContext:
        self.calls.append("before_tool")
        return ctx

    def after_tool(self, ctx: ToolContext, result: dict) -> dict:
        self.calls.append("after_tool")
        result["touched"] = True
        return result


def test_chain_runs_hooks_in_order():
    mw = _RecordMW()
    chain = MiddlewareChain([mw, _RecordMW()])

    chain.before_llm(LLMContext(role="planner", messages=[]))
    assert mw.calls == ["before_llm"]

    result = chain.after_llm(LLMContext(role="planner", messages=[]), "hi")
    assert result == "hi!!"  # 两个中间件各追加一个 !

    chain.before_tool(ToolContext(name="t", params={}))
    out = chain.after_tool(ToolContext(name="t", params={}), {"a": 1})
    assert out["touched"] is True


def test_default_chain_contains_disclosure_and_timeout():
    chain = get_default_chain()
    kinds = [type(mw).__name__ for mw in chain.middlewares]
    assert "SkillDisclosureMiddleware" in kinds
    assert "ToolTimeoutMiddleware" in kinds


# ---------------------------------------------------------------------------
# SkillDisclosureMiddleware：before_llm 注入 + 去重
# ---------------------------------------------------------------------------


def _messages(*contents):
    from langchain_core.messages import SystemMessage, HumanMessage

    return [HumanMessage(content=c) for c in contents]


def test_disclosure_injects_skill_prompt_when_unloaded():
    mw = SkillDisclosureMiddleware()
    ctx = LLMContext(role="planner", messages=_messages("查订单"), active_skill=None)
    ctx = mw.before_llm(ctx)
    joined = str(ctx.messages[0].content)
    assert "【动态 Skill 路由】" in joined
    assert "order" in joined  # 索引含 order 领域
    assert len(ctx.messages) == 2  # 系统提示词前置，原消息保留


def test_disclosure_injects_guidance_when_loaded():
    mw = SkillDisclosureMiddleware()
    ctx = LLMContext(role="planner", messages=_messages("退这个单"), active_skill="order")
    ctx = mw.before_llm(ctx)
    joined = str(ctx.messages[0].content)
    assert "【当前 Skill：订单售后" in joined
    assert "二次确认" in joined  # guidance 注入


def test_disclosure_does_not_double_inject():
    mw = SkillDisclosureMiddleware()
    from langchain_core.messages import SystemMessage

    ctx = LLMContext(
        role="planner",
        messages=[SystemMessage(content="【动态 Skill 路由】\n可用 Skill...")],
        active_skill=None,
    )
    ctx2 = mw.before_llm(ctx)
    assert len(ctx2.messages) == 1  # 已含 skill 提示 → 不重复注入


# ---------------------------------------------------------------------------
# SkillDisclosureMiddleware：before_tool 隔离
# ---------------------------------------------------------------------------


def test_disclosure_allows_load_skill():
    mw = SkillDisclosureMiddleware()
    ctx = ToolContext(name="load_skill", params={"skill_code": "order"}, active_skill=None)
    # 不抛异常即通过
    assert mw.before_tool(ctx).name == "load_skill"


def test_disclosure_rejects_cross_skill_tool():
    mw = SkillDisclosureMiddleware()
    ctx = ToolContext(name="query_product", params={}, active_skill="order")
    with pytest.raises(ToolNotAllowed):
        mw.before_tool(ctx)


def test_disclosure_allows_own_skill_tool():
    mw = SkillDisclosureMiddleware()
    # order 领域内允许 query_order（写前置查询）
    ctx = ToolContext(name="query_order", params={}, active_skill="order")
    assert mw.before_tool(ctx).name == "query_order"


# ---------------------------------------------------------------------------
# ToolTimeoutMiddleware：差异化超时 + 超时执行
# ---------------------------------------------------------------------------


def test_timeout_by_business_type():
    mw = ToolTimeoutMiddleware()
    assert mw.timeout_for("query_order") == 10.0
    assert mw.timeout_for("cancel_order") == 15.0  # WRITE 类
    assert mw.timeout_for("image_understand") == 20.0  # MULTIMODAL
    assert mw.timeout_for("unknown_tool") == 10.0  # 默认


def test_tool_type_mapping():
    assert tool_type("query_product") == BusinessType.QUERY
    assert tool_type("cancel_order") == BusinessType.WRITE
    assert tool_type("image_understand") == BusinessType.MULTIMODAL


def test_run_with_timeout_returns_result():
    timed_out, result = run_tool_with_timeout(lambda: 42, timeout=1.0)
    assert timed_out is False
    assert result == 42


def test_run_with_timeout_times_out():
    def slow():
        time.sleep(0.3)
        return 1

    timed_out, result = run_tool_with_timeout(slow, timeout=0.05)
    assert timed_out is True
    assert result is None


def test_run_with_timeout_propagates_exception():
    def boom():
        raise ValueError("x")

    timed_out, result = run_tool_with_timeout(boom, timeout=1.0)
    assert timed_out is False
    assert isinstance(result, ValueError)
