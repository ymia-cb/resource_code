"""Agent 中间件层：在模型调用 / 工具调用前后统一执行横切逻辑。

中间件解决的问题（回应「没有加入 agent 中间件的调用」）：
- skill 渐进披露（提示词注入 + 工具裁剪）从节点/提示词硬编码抽成中间件。
- 工具调用超时统一收口成中间件。
- 未来的横切逻辑（审计、限流、日志、重试）都挂到这里，不侵入业务节点。

设计：
- ``Middleware`` 抽象基类：before / after 钩子。
- ``MiddlewareChain``：按顺序编排多个中间件。
- 中间件在「模型调用前」（before_llm）和「工具调用前」（before_tool）介入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class LLMContext:
    """一次模型调用的上下文（中间件 before_llm 可改写 messages）。"""

    role: str  # classify / rag / planner / summarize / vlm
    messages: list[Any]  # 消息列表
    active_skill: str | None = None  # 当前激活的 skill
    extra: dict = field(default_factory=dict)


@dataclass
class ToolContext:
    """一次工具调用的上下文（中间件 before_tool 可拦截/改参数）。"""

    name: str  # 工具名
    params: dict  # 参数
    active_skill: str | None = None  # 当前激活的 skill
    extra: dict = field(default_factory=dict)


class Middleware:
    """中间件基类。子类按需覆写钩子。"""

    # 模型调用前：可改写 messages / 拦截（返回非 None 则短路）
    def before_llm(self, ctx: LLMContext) -> LLMContext:
        return ctx

    # 模型调用后：可改写结果
    def after_llm(self, ctx: LLMContext, result: str) -> str:
        return result

    # 工具调用前：可拦截（抛异常或改参数）
    def before_tool(self, ctx: ToolContext) -> ToolContext:
        return ctx

    # 工具调用后：可改写结果
    def after_tool(self, ctx: ToolContext, result: dict) -> dict:
        return result


class MiddlewareChain:
    """中间件链：按顺序执行多个中间件。"""

    def __init__(self, middlewares: list[Middleware] | None = None):
        self.middlewares = middlewares or []

    def add(self, mw: Middleware) -> "MiddlewareChain":
        self.middlewares.append(mw)
        return self

    def before_llm(self, ctx: LLMContext) -> LLMContext:
        for mw in self.middlewares:
            ctx = mw.before_llm(ctx)
        return ctx

    def after_llm(self, ctx: LLMContext, result: str) -> str:
        for mw in self.middlewares:
            result = mw.after_llm(ctx, result)
        return result

    def before_tool(self, ctx: ToolContext) -> ToolContext:
        for mw in self.middlewares:
            ctx = mw.before_tool(ctx)
        return ctx

    def after_tool(self, ctx: ToolContext, result: dict) -> dict:
        for mw in self.middlewares:
            result = mw.after_tool(ctx, result)
        return result


# 进程级默认中间件链（惰性组装，避免循环导入）
_default_chain: MiddlewareChain | None = None


def get_default_chain() -> MiddlewareChain:
    """返回进程级默认中间件链（含渐进披露 + 工具超时）。"""
    global _default_chain
    if _default_chain is None:
        from app.middleware.skill_disclosure import SkillDisclosureMiddleware
        from app.middleware.tool_timeout import ToolTimeoutMiddleware

        _default_chain = MiddlewareChain(
            [
                SkillDisclosureMiddleware(),
                ToolTimeoutMiddleware(),
            ]
        )
    return _default_chain
