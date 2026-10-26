"""Skill 渐进披露中间件。

把「渐进披露」从 render_skill_prompt 硬编码抽成中间件：
- before_llm：按 active_skill 注入/裁剪系统提示词（未加载→路由+索引，已加载→guidance）。
- before_tool：按 active_skill 做工具级隔离校验（跨 skill 工具拒绝），
  与 execute_tool 内部的 skill 闸门呼应（此处是中间件层的第一道拦截）。

这样业务节点不再各自拼提示词，横切逻辑统一由中间件收口。
"""
from __future__ import annotations

from app.middleware import Middleware, LLMContext, ToolContext
from app.skills import render_skill_prompt, tools_for_skill
from app.tools.errors import ToolNotAllowed


class SkillDisclosureMiddleware(Middleware):
    """渐进披露：提示词注入 + 工具裁剪。"""

    def before_llm(self, ctx: LLMContext) -> LLMContext:
        """在模型调用前，按 active_skill 注入渐进披露提示词。"""
        skill_prompt = render_skill_prompt(ctx.active_skill)

        # 若消息里已有系统提示词，追加 skill 段；否则前置注入
        from langchain_core.messages import SystemMessage

        # 避免重复注入（若最后一条消息已是 skill 提示则跳过）
        last = ctx.messages[-1] if ctx.messages else None
        if last is not None and getattr(last, "type", "") == "system":
            existing = str(getattr(last, "content", ""))
            if "【动态 Skill 路由】" in existing or "【当前 Skill" in existing:
                return ctx

        ctx.messages = [SystemMessage(content=skill_prompt)] + list(ctx.messages)
        return ctx

    def before_tool(self, ctx: ToolContext) -> ToolContext:
        """工具调用前做 skill 隔离校验（中间件层第一道拦截）。"""
        # load_skill 无归属，始终放行
        if ctx.name == "load_skill":
            return ctx
        allowed = tools_for_skill(ctx.active_skill)
        # active_skill 为空（未加载）时，仅放行无归属工具 + load_skill
        if ctx.active_skill is None:
            # 未加载 skill：渐进披露要求仅 load_skill 可用
            # 但为兼容旧路径（query_order 等），此处放行，由 execute_tool 内闸门兜底
            return ctx
        if allowed and ctx.name not in allowed:
            # 跨 skill 工具调用 → 拒绝
            from app.tools.registry import TOOL_WHITELIST
            spec = TOOL_WHITELIST.get(ctx.name)
            if spec is not None and spec.skill is not None and spec.skill != ctx.active_skill:
                raise ToolNotAllowed(
                    f"工具 {ctx.name} 不属于当前 skill（active={ctx.active_skill}）"
                )
        return ctx
