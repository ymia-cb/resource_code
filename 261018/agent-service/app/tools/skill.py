"""load_skill 工具：渐进披露的入口。

未加载 skill 时，工具层仅开放 load_skill（+ 无归属的跨领域工具）。
load_skill 把所选 skill 写入运行状态（active_skill），返回其完整说明，
后续提示词与工具白名单切换到该 skill。

load_skill 本身是无副作用、幂等的（重复加载同一 skill 无害）。
"""
from __future__ import annotations

from app.skills import get_skill, SKILL_CATALOG, tools_for_skill, type_label


def load_skill(skill_code: str, user_id: str | None = None) -> dict:
    """加载 skill，返回完整说明（含 guidance + 工具）。

    user_id 由 execute_tool 注入（权限校验结果），本工具不使用但需接收以兼容。

    Returns:
        {"skill_code": ..., "name": ..., "type": ..., "guidance": [...], "tools": [...]}
    未知 skill_code 返回 {"error": ...} + available（不抛异常，由上层引导重试）。
    """
    skill = get_skill(skill_code)
    if skill is None:
        return {
            "error": f"未知 Skill: {skill_code}",
            "available": sorted(SKILL_CATALOG.keys()),
        }
    return {
        "skill_code": skill.code,
        "name": skill.name,
        "type": type_label(skill.type),
        "description": skill.description,
        "guidance": list(skill.guidance),
        "tools": tools_for_skill(skill.code),
    }
