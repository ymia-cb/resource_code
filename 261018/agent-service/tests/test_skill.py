"""load_skill + 渐进披露测试。

运行：python -m pytest tests/test_skill.py -v
"""
from __future__ import annotations

import pytest

from app.skills import (
    SKILL_CATALOG,
    render_index,
    render_skill_prompt,
    render_support_prompt,
    get_skill,
    tools_for_skill,
)
from app.tools.skill import load_skill
from app.tools.registry import execute_tool
from app.tools.errors import ToolNotAllowed


# ---------------------------------------------------------------------------
# SKILL_CATALOG
# ---------------------------------------------------------------------------


def test_catalog_has_order_and_product():
    assert "order" in SKILL_CATALOG
    assert "product" in SKILL_CATALOG


def test_order_skill_tools():
    tools = tools_for_skill("order")
    assert "query_order" in tools
    assert "cancel_order" in tools
    assert "apply_refund" in tools
    assert "query_product" not in tools  # 商品工具不属于 order


def test_product_skill_tools():
    tools = tools_for_skill("product")
    assert "query_product" in tools
    assert "cancel_order" not in tools


def test_no_skill_returns_empty_tools():
    assert tools_for_skill(None) == []


# ---------------------------------------------------------------------------
# 渐进披露提示词
# ---------------------------------------------------------------------------


def test_render_skill_prompt_unloaded_shows_index():
    prompt = render_skill_prompt(None)
    assert "load_skill" in prompt  # 引导先加载
    assert "order" in prompt
    assert "product" in prompt


def test_render_skill_prompt_loaded_shows_guidance():
    prompt = render_skill_prompt("order")
    assert "当前 Skill" in prompt
    assert "订单售后" in prompt
    assert "取消" in prompt  # guidance 内容


def test_render_support_prompt_contains_conflict_constraint():
    prompt = render_support_prompt("order")
    assert "矛盾" in prompt  # 矛盾请求约束


# ---------------------------------------------------------------------------
# load_skill 工具
# ---------------------------------------------------------------------------


def test_load_skill_returns_full_description():
    result = load_skill("order")
    assert result["skill_code"] == "order"
    assert result["name"] == "订单售后"
    assert "guidance" in result
    assert "tools" in result


def test_load_skill_unknown_code():
    result = load_skill("nonexistent")
    assert "error" in result
    assert "order" in result["available"]


# ---------------------------------------------------------------------------
# load_skill 通过 execute_tool + skill 闸门
# ---------------------------------------------------------------------------


def _seed_session(thread_id: str, user_id: str = "user-1") -> None:
    from app.db.models.session import Session
    from app.db.session import get_session
    with get_session() as s:
        if s.query(Session).filter(Session.thread_id == thread_id).count() == 0:
            s.add(Session(thread_id=thread_id, user_id=user_id))
        s.commit()


def test_load_skill_via_execute_tool():
    _seed_session("skill-t1")
    result = execute_tool("load_skill", {"skill_code": "order"}, "skill-t1")
    assert result["status"] == "ok"
    assert result["data"]["skill_code"] == "order"


def test_load_skill_available_any_skill_state():
    """load_skill 无 skill 归属，任何 active_skill 下都可用。"""
    _seed_session("skill-t2")
    # active_skill=product 时也能调 load_skill（切换领域）
    result = execute_tool("load_skill", {"skill_code": "order"}, "skill-t2", active_skill="product")
    assert result["status"] == "ok"


def test_skill_gate_blocks_order_tool_when_product_loaded():
    """加载 product 后，调 order 工具被拒绝（工具隔离）。"""
    _seed_session("skill-t3")
    with pytest.raises(ToolNotAllowed):
        execute_tool("query_order", {"order_id": "X"}, "skill-t3", active_skill="product")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
