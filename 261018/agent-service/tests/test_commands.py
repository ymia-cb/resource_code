"""命令编排 + 矛盾检测 + 幂等新行为 + skill 隔离测试。

运行：python -m pytest tests/test_commands.py -v
"""
from __future__ import annotations

import pytest

from app.commands import (
    CommandAtomicity,
    detect_conflicts,
    is_write_command,
    get_command,
)
from app.commands.orchestrator import (
    orchestrate,
    topo_sort,
    detect_conflicts_in_plan,
    build_conflict_prompt,
)
from app.tools.registry import execute_tool
from app.tools.errors import ToolNotAllowed
from app.tools.idempotency import make_idempotency_key
from app.db.models.order import Order, OrderStatus
from app.db.session import get_session


# ---------------------------------------------------------------------------
# 命令模型
# ---------------------------------------------------------------------------


def test_command_catalog_has_cancel_order():
    spec = get_command("cancel_order")
    assert spec is not None
    assert spec.atomicity == CommandAtomicity.ATOMIC
    assert spec.skill == "order"


def test_is_write_command():
    assert is_write_command("cancel_order") is True
    assert is_write_command("apply_refund") is True
    assert is_write_command("query_order") is False
    assert is_write_command("query_coupon") is False


# ---------------------------------------------------------------------------
# 矛盾检测
# ---------------------------------------------------------------------------


def test_detect_conflict_cancel_and_ship():
    conflicts = detect_conflicts(["cancel_order", "continue_ship"])
    assert ("cancel_order", "continue_ship") in conflicts


def test_detect_conflict_refund_and_return():
    conflicts = detect_conflicts(["apply_refund", "apply_return"])
    assert ("apply_refund", "apply_return") in conflicts


def test_no_conflict_for_independent_reads():
    assert detect_conflicts(["query_order", "query_coupon"]) == []


def test_no_conflict_for_cancel_alone():
    assert detect_conflicts(["cancel_order"]) == []


def test_conflict_prompt_asks_for_clarification():
    prompt = build_conflict_prompt([("cancel_order", "continue_ship")])
    assert "冲突" in prompt
    assert "无法同时执行" in prompt


# ---------------------------------------------------------------------------
# 编排：拓扑分层
# ---------------------------------------------------------------------------


def test_topo_sort_all_read_level_0():
    levels = topo_sort(["query_order", "query_coupon", "query_product"])
    # 无依赖 → 同层
    assert len(levels) == 1
    assert set(levels[0]) == {"query_order", "query_coupon", "query_product"}


def test_orchestrate_marks_write():
    steps = orchestrate(["query_order", "cancel_order"], {
        "query_order": {"order_id": "X"},
        "cancel_order": {"order_id": "X", "confirm": True},
    })
    write_flags = {s.command: s.write for s in steps}
    assert write_flags["query_order"] is False
    assert write_flags["cancel_order"] is True


# ---------------------------------------------------------------------------
# 幂等新行为：update_address 键含 address
# ---------------------------------------------------------------------------


def test_update_address_idempotency_key_differs_by_address():
    k1 = make_idempotency_key("update_address", "t1", {"order_id": "O1", "address": "地址A"}, ("order_id", "address"))
    k2 = make_idempotency_key("update_address", "t1", {"order_id": "O1", "address": "地址B"}, ("order_id", "address"))
    assert k1 != k2  # 不同地址 → 不同幂等键 → 允许重写


def test_cancel_idempotency_key_ignores_reason():
    k1 = make_idempotency_key("cancel_order", "t1", {"order_id": "O1", "reason": "不想要了"}, ("order_id",))
    k2 = make_idempotency_key("cancel_order", "t1", {"order_id": "O1", "reason": "买错了"}, ("order_id",))
    assert k1 == k2  # reason 不参与幂等键 → 允许重试


# ---------------------------------------------------------------------------
# skill 隔离：跨 skill 调用被拒绝
# ---------------------------------------------------------------------------


def test_skill_gate_rejects_cross_skill_call():
    """active_skill=product 时，调 order 工具应被拒绝。"""
    with pytest.raises(ToolNotAllowed):
        execute_tool("query_order", {"order_id": "X"}, "t-skill", active_skill="product")


def _seed_session(thread_id: str, user_id: str = "user-1") -> None:
    from app.db.models.session import Session
    with get_session() as s:
        if s.query(Session).filter(Session.thread_id == thread_id).count() == 0:
            s.add(Session(thread_id=thread_id, user_id=user_id))
        s.commit()


def test_skill_gate_allows_matching_skill():
    """active_skill=order 时，调 order 工具应放行。"""
    _seed_session("t-skill2")
    result = execute_tool("query_order", {"order_id": "NONEXIST"}, "t-skill2", active_skill="order")
    # query_order 查无此单返回 status ok + data None（只读工具不抛错）
    assert result["status"] == "ok"


def test_skill_gate_lenient_when_no_skill():
    """active_skill=None（未加载 skill，兼容旧路径）时放行。"""
    _seed_session("t-skill3")
    result = execute_tool("query_order", {"order_id": "NONEXIST"}, "t-skill3", active_skill=None)
    assert result["status"] == "ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
