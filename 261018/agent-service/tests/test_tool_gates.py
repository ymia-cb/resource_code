"""五道安全闸门测试：execute_tool 全链路。

覆盖：白名单/参数校验/权限/幂等/二次确认/审计 + 各错误码。
运行：python -m pytest tests/test_tool_gates.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.models.session import Session
from app.db.models.audit import AuditLog
from app.db.models.idempotency import IdempotencyRecord
from app.db.session import get_session
from app.tools.registry import execute_tool, TOOL_WHITELIST
from app.tools.errors import (
    ToolNotAllowed,
    ParamInvalid,
    PermissionDenied,
    DuplicateRequest,
)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    with get_session() as s:
        s.query(IdempotencyRecord).delete(synchronize_session=False)
        s.query(AuditLog).filter(AuditLog.thread_id.like("gate-%")).delete(
            synchronize_session=False
        )
        s.query(OrderItem).filter(OrderItem.order_id.like("GATE-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("GATE-%")).delete(
            synchronize_session=False
        )
        s.query(Session).filter(Session.thread_id.like("gate-%")).delete(
            synchronize_session=False
        )


def _seed_session(thread_id: str, user_id: str = "user-1"):
    with get_session() as s:
        s.add(Session(thread_id=thread_id, user_id=user_id, status="active"))


def _seed_order(order_id: str, user_id: str = "user-1", status: str = OrderStatus.PENDING_SHIP.value):
    with get_session() as s:
        order = Order(
            order_id=order_id, user_id=user_id, status=status,
            amount=Decimal("199.00"), paid=True, stock_deducted=True,
        )
        order.items.append(
            OrderItem(sku="SKU-1", product_name="商品", quantity=1, unit_price=Decimal("199.00"))
        )
        s.add(order)


# ---------------------------------------------------------------------------
# 闸门①：白名单
# ---------------------------------------------------------------------------


def test_gate1_not_in_whitelist():
    with pytest.raises(ToolNotAllowed):
        execute_tool("drop_table", {}, "gate-t1")


# ---------------------------------------------------------------------------
# 闸门②：参数校验
# ---------------------------------------------------------------------------


def test_gate2_invalid_params():
    _seed_session("gate-t2")
    with pytest.raises(ParamInvalid):
        execute_tool("query_order", {"order_id": ""}, "gate-t2")  # 空 order_id 违反 min_length


# ---------------------------------------------------------------------------
# 闸门③：权限（thread 不存在）
# ---------------------------------------------------------------------------


def test_gate3_permission_denied():
    # thread 未注册 → resolve_user 失败
    with pytest.raises(PermissionDenied):
        execute_tool("query_order", {"order_id": "x"}, "gate-nonexistent")


# ---------------------------------------------------------------------------
# 闸门④ + 二次确认 + 幂等
# ---------------------------------------------------------------------------


def test_cancel_requires_confirm():
    _seed_session("gate-t4")
    _seed_order("GATE-1")
    # confirm=False → 返回 need_confirm，不执行
    result = execute_tool(
        "cancel_order", {"order_id": "GATE-1", "confirm": False}, "gate-t4"
    )
    assert result["status"] == "need_confirm"


def test_cancel_executes_and_idempotent():
    _seed_session("gate-t5")
    _seed_order("GATE-2")
    # 首次执行
    r1 = execute_tool(
        "cancel_order", {"order_id": "GATE-2", "confirm": True}, "gate-t5"
    )
    assert r1["status"] == "ok"
    assert r1["data"]["success"] is True

    # 重复执行 → 幂等命中，正常返回已有结果（不重复退款）
    r2 = execute_tool(
        "cancel_order", {"order_id": "GATE-2", "confirm": True}, "gate-t5"
    )
    assert r2["status"] == "ok"
    assert r2["idempotent_replay"] is True  # 命中幂等，未重新执行
    assert r2["data"]["success"] is True


def test_readonly_tool_no_idempotency_record():
    _seed_session("gate-t6")
    _seed_order("GATE-3")
    r = execute_tool("query_order", {"order_id": "GATE-3"}, "gate-t6")
    assert r["status"] == "ok"
    # 只读工具不产生幂等记录
    with get_session() as s:
        assert s.query(IdempotencyRecord).count() == 0


# ---------------------------------------------------------------------------
# 闸门⑤：审计落库
# ---------------------------------------------------------------------------


def test_write_tool_audits():
    _seed_session("gate-t7")
    _seed_order("GATE-4")
    execute_tool("cancel_order", {"order_id": "GATE-4", "confirm": True}, "gate-t7")
    with get_session() as s:
        logs = (
            s.query(AuditLog)
            .filter(AuditLog.thread_id == "gate-t7")
            .filter(AuditLog.tool_name == "cancel_order")
            .all()
        )
        assert len(logs) == 1


def test_readonly_tool_audits_too():
    _seed_session("gate-t8")
    _seed_order("GATE-5")
    execute_tool("query_order", {"order_id": "GATE-5"}, "gate-t8")
    with get_session() as s:
        logs = (
            s.query(AuditLog)
            .filter(AuditLog.thread_id == "gate-t8")
            .filter(AuditLog.tool_name == "query_order")
            .all()
        )
        assert len(logs) == 1


def test_whitelist_has_ten_tools():
    assert set(TOOL_WHITELIST.keys()) == {
        "load_skill",
        "query_order",
        "query_logistics",
        "query_coupon",
        "query_product",
        "cancel_order",
        "update_address",
        "apply_refund",
        "apply_return",
        "image_understand",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
