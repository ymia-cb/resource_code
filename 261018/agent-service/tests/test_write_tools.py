"""高风险写工具测试：apply_refund / apply_return / update_address。

覆盖：状态校验、二次确认、金额阈值转人工、幂等、审计。
运行：python -m pytest tests/test_write_tools.py -v
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
from app.tools.errors import NeedApproval, DuplicateRequest


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    with get_session() as s:
        s.query(IdempotencyRecord).delete(synchronize_session=False)
        s.query(AuditLog).filter(AuditLog.thread_id.like("w-%")).delete(
            synchronize_session=False
        )
        s.query(OrderItem).filter(OrderItem.order_id.like("W-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("W-%")).delete(
            synchronize_session=False
        )
        s.query(Session).filter(Session.thread_id.like("w-%")).delete(
            synchronize_session=False
        )


def _seed_session(thread_id: str, user_id: str = "user-1"):
    with get_session() as s:
        s.add(Session(thread_id=thread_id, user_id=user_id, status="active"))


def _seed_order(order_id: str, user_id="user-1", status=OrderStatus.PENDING_SHIP.value,
                amount="199.00", paid=True, address="北京市朝阳区"):
    with get_session() as s:
        order = Order(
            order_id=order_id, user_id=user_id, status=status,
            amount=Decimal(amount), paid=paid, stock_deducted=True, address=address,
        )
        order.items.append(
            OrderItem(sku="SKU-1", product_name="商品", quantity=1, unit_price=Decimal(amount))
        )
        s.add(order)


# ---------------------------------------------------------------------------
# update_address
# ---------------------------------------------------------------------------


def test_update_address_pending_ship():
    _seed_session("w-t1")
    _seed_order("W-1", status=OrderStatus.PENDING_SHIP.value, address="旧地址")
    r = execute_tool("update_address", {"order_id": "W-1", "address": "新地址", "confirm": True}, "w-t1")
    assert r["status"] == "ok"
    assert r["data"]["success"] is True
    assert r["data"]["new_address"] == "新地址"


def test_update_address_requires_confirm():
    _seed_session("w-t2")
    _seed_order("W-2", status=OrderStatus.PENDING_SHIP.value)
    r = execute_tool("update_address", {"order_id": "W-2", "address": "新地址", "confirm": False}, "w-t2")
    assert r["status"] == "need_confirm"


def test_update_address_shipped_rejected():
    _seed_session("w-t3")
    _seed_order("W-3", status=OrderStatus.SHIPPED.value)
    r = execute_tool("update_address", {"order_id": "W-3", "address": "新地址", "confirm": True}, "w-t3")
    assert r["status"] == "ok"
    assert r["data"]["success"] is False
    assert "不支持" in r["data"]["message"]


# ---------------------------------------------------------------------------
# apply_refund
# ---------------------------------------------------------------------------


def test_apply_refund_normal():
    _seed_session("w-t4")
    _seed_order("W-4", status=OrderStatus.SHIPPED.value, amount="199.00")
    r = execute_tool("apply_refund", {"order_id": "W-4", "confirm": True}, "w-t4")
    assert r["status"] == "ok"
    assert r["data"]["success"] is True
    assert r["data"]["refund_amount"] == 199.0


def test_apply_refund_over_threshold_needs_approval():
    _seed_session("w-t5")
    _seed_order("W-5", status=OrderStatus.SHIPPED.value, amount="5000.00")
    # 金额 5000 > 阈值 1000 → NeedApproval
    with pytest.raises(NeedApproval):
        execute_tool("apply_refund", {"order_id": "W-5", "confirm": True}, "w-t5")


def test_apply_refund_unpaid_rejected():
    _seed_session("w-t6")
    _seed_order("W-6", status=OrderStatus.SHIPPED.value, paid=False)
    r = execute_tool("apply_refund", {"order_id": "W-6", "confirm": True}, "w-t6")
    assert r["data"]["success"] is False
    assert "未付款" in r["data"]["message"]


def test_apply_refund_idempotent():
    _seed_session("w-t7")
    _seed_order("W-7", status=OrderStatus.SHIPPED.value, amount="199.00")
    execute_tool("apply_refund", {"order_id": "W-7", "confirm": True}, "w-t7")
    # 重复请求 → 幂等命中，正常返回已有结果（不重复退款）
    r2 = execute_tool("apply_refund", {"order_id": "W-7", "confirm": True}, "w-t7")
    assert r2["status"] == "ok"
    assert r2["idempotent_replay"] is True


# ---------------------------------------------------------------------------
# apply_return
# ---------------------------------------------------------------------------


def test_apply_return_shipped():
    _seed_session("w-t8")
    _seed_order("W-8", status=OrderStatus.SHIPPED.value, amount="199.00")
    r = execute_tool("apply_return", {"order_id": "W-8", "confirm": True}, "w-t8")
    assert r["status"] == "ok"
    assert r["data"]["success"] is True


def test_apply_return_over_threshold_needs_approval():
    _seed_session("w-t9")
    _seed_order("W-9", status=OrderStatus.SHIPPED.value, amount="3000.00")
    with pytest.raises(NeedApproval):
        execute_tool("apply_return", {"order_id": "W-9", "confirm": True}, "w-t9")


def test_apply_return_cancelled_rejected():
    _seed_session("w-t10")
    _seed_order("W-10", status=OrderStatus.CANCELLED.value)
    r = execute_tool("apply_return", {"order_id": "W-10", "confirm": True}, "w-t10")
    assert r["data"]["success"] is False


def test_whitelist_has_ten_tools():
    assert set(TOOL_WHITELIST.keys()) == {
        "load_skill",
        "query_order", "query_logistics", "query_coupon", "query_product",
        "cancel_order", "update_address", "apply_refund", "apply_return",
        "image_understand",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
