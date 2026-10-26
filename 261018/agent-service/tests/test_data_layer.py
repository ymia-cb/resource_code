"""数据层单元测试：ORM + 取消订单规则引擎 + 取消执行联动。

依赖真实 MySQL（本机 Docker mysql-server，root/root@127.0.0.1:3306/ecommerce_agent）。
运行：python -m pytest tests/test_data_layer.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.models.audit import AuditLog
from app.db.session import get_session
from app.tools.order import check_cancelable, cancel_order, query_order


@pytest.fixture(autouse=True)
def _cleanup_orders():
    """每个用例结束后清理测试订单，保证可重复运行。"""
    yield
    with get_session() as s:
        s.query(OrderItem).filter(OrderItem.order_id.like("ORD-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("ORD-%")).delete(
            synchronize_session=False
        )
        s.query(AuditLog).filter(AuditLog.thread_id.like("t-%")).delete(
            synchronize_session=False
        )


# ---------------------------------------------------------------------------
# check_cancelable 规则引擎（纯函数，无 DB 依赖）
# ---------------------------------------------------------------------------


def test_check_cancelable_all_branches():
    assert check_cancelable("pending_payment") == (True, "可取消")
    assert check_cancelable("pending_ship") == (True, "可取消，将退款")
    assert check_cancelable("shipped") == (False, "已发货，请走退货流程")
    assert check_cancelable("completed") == (False, "已完成，请走售后流程")
    assert check_cancelable("cancelled") == (False, "订单已取消")
    assert check_cancelable("unknown_status") == (False, "未知状态，转人工")


# ---------------------------------------------------------------------------
# ORM 建表 + 插入 + 查询
# ---------------------------------------------------------------------------


def _make_order(
    order_id: str,
    user_id: str,
    status: str,
    amount: str = "199.00",
    paid: bool = True,
    stock_deducted: bool = True,
    coupon_code: str | None = None,
) -> None:
    with get_session() as s:
        order = Order(
            order_id=order_id,
            user_id=user_id,
            status=status,
            amount=Decimal(amount),
            paid=paid,
            stock_deducted=stock_deducted,
            coupon_code=coupon_code,
        )
        order.items.append(
            OrderItem(
                sku="SKU-001",
                product_name="测试商品",
                quantity=1,
                unit_price=Decimal(amount),
            )
        )
        s.add(order)


def test_insert_and_query_order():
    _make_order("ORD-TEST-1", "user-1", OrderStatus.PENDING_SHIP.value)
    info = query_order("ORD-TEST-1", "user-1")
    assert info is not None
    assert info["status"] == "pending_ship"
    assert info["amount"] == 199.0
    assert info["paid"] is True


def test_query_order_wrong_user_returns_none():
    _make_order("ORD-TEST-2", "user-2", OrderStatus.PENDING_SHIP.value)
    # 其他用户查不到该订单（越权保护）
    assert query_order("ORD-TEST-2", "user-999") is None


# ---------------------------------------------------------------------------
# cancel_order 联动
# ---------------------------------------------------------------------------


def test_cancel_pending_ship_refund_and_restock():
    _make_order(
        "ORD-CANCEL-1", "user-1", OrderStatus.PENDING_SHIP.value,
        amount="199.00", paid=True, stock_deducted=True, coupon_code="CP-001",
    )
    out = cancel_order("ORD-CANCEL-1", "user-1", reason="不想要了", thread_id="t-cancel")
    assert out.success is True
    assert out.need_refund is True
    assert out.restock is True
    assert out.refund_amount == Decimal("199.00")

    info = query_order("ORD-CANCEL-1", "user-1")
    assert info["status"] == "cancelled"


def test_cancel_pending_payment_no_refund():
    _make_order(
        "ORD-CANCEL-2", "user-1", OrderStatus.PENDING_PAYMENT.value,
        paid=False, stock_deducted=False,
    )
    out = cancel_order("ORD-CANCEL-2", "user-1")
    assert out.success is True
    assert out.need_refund is False
    assert out.restock is False
    assert out.refund_amount is None


def test_cancel_shipped_rejected():
    _make_order("ORD-CANCEL-3", "user-1", OrderStatus.SHIPPED.value)
    out = cancel_order("ORD-CANCEL-3", "user-1")
    assert out.success is False
    assert "退货" in out.message
    # 状态未被改动
    assert query_order("ORD-CANCEL-3", "user-1")["status"] == "shipped"


def test_cancel_idempotent_when_already_cancelled():
    _make_order(
        "ORD-CANCEL-4", "user-1", OrderStatus.CANCELLED.value,
        paid=True, stock_deducted=False,
    )
    out = cancel_order("ORD-CANCEL-4", "user-1")
    assert out.success is True
    assert "幂等" in out.message
    # 已取消订单不重复退款
    assert out.need_refund is False


def test_cancel_order_not_found():
    out = cancel_order("ORD-NOT-EXIST", "user-1")
    assert out.success is False
    assert "不存在" in out.message


def test_cancel_writes_audit_log():
    """审计职责已上移到 execute_tool 闸门⑤，cancel_order 业务函数不再直接写审计。

    这里验证：cancel_order 直接调用不会产生 audit_log（审计统一走 execute_tool）。
    """
    _make_order(
        "ORD-CANCEL-5", "user-1", OrderStatus.PENDING_SHIP.value,
        paid=True, stock_deducted=True, coupon_code="CP-002",
    )
    out = cancel_order("ORD-CANCEL-5", "user-1", reason="测试审计", thread_id="t-audit")
    assert out.success is True
    # 业务函数直接调用不写审计（审计由 execute_tool 闸门⑤统一负责）
    with get_session() as s:
        logs = (
            s.query(AuditLog)
            .filter(AuditLog.thread_id == "t-audit")
            .filter(AuditLog.tool_name == "cancel_order")
            .all()
        )
        assert len(logs) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
