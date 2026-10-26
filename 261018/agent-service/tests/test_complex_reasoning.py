"""复杂推理（Plan-Execute-Reflect）测试。

覆盖：取消/退款/退货多步场景、大额转人工、步数超限、无法拆解转人工。
运行：python -m pytest tests/test_complex_reasoning.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.graph.build import build_graph
from app.memory.checkpointer import get_checkpointer
from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.models.session import Session
from app.db.models.audit import AuditLog
from app.db.models.idempotency import IdempotencyRecord
from app.db.session import get_session
from langchain_core.messages import HumanMessage


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    from app.memory.mysql_saver import MySQLSaver

    saver = MySQLSaver()
    with get_session() as s:
        s.query(IdempotencyRecord).delete(synchronize_session=False)
        s.query(AuditLog).filter(AuditLog.thread_id.like("cr-%")).delete(
            synchronize_session=False
        )
        s.query(OrderItem).filter(OrderItem.order_id.like("CR-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("CR-%")).delete(
            synchronize_session=False
        )
        s.query(Session).filter(Session.thread_id.like("cr-%")).delete(
            synchronize_session=False
        )
    # 清理 checkpoint（同 thread_id 的历史累积会导致重复执行）
    for tid in ["cr-1", "cr-2", "cr-3", "cr-4", "cr-5", "cr-6"]:
        saver.delete_thread(tid)


def _seed(thread_id, user_id, order_id, status, amount="199.00", paid=True):
    with get_session() as s:
        s.add(Session(thread_id=thread_id, user_id=user_id, status="active"))
        o = Order(
            order_id=order_id, user_id=user_id, status=status,
            amount=Decimal(amount), paid=paid, stock_deducted=True,
        )
        o.items.append(
            OrderItem(sku="SKU-1", product_name="商品", quantity=1, unit_price=Decimal(amount))
        )
        s.add(o)


def _run(text, thread_id):
    app = build_graph(get_checkpointer())
    return app.invoke(
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": thread_id}},
    )


def test_cancel_order_flow():
    _seed("cr-1", "user-1", "CR-1", OrderStatus.PENDING_SHIP.value)
    r = _run("帮我取消订单CR-1", "cr-1")
    assert r["intent"] == "multi_step"
    assert r["needs_human"] is False
    assert "已取消" in r["final_answer"]
    # 两步：query_order → cancel_order
    assert [x["tool"] for x in r["scratchpad"]] == ["query_order", "cancel_order"]


def test_refund_flow():
    _seed("cr-2", "user-1", "CR-2", OrderStatus.SHIPPED.value, amount="199.00")
    r = _run("我要退款订单CR-2", "cr-2")
    assert r["intent"] == "multi_step"
    assert r["needs_human"] is False
    assert "退款" in r["final_answer"]
    assert [x["tool"] for x in r["scratchpad"]] == ["query_order", "apply_refund"]


def test_return_flow():
    _seed("cr-3", "user-1", "CR-3", OrderStatus.SHIPPED.value, amount="199.00")
    r = _run("我要退货订单CR-3", "cr-3")
    assert r["intent"] == "multi_step"
    assert r["needs_human"] is False
    assert "退货" in r["final_answer"]
    assert [x["tool"] for x in r["scratchpad"]] == ["query_order", "apply_return"]


def test_refund_large_amount_goes_human():
    # 金额 5000 > 阈值 1000 → 转人工
    _seed("cr-4", "user-1", "CR-4", OrderStatus.SHIPPED.value, amount="5000.00")
    r = _run("我要退款订单CR-4", "cr-4")
    assert r["needs_human"] is True
    assert "人工" in r["final_answer"]


def test_cancel_shipped_goes_human():
    # 已发货订单取消 → 业务规则拒绝 → 转人工引导退货
    _seed("cr-5", "user-1", "CR-5", OrderStatus.SHIPPED.value)
    r = _run("帮我取消订单CR-5", "cr-5")
    assert r["needs_human"] is True


def test_unrecognizable_intent_goes_human():
    # 无订单号、无法拆解 → 转人工
    r = _run("这个问题很复杂我想想", "cr-6")
    assert r["needs_human"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
