"""接口层测试：/chat 多轮记忆 + /health + /orders cancel 二次确认。

运行：python -m pytest tests/test_api.py -v
依赖真实 MySQL（订单相关用例）。
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.session import get_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_orders():
    """每个用例结束后清理测试订单，保证可重复运行。"""
    yield
    with get_session() as s:
        # 先删子表 order_items，再删父表 orders（外键约束）
        s.query(OrderItem).filter(OrderItem.order_id.like("ORD-API-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("ORD-API-%")).delete(
            synchronize_session=False
        )


def _make_order(order_id: str, user_id: str, status: str, amount="199.00", paid=True, stock=True):
    with get_session() as s:
        order = Order(
            order_id=order_id,
            user_id=user_id,
            status=status,
            amount=Decimal(amount),
            paid=paid,
            stock_deducted=stock,
        )
        order.items.append(
            OrderItem(sku="SKU-1", product_name="测试商品", quantity=1, unit_price=Decimal(amount))
        )
        s.add(order)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /chat 多轮记忆
# ---------------------------------------------------------------------------


def test_chat_simple_answer():
    resp = client.post("/chat", json={"thread_id": "api-t1", "message": {"role": "user", "text": "退款政策是什么"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "knowledge"
    assert body["answer"]  # 非空
    assert body["needs_human"] is False


def test_chat_empty_text_400():
    resp = client.post("/chat", json={"thread_id": "api-t2", "message": {"role": "user", "text": ""}})
    assert resp.status_code == 400


def test_chat_session_history_persists():
    tid = "api-t-history"
    client.post("/chat", json={"thread_id": tid, "message": {"role": "user", "text": "退款政策是什么"}})
    resp = client.get(f"/session/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == tid
    # checkpoint 记忆：历史里应包含用户消息与助手的回复
    assert len(body["history"]) >= 2


# ---------------------------------------------------------------------------
# /orders 取消（含二次确认）
# ---------------------------------------------------------------------------


def test_order_cancelable_readonly():
    _make_order("ORD-API-1", "user-1", OrderStatus.PENDING_SHIP.value)
    resp = client.get("/orders/ORD-API-1/cancelable", params={"user_id": "user-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelable"] is True


def test_order_cancel_requires_confirm():
    _make_order("ORD-API-2", "user-1", OrderStatus.PENDING_SHIP.value)
    # 未确认 → 不执行，返回需二次确认
    resp = client.post("/orders/cancel", json={
        "order_id": "ORD-API-2", "user_id": "user-1", "confirm": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert "确认" in body["message"]

    # 订单状态未被改动
    info = client.get("/orders/ORD-API-2", params={"user_id": "user-1"}).json()
    assert info["status"] == "pending_ship"


def test_order_cancel_confirmed():
    _make_order("ORD-API-3", "user-1", OrderStatus.PENDING_SHIP.value, paid=True, stock=True)
    resp = client.post("/orders/cancel", json={
        "order_id": "ORD-API-3", "user_id": "user-1", "confirm": True, "reason": "不想要了",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["need_refund"] is True
    assert body["restock"] is True
    assert body["refund_amount"] == 199.0

    info = client.get("/orders/ORD-API-3", params={"user_id": "user-1"}).json()
    assert info["status"] == "cancelled"


def test_order_cancel_shipped_conflict():
    _make_order("ORD-API-4", "user-1", OrderStatus.SHIPPED.value)
    resp = client.post("/orders/cancel", json={
        "order_id": "ORD-API-4", "user_id": "user-1", "confirm": True,
    })
    assert resp.status_code == 409
    assert "退货" in resp.json()["detail"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
