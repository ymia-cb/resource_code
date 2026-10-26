"""MCP Streamable HTTP 端点测试。

覆盖：initialize / tools/list / tools/call / 鉴权 / 未知方法 / 参数错误。
运行：python -m pytest tests/test_mcp.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.models.session import Session
from app.db.session import get_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    with get_session() as s:
        s.query(OrderItem).filter(OrderItem.order_id.like("MCP-%")).delete(
            synchronize_session=False
        )
        s.query(Order).filter(Order.order_id.like("MCP-%")).delete(
            synchronize_session=False
        )
        s.query(Session).filter(Session.thread_id.like("mcp-%")).delete(
            synchronize_session=False
        )


def _seed(thread_id, order_id, status=OrderStatus.PENDING_SHIP.value):
    with get_session() as s:
        s.add(Session(thread_id=thread_id, user_id="user-1", status="active"))
        o = Order(
            order_id=order_id, user_id="user-1", status=status,
            amount=Decimal("199.00"), paid=True, stock_deducted=True,
        )
        o.items.append(
            OrderItem(sku="SKU-1", product_name="商品", quantity=1, unit_price=Decimal("199.00"))
        )
        s.add(o)


def _post(method, params=None, req_id=1):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body)


def test_initialize():
    resp = _post("initialize", {"protocolVersion": "2026-07-28"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["protocolVersion"] == "2026-07-28"
    assert body["result"]["serverInfo"]["name"] == "ecommerce-customer-service-agent"


def test_tools_list():
    resp = _post("tools/list")
    assert resp.status_code == 200
    tools = resp.json()["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "load_skill",
        "query_order", "query_logistics", "query_coupon", "query_product",
        "cancel_order", "update_address", "apply_refund", "apply_return",
        "image_understand",
    }
    # 每个工具有 inputSchema
    assert all("inputSchema" in t for t in tools)


def test_tools_call_query_order():
    _seed("mcp-1", "MCP-1")
    resp = _post("tools/call", {
        "name": "query_order",
        "arguments": {"order_id": "MCP-1", "thread_id": "mcp-1"},
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is False
    assert "MCP-1" in result["content"][0]["text"]


def test_tools_call_unknown_tool():
    resp = _post("tools/call", {
        "name": "nonexistent_tool",
        "arguments": {"thread_id": "mcp-2"},
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is True
    assert "TOOL_NOT_ALLOWED" in result["content"][0]["text"]


def test_unknown_method():
    resp = _post("resources/list")
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601  # METHOD_NOT_FOUND


def test_invalid_jsonrpc():
    resp = client.post("/mcp", json={"foo": "bar"})
    assert resp.status_code == 400


def test_cancel_order_via_mcp():
    _seed("mcp-3", "MCP-3", status=OrderStatus.PENDING_SHIP.value)
    resp = _post("tools/call", {
        "name": "cancel_order",
        "arguments": {"order_id": "MCP-3", "confirm": True, "thread_id": "mcp-3"},
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is False


def test_auth_required(monkeypatch):
    # 配置 token 后未带 Authorization 应 401
    monkeypatch.setenv("MCP_API_TOKEN", "secret-token")
    resp = _post("initialize", {"protocolVersion": "2026-07-28"})
    assert resp.status_code == 401
    monkeypatch.delenv("MCP_API_TOKEN")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
