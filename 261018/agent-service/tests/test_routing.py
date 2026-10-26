"""细粒度路由测试：knowledge / query / multi_step / clarify / human 五类意图。

覆盖：
1. mock_classify 细粒度分类（六类关键词 + sub_intent + 条件分支）
2. query_node 一步查询（物流/订单状态，缺订单号转澄清）
3. clarify_node 反问补充信息（不终止会话）
4. planner 多步规划 + 条件分支
5. executor 条件分支判断
6. 图级路由冒烟（五类意图各自落到正确节点）

运行：python -m pytest tests/test_routing.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.llm.providers.mock import mock_classify, _detect_sub_intent, _has_conditional_logic
from app.graph.nodes.query import query_node, _pick_query_tool
from app.graph.nodes.clarify import clarify_node
from app.graph.nodes.planner import _build_plan, _detect_condition_branches
from app.graph.nodes.executor import _condition_met
from app.graph.build import build_graph
from app.memory.checkpointer import get_checkpointer
from app.db.models.order import Order, OrderItem, OrderStatus
from app.db.models.session import Session
from app.db.session import get_session
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# mock_classify 细粒度分类
# ---------------------------------------------------------------------------


def test_classify_knowledge():
    intent, skill, sub, conf = mock_classify("退款政策是什么")
    assert intent == "knowledge"
    assert skill is None
    assert sub is None


def test_classify_query_logistics():
    intent, skill, sub, conf = mock_classify("我的订单到哪了")
    assert intent == "query"
    assert skill == "order"
    assert sub is None


def test_classify_query_order_status():
    intent, skill, sub, conf = mock_classify("查一下订单状态")
    assert intent == "query"


def test_classify_multi_step_cancel():
    intent, skill, sub, conf = mock_classify("帮我取消订单")
    assert intent == "multi_step"
    assert sub == "cancel_order"


def test_classify_multi_step_refund():
    intent, skill, sub, conf = mock_classify("我要退款订单1234")
    assert intent == "multi_step"
    assert sub == "apply_refund"


def test_classify_multi_step_conditional():
    # 条件分支（没发货就取消，发货了就改地址）→ multi_step
    intent, skill, sub, conf = mock_classify("如果订单没发货就取消，发货了就改地址")
    assert intent == "multi_step"
    assert sub is None


def test_classify_clarify_no_order_id():
    # 有「退款」动作词但无订单号 → clarify
    intent, skill, sub, conf = mock_classify("我想退款")
    assert intent == "clarify"


def test_classify_human_unknown():
    intent, skill, sub, conf = mock_classify("今天天气怎么样")
    assert intent == "human"


def test_detect_sub_intent():
    assert _detect_sub_intent("取消订单") == "cancel_order"
    assert _detect_sub_intent("退货") == "apply_return"
    assert _detect_sub_intent("退款") == "apply_refund"
    assert _detect_sub_intent("改地址") == "update_address"


# ---------------------------------------------------------------------------
# query_node 一步查询
# ---------------------------------------------------------------------------


def test_pick_query_tool():
    assert _pick_query_tool("我的订单到哪了") == "query_logistics"
    assert _pick_query_tool("查订单状态") == "query_order"


def test_query_node_missing_order_id_goes_clarify():
    state = {"messages": [HumanMessage(content="我的订单到哪了")], "active_skill": "order"}
    r = query_node(state, {"configurable": {"thread_id": "q-1"}})
    assert r["needs_clarify"] is True
    assert r["needs_human"] is False


def test_query_node_order_status():
    # 播种订单
    with get_session() as s:
        s.add(Session(thread_id="q-2", user_id="user-1", status="active"))
        o = Order(order_id="Q-1", user_id="user-1", status=OrderStatus.SHIPPED.value,
                  amount=Decimal("199.00"), paid=True, stock_deducted=True)
        s.add(o)

    state = {"messages": [HumanMessage(content="查订单Q-1的状态")], "active_skill": "order"}
    r = query_node(state, {"configurable": {"thread_id": "q-2"}})
    assert r["needs_human"] is False
    assert "已发货" in r["final_answer"]

    with get_session() as s:
        s.query(Order).filter(Order.order_id == "Q-1").delete(synchronize_session=False)
        s.query(Session).filter(Session.thread_id == "q-2").delete(synchronize_session=False)


def test_query_node_logistics():
    with get_session() as s:
        s.add(Session(thread_id="q-3", user_id="user-1", status="active"))
        o = Order(order_id="Q-2", user_id="user-1", status=OrderStatus.SHIPPED.value,
                  amount=Decimal("199.00"), paid=True, stock_deducted=True)
        s.add(o)

    state = {"messages": [HumanMessage(content="我的订单Q-2到哪了")], "active_skill": "order"}
    r = query_node(state, {"configurable": {"thread_id": "q-3"}})
    assert r["needs_human"] is False
    assert "物流" in r["final_answer"]

    with get_session() as s:
        s.query(Order).filter(Order.order_id == "Q-2").delete(synchronize_session=False)
        s.query(Session).filter(Session.thread_id == "q-3").delete(synchronize_session=False)


# ---------------------------------------------------------------------------
# clarify_node 反问
# ---------------------------------------------------------------------------


def test_clarify_node_asks_for_order_id():
    state = {"messages": [HumanMessage(content="我想退款")], "intent": "clarify",
             "sub_intent": "apply_refund"}
    r = clarify_node(state)
    assert r["needs_clarify"] is True
    assert r["needs_human"] is False
    assert "订单号" in r["final_answer"]


def test_clarify_node_generic_prompt():
    state = {"messages": [HumanMessage(content="帮我处理一下")], "intent": "clarify",
             "sub_intent": None}
    r = clarify_node(state)
    assert r["needs_clarify"] is True
    assert r["needs_human"] is False


# ---------------------------------------------------------------------------
# planner 多步规划 + 条件分支
# ---------------------------------------------------------------------------


def test_has_conditional_logic():
    assert _has_conditional_logic("如果订单没发货就取消") is True
    assert _has_conditional_logic("没发货就取消，发货了就改地址") is True
    assert _has_conditional_logic("帮我取消订单") is False


def test_detect_condition_branches():
    branches = _detect_condition_branches("没发货就取消，发货了就改地址")
    # 提取 (command, status) 对比较
    pairs = {(b["command"], b["status"]) for b in branches}
    assert ("cancel_order", "pending_ship") in pairs
    assert ("update_address", "shipped") in pairs


def test_build_plan_conditional_branches():
    # 条件分支 → 先查状态 + 带 condition 的写步骤
    plan = _build_plan("multi_step", "没发货就取消，发货了就改地址", None)
    assert plan[0]["tool"] == "query_order"
    # 后续步骤带 condition
    assert any(p.get("condition") for p in plan[1:])
    # cancel_order 的条件是 pending_ship，update_address 的条件是 shipped
    conditions = {p["tool"]: p["condition"] for p in plan if p.get("condition")}
    assert conditions["cancel_order"]["value"] == "pending_ship"
    assert conditions["update_address"]["value"] == "shipped"


def test_build_plan_single_write():
    # 无条件的单一写操作 → 先查后写二步，无 condition
    plan = _build_plan("multi_step", "帮我取消订单X", "cancel_order")
    assert [p["tool"] for p in plan] == ["query_order", "cancel_order"]
    assert all("condition" not in p for p in plan)


# ---------------------------------------------------------------------------
# executor 条件分支判断
# ---------------------------------------------------------------------------


def test_condition_met_true():
    scratchpad = [
        {"step": 1, "tool": "query_order",
         "result": {"status": "ok", "data": {"status": "pending_ship"}}}
    ]
    cond = {"field": "status", "op": "==", "value": "pending_ship"}
    assert _condition_met(cond, scratchpad) is True


def test_condition_met_false():
    scratchpad = [
        {"step": 1, "tool": "query_order",
         "result": {"status": "ok", "data": {"status": "shipped"}}}
    ]
    cond = {"field": "status", "op": "==", "value": "pending_ship"}
    assert _condition_met(cond, scratchpad) is False


def test_condition_met_no_result():
    assert _condition_met({"field": "status", "op": "==", "value": "x"}, []) is False


# ---------------------------------------------------------------------------
# 图级路由冒烟
# ---------------------------------------------------------------------------


def _invoke(text, tid):
    app = build_graph(get_checkpointer())
    return app.invoke(
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": tid}},
    )


def test_graph_routes_knowledge_to_rag():
    r = _invoke("退款政策是什么", "rt-1")
    # knowledge 分支走 rag（无知识库命中则转人工，但 intent 应是 knowledge）
    assert r["intent"] == "knowledge"


def test_graph_routes_write_to_planner():
    # 播种待发货订单
    with get_session() as s:
        s.add(Session(thread_id="rt-2", user_id="user-1", status="active"))
        o = Order(order_id="RT-1", user_id="user-1", status=OrderStatus.PENDING_SHIP.value,
                  amount=Decimal("199.00"), paid=True, stock_deducted=True)
        s.add(o)
    r = _invoke("帮我取消订单RT-1", "rt-2")
    assert r["intent"] == "multi_step"
    assert [x["tool"] for x in r["scratchpad"]] == ["query_order", "cancel_order"]


def test_graph_conditional_branch_cancel():
    # 订单待发货 → 「没发货就取消」条件满足，走取消分支
    with get_session() as s:
        s.add(Session(thread_id="rt-3", user_id="user-1", status="active"))
        o = Order(order_id="RT-2", user_id="user-1", status=OrderStatus.PENDING_SHIP.value,
                  amount=Decimal("199.00"), paid=True, stock_deducted=True)
        s.add(o)
    r = _invoke("订单RT-2如果没发货就取消，发货了就改地址", "rt-3")
    assert r["intent"] == "multi_step"
    # scratchpad 应有 query_order（查状态）+ cancel_order（条件满足执行）+ 跳过的 update_address
    tools = [x["tool"] for x in r["scratchpad"]]
    assert "query_order" in tools
    assert "cancel_order" in tools
    # 订单是 pending_ship → update_address 条件（shipped）不满足 → 应跳过
    update_records = [x for x in r["scratchpad"] if x["tool"] == "update_address"]
    assert update_records and update_records[0]["result"].get("status") == "skipped"
    # 最终答案包含取消成功
    assert "已取消" in r["final_answer"]


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    from app.db.models.audit import AuditLog
    from app.db.models.idempotency import IdempotencyRecord
    from app.memory.mysql_saver import MySQLSaver

    saver = MySQLSaver()
    with get_session() as s:
        s.query(IdempotencyRecord).delete(synchronize_session=False)
        s.query(AuditLog).filter(AuditLog.thread_id.like("rt-%")).delete(synchronize_session=False)
        s.query(AuditLog).filter(AuditLog.thread_id.like("q-%")).delete(synchronize_session=False)
        s.query(OrderItem).filter(OrderItem.order_id.like("Q-%")).delete(synchronize_session=False)
        s.query(OrderItem).filter(OrderItem.order_id.like("RT-%")).delete(synchronize_session=False)
        s.query(Order).filter(Order.order_id.like("Q-%")).delete(synchronize_session=False)
        s.query(Order).filter(Order.order_id.like("RT-%")).delete(synchronize_session=False)
        s.query(Session).filter(Session.thread_id.like("q-%")).delete(synchronize_session=False)
        s.query(Session).filter(Session.thread_id.like("rt-%")).delete(synchronize_session=False)
    for tid in ["rt-1", "rt-2", "rt-3"]:
        saver.delete_thread(tid)
