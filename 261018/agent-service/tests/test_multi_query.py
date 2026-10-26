"""多领域多子问题编排测试：拆解 → 依赖图 → 并行/串行 → 汇总报告。

运行：python -m pytest tests/test_multi_query.py -v
"""
from __future__ import annotations

import time

import pytest

from app.commands.multi_query import (
    split_sub_questions,
    execute_multi_query,
    execute_writes,
    summarize_results,
    MultiQueryPlan,
    SubQuestion,
)
from app.commands import detect_conflicts


# ---------------------------------------------------------------------------
# 子问题拆解
# ---------------------------------------------------------------------------


def test_split_order_and_product():
    subs = split_sub_questions("帮我查订单1234的状态，顺便看下商品库存")
    domains = {s.domain for s in subs}
    assert "order" in domains
    assert "product" in domains


def test_split_single_domain():
    subs = split_sub_questions("我要退款订单A1")
    assert len(subs) == 1
    assert subs[0].domain == "order"


def test_split_no_domain():
    assert split_sub_questions("你好") == []


def test_split_caps_at_max():
    # 构造包含多领域的长文本（这里只验证不超过上限，实际拆解数由领域数决定）
    text = "订单 商品 优惠券 库存 退款 退货 地址 价格 售后 发货 " * 10
    subs = split_sub_questions(text)
    assert len(subs) <= 8


# ---------------------------------------------------------------------------
# 执行：并行只读 + 局部失败如实告知
# ---------------------------------------------------------------------------


def _fake_execute(success_map):
    """构造一个 execute 函数，按命令返回成功/失败。"""
    def execute(command, params):
        ok = success_map.get(command, True)
        if ok:
            return {"tool": command, "success": True, "data": {"order_id": params.get("order_id", ""), "status": "ok"}}
        return {"tool": command, "success": False, "data": None, "error": "查询失败"}
    return execute


def test_execute_multi_query_all_success():
    subs = [
        SubQuestion(qid="q1", domain="order", text="查订单", commands=["query_order"], params_map={"query_order": {"order_id": "O1"}}),
        SubQuestion(qid="q2", domain="product", text="查商品", commands=["query_product"], params_map={"query_product": {"product_id": "P1"}}),
    ]
    plan = MultiQueryPlan(sub_questions=subs)
    result = execute_multi_query(plan, _fake_execute({}))
    assert result.all_success is True
    assert result.failed_domains == []


def test_execute_multi_query_partial_failure_reported():
    """某领域失败 → 明确说明，不隐藏成整体成功。"""
    subs = [
        SubQuestion(qid="q1", domain="order", text="查订单", commands=["query_order"], params_map={"query_order": {"order_id": "O1"}}),
        SubQuestion(qid="q2", domain="product", text="查商品", commands=["query_product"], params_map={"query_product": {"product_id": "P1"}}),
    ]
    plan = MultiQueryPlan(sub_questions=subs)
    # query_product 失败
    result = execute_multi_query(plan, _fake_execute({"query_product": False}))
    assert result.all_success is False
    assert "product" in result.failed_domains
    # 汇总明确说明「无法确认」
    assert "无法确认" in result.summary or "失败" in result.summary


def test_execute_multi_query_conflict_rejected():
    """矛盾命令 → 拒绝执行，返回澄清提示。"""
    subs = [
        SubQuestion(qid="q1", domain="order", text="取消", commands=["cancel_order"], params_map={"cancel_order": {"order_id": "O1"}}),
        SubQuestion(qid="q2", domain="order", text="退款", commands=["apply_refund"], params_map={"apply_refund": {"order_id": "O1"}}),
    ]
    plan = MultiQueryPlan(sub_questions=subs)
    result = execute_multi_query(plan, _fake_execute({}))
    assert result.all_success is False
    assert "冲突" in result.summary


def test_parallel_reads_actually_concurrent():
    """无依赖只读查询应并发（总耗时 < 串行耗时）。"""
    call_times = []

    def slow_execute(command, params):
        call_times.append(time.monotonic())
        time.sleep(0.1)
        return {"tool": command, "success": True, "data": {"order_id": params.get("order_id", "")}}

    subs = [
        SubQuestion(qid="q1", domain="order", text="查订单1", commands=["query_order"], params_map={"query_order": {"order_id": "O1"}}),
        SubQuestion(qid="q2", domain="order", text="查订单2", commands=["query_coupon"], params_map={"query_coupon": {"coupon_code": "C1"}}),
        SubQuestion(qid="q3", domain="product", text="查商品", commands=["query_product"], params_map={"query_product": {"product_id": "P1"}}),
    ]
    plan = MultiQueryPlan(sub_questions=subs)
    start = time.monotonic()
    execute_multi_query(plan, slow_execute, max_concurrency=4, per_query_timeout=5.0)
    elapsed = time.monotonic() - start
    # 3 个 0.1s 查询，若串行需 0.3s，并发应 < 0.25s
    assert elapsed < 0.25, f"并发失效，耗时 {elapsed}"


# ---------------------------------------------------------------------------
# 写操作单独控制流
# ---------------------------------------------------------------------------


def test_execute_writes_requires_confirm():
    """写操作未确认 → 不执行，要求确认。"""
    result = execute_writes(
        ["cancel_order"],
        {"cancel_order": {"order_id": "O1", "confirm": False}},
        _fake_execute({}),
        require_confirm=True,
    )
    assert result.all_success is False
    assert "确认" in result.summary


def test_execute_writes_conflict_rejected():
    """矛盾写操作 → 拒绝。"""
    result = execute_writes(
        ["cancel_order", "apply_refund"],
        {
            "cancel_order": {"order_id": "O1", "confirm": True},
            "apply_refund": {"order_id": "O1", "confirm": True},
        },
        _fake_execute({}),
        require_confirm=True,
    )
    assert result.all_success is False
    assert "冲突" in result.summary


def test_execute_writes_partial_failure_itemized():
    """部分写失败 → 逐项告知成功/失败。"""
    result = execute_writes(
        ["apply_refund", "update_address"],
        {
            "apply_refund": {"order_id": "O1", "confirm": True},
            "update_address": {"order_id": "O1", "address": "新地址", "confirm": True},
        },
        _fake_execute({"update_address": False}),
        require_confirm=True,
    )
    assert result.all_success is False
    assert "1 项成功" in result.summary
    assert "1 项失败" in result.summary


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------


def test_summarize_partial_reports_unconfirmed_part():
    sq = SubQuestion(
        qid="q1", domain="order", text="查订单",
        commands=["query_order", "query_coupon"],
        params_map={},
        evidence=[
            {"tool": "query_order", "success": True, "data": {"order_id": "O1", "status": "pending_ship"}},
            {"tool": "query_coupon", "success": False, "data": None, "error": "查询失败"},
        ],
    )
    sq.status = "partial"
    summary = summarize_results([sq])
    assert "部分查询成功" in summary
    assert "无法确认" in summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
