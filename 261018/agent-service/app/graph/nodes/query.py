"""Query 节点：一步查询（物流/订单状态等），轻量路径。

与 write（planner 重推理）的区别：
- query 只需单次工具调用即可得到答案，不需要 Plan-Execute-Reflect 重循环。
- 直接调对应查询工具 → 汇总 → 返回，跳过 planner/reflector。

选工具：确定性规则（物流→query_logistics，其余订单/优惠券/商品→query_order）。
真实场景可换小模型决定工具，此处规则保证链路可跑、可测。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph.state import AgentState
from app.tools.registry import execute_tool
from app.tools.errors import ToolError
from app.middleware import get_default_chain, ToolContext
from app.middleware.tool_timeout import run_tool_with_timeout, ToolTimeoutMiddleware


def _pick_query_tool(text: str) -> str:
    """确定性选择查询工具：物流关键词→query_logistics，否则 query_order。"""
    logistics_keywords = ["物流", "到哪", "快递", "单号", "配送", "到货", "签收", "运输", "轨迹"]
    if any(k in text for k in logistics_keywords):
        return "query_logistics"
    return "query_order"


def _extract_order_id(text: str) -> str | None:
    """从文本提取订单号（与 planner 一致）。"""
    import re

    m = re.search(r"订单\s*([A-Za-z0-9\-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z0-9\-]{4,})", text)
    return m.group(1) if m else None


def query_node(state: AgentState, config=None) -> dict:
    """一步查询：选工具 → 执行 → 汇总，不经重推理循环。"""
    last = state["messages"][-1].content if state.get("messages") else ""
    tool_name = _pick_query_tool(last)
    order_id = _extract_order_id(last)
    active_skill = state.get("active_skill")
    thread_id = "unknown"
    if config and config.get("configurable"):
        thread_id = str(config["configurable"].get("thread_id", "unknown"))

    # 缺订单号 → 转澄清（反问补充）
    if order_id is None:
        question = "请问您要查询哪个订单？请提供订单号，以便我帮您查询。"
        return {
            "messages": [AIMessage(content=question)],
            "final_answer": question,
            "needs_clarify": True,
            "needs_human": False,
        }

    # 中间件：工具调用前校验（skill 隔离）
    chain = get_default_chain()
    try:
        chain.before_tool(ToolContext(name=tool_name, params={"order_id": order_id}, active_skill=active_skill))
    except ToolError as e:
        return {
            "messages": [AIMessage(content=f"抱歉，暂时无法为您查询：{e}。")],
            "final_answer": f"抱歉，暂时无法为您查询。",
            "needs_human": True,
        }

    # 执行（带超时保护）
    timeout = ToolTimeoutMiddleware().timeout_for(tool_name)
    timed_out, result = run_tool_with_timeout(
        execute_tool, timeout, tool_name, {"order_id": order_id}, thread_id, active_skill
    )
    if timed_out:
        answer = "查询超时，请稍后重试。"
        return {"messages": [AIMessage(content=answer)], "final_answer": answer, "needs_human": False}

    if isinstance(result, Exception) or result is None:
        answer = "抱歉，未查询到该订单信息，请核对订单号后重试。"
        return {"messages": [AIMessage(content=answer)], "final_answer": answer, "needs_human": False}

    # execute_tool 返回 {"status": "ok", "data": {...}}，取 data 里的业务数据
    data = result.get("data") if isinstance(result, dict) else None
    if not data:
        answer = "抱歉，未查询到该订单信息，请核对订单号后重试。"
        return {"messages": [AIMessage(content=answer)], "final_answer": answer, "needs_human": False}

    answer = _format_query_answer(tool_name, data)
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "needs_human": False,
    }


def _format_query_answer(tool_name: str, result: dict) -> str:
    """把查询工具结果格式化为自然语言回答。"""
    if tool_name == "query_logistics":
        desc = result.get("logistics_desc", "暂无物流信息")
        return f"您的订单 {result.get('order_id', '')} 当前物流状态：{desc}。"
    status = result.get("status", "未知")
    status_label = {
        "pending_payment": "待付款",
        "pending_ship": "待发货",
        "shipped": "已发货",
        "completed": "已完成",
        "cancelled": "已取消",
        "refunding": "退款中",
        "returning": "退货中",
    }.get(status, status)
    return f"您的订单 {result.get('order_id', '')} 当前状态：{status_label}。"
