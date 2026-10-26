"""Summarize 节点：基于 scratchpad 汇总最终答案 + 事实校验。

原则（防幻觉）：
- 只依据 scratchpad 里工具成功结果的事实，不臆造。
- 生成回答后，提取回答中的业务实体，用「工具成功结果」做事实校验；
  出现不在事实源里的实体（疑似幻觉）→ 剔除并转人工（宁可转人工不可给错信息）。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph.state import AgentState
from app.graph.grounding import (
    build_fact_source,
    verify_grounding,
)


def summarize_node(state: AgentState) -> dict:
    """汇总 scratchpad 的执行结果，生成最终答案，并做事实校验。"""
    scratchpad = state.get("scratchpad", [])
    needs_human = state.get("needs_human", False)

    if needs_human:
        answer = state.get("final_answer") or "已为您转接人工客服。"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "sources": [],
        }

    # 提取最后一个写操作的成功结果作为答案依据
    last_write = None
    for record in scratchpad:
        result = record.get("result") or {}
        if result.get("status") == "ok" and isinstance(result.get("data"), dict):
            data = result["data"]
            if data.get("success") is True and data.get("message"):
                last_write = data["message"]

    if last_write:
        answer = f"{last_write}。"
    else:
        # 无成功写操作：可能是条件分支全部跳过，或纯查询场景
        answer = _conditional_skip_answer(scratchpad) or "处理完成。"

    # 部分完成检测：弱关联批量命令允许多个写操作，逐项告知成功/失败
    answer = _merge_partial_results(scratchpad, answer)

    # 事实校验：summarize 为确定性拼接（无模型自报 entities），走兜底文本提取核对
    fact_source = build_fact_source(scratchpad, sources=state.get("sources", []))
    hallucinated = verify_grounding(declared_entities=[], answer_text=answer, fact_source=fact_source)

    if hallucinated:
        # 回答中出现疑似幻觉实体（不在工具成功结果里）→ 转人工，不给不可靠信息
        answer = "抱歉，处理结果需要人工进一步确认，已为您转接人工客服。"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "sources": [],
            "needs_human": True,
        }

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "sources": [],
    }


def _conditional_skip_answer(scratchpad: list[dict]) -> str | None:
    """条件分支全部跳过时的说明。

    当 plan 带条件分支（有 skipped 步骤）且没有成功写操作时，
    说明「订单当前状态不满足操作条件」，而非笼统的「处理完成」。
    """
    skipped = [r for r in scratchpad if (r.get("result") or {}).get("status") == "skipped"]
    # 有成功写操作 → 返回 None（由主流程用 last_write 回答）
    has_success_write = any(
        (r.get("result") or {}).get("status") == "ok"
        and isinstance((r.get("result") or {}).get("data"), dict)
        and (r.get("result") or {}).get("data", {}).get("success") is True
        for r in scratchpad
    )
    if skipped and not has_success_write:
        # 取查询步骤得到的订单状态，用于友好说明
        status_desc = _last_queried_status(scratchpad)
        if status_desc:
            return f"已为您查询：{status_desc}。当前订单状态不满足您要求的操作条件，未执行相关操作。"
        return "当前订单状态不满足操作条件，未执行相关操作。"
    return None


def _last_queried_status(scratchpad: list[dict]) -> str | None:
    """从查询步骤结果里取订单状态的中文描述。"""
    status_label = {
        "pending_payment": "待付款",
        "pending_ship": "待发货",
        "shipped": "已发货",
        "completed": "已完成",
        "cancelled": "已取消",
        "refunding": "退款中",
        "returning": "退货中",
    }
    for record in scratchpad:
        result = record.get("result") or {}
        data = result.get("data") if isinstance(result, dict) else None
        if isinstance(data, dict) and "status" in data:
            raw = data["status"]
            return status_label.get(raw, raw)
    return None


def _merge_partial_results(scratchpad: list[dict], default_answer: str) -> str:
    """逐项汇总写操作结果，支持部分完成时明确告知哪些成功、哪些失败。

    - 无写操作 → 返回 default_answer。
    - 全部成功 → 返回 default_answer（或合并各成功消息）。
    - 部分成功 → 逐项列出「成功：X / 失败：Y」，明确告知。
    """
    # 收集所有写操作（有 success 字段的）结果
    writes: list[tuple[str, bool, str]] = []  # (tool, success, message)
    for record in scratchpad:
        result = record.get("result") or {}
        if result.get("status") != "ok":
            continue
        data = result.get("data")
        if not isinstance(data, dict) or "success" not in data:
            continue
        tool = record.get("tool", "未知操作")
        success = bool(data.get("success"))
        message = str(data.get("message", ""))
        writes.append((tool, success, message))

    if not writes:
        return default_answer

    succeeded = [w for w in writes if w[1]]
    failed = [w for w in writes if not w[1]]

    # 全部成功 → 返回默认（单一写）或合并
    if not failed:
        if len(succeeded) == 1:
            return f"{succeeded[0][2]}。"
        parts = "；".join(f"{w[2]}" for w in succeeded if w[2])
        return f"{parts}。"

    # 部分成功 → 逐项告知
    lines: list[str] = []
    for tool, success, message in writes:
        mark = "✅" if success else "❌"
        desc = message if message else tool
        lines.append(f"{mark} {desc}")

    ok_count = len(succeeded)
    fail_count = len(failed)
    summary = (
        f"部分操作已完成：{ok_count} 项成功，{fail_count} 项失败。\n"
        + "\n".join(lines)
        + "\n失败的项已为您转接人工客服处理。"
    )
    return summary
