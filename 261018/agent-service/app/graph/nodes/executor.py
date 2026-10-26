"""Executor 节点：执行 plan 的当前步骤（经 execute_tool 五道闸门）。

结果写回 scratchpad，供 reflector 判断与 summarize 汇总。

thread_id 从 config 注入（LangGraph 自动传入 RunnableConfig），
用于 execute_tool 的权限校验（resolve_user）。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.graph.state import AgentState
from app.tools.registry import execute_tool
from app.tools.errors import ToolError, NeedApproval
from app.observability.run_tracker import record_tool_execution
from app.middleware import get_default_chain, ToolContext
from app.middleware.tool_timeout import run_tool_with_timeout, ToolTimeoutMiddleware


def executor_node(state: AgentState, config: RunnableConfig) -> dict:
    """执行 plan[plan_index]，结果追加到 scratchpad。"""
    plan = state.get("plan", [])
    idx = state.get("plan_index", 0)
    scratchpad = list(state.get("scratchpad", []))
    thread_id = str(config["configurable"].get("thread_id", "unknown"))
    run_id = str(config["configurable"].get("run_id", ""))
    # 当前激活的 skill（渐进披露：工具按 skill 隔离）
    active_skill = state.get("active_skill")

    if idx >= len(plan):
        return {"plan_done": True, "scratchpad": scratchpad}

    step = plan[idx]
    tool_name = step["tool"]
    params = step.get("params", {})

    # 条件分支：step 带 condition 时，先根据前面查询结果判断是否执行
    condition = step.get("condition")
    if condition is not None and not _condition_met(condition, scratchpad):
        # 条件不满足 → 跳过该步，标记 skipped，不调用工具
        record = {
            "step": step["step"],
            "tool": tool_name,
            "params": params,
            "description": step.get("description", ""),
            "result": {"status": "skipped", "reason": "条件不满足，跳过"},
            "error": None,
        }
        scratchpad.append(record)
        return {"scratchpad": scratchpad, "plan_index": idx + 1}

    # 过滤 None 参数（如 update_address 未提取到 address）
    params = {k: v for k, v in params.items() if v is not None}

    record = {
        "step": step["step"],
        "tool": tool_name,
        "params": params,
        "description": step.get("description", ""),
        "result": None,
        "error": None,
    }

    # 中间件：工具调用前（skill 隔离校验）
    chain = get_default_chain()
    try:
        tool_ctx = chain.before_tool(ToolContext(name=tool_name, params=params, active_skill=active_skill))
    except ToolError as e:
        record["result"] = {"status": "error", "code": e.code, "reason": str(e)}
        record["error"] = e.code
        _record_exec(run_id, tool_name, "error", error=str(e))
        scratchpad.append(record)
        return {"scratchpad": scratchpad, "plan_index": idx + 1}

    try:
        # 工具超时保护（按业务类型差异化超时）
        timeout = ToolTimeoutMiddleware().timeout_for(tool_name)
        timed_out, result = run_tool_with_timeout(
            execute_tool, timeout, tool_name, params, thread_id, active_skill
        )
        if timed_out:
            record["result"] = {"status": "error", "code": "E_TOOL_TIMEOUT", "reason": f"工具 {tool_name} 执行超时（>{timeout}s）"}
            record["error"] = "timeout"
            _record_exec(run_id, tool_name, "error", error=f"timeout >{timeout}s")
            scratchpad.append(record)
            return {"scratchpad": scratchpad, "plan_index": idx + 1}
        if isinstance(result, Exception):
            raise result
        # 中间件：工具调用后
        result = chain.after_tool(tool_ctx, result)
        record["result"] = result
        _record_exec(run_id, tool_name, "success", result)
    except NeedApproval as e:
        # 金额超阈值 → 转人工（不终止，但标记）
        record["result"] = {"status": "pending_human", "reason": str(e)}
        record["error"] = "need_approval"
        _record_exec(run_id, tool_name, "need_approval", error=str(e))
        scratchpad.append(record)
        return {
            "scratchpad": scratchpad,
            "plan_done": True,
            "needs_human": True,
            "final_answer": "该操作涉及大额金额，已为您转接人工客服处理。",
        }
    except ToolError as e:
        record["result"] = {"status": "error", "code": e.code, "reason": str(e)}
        record["error"] = e.code
        _record_exec(run_id, tool_name, "error", error=str(e))
    except Exception as e:  # noqa: BLE001
        record["result"] = {"status": "error", "reason": str(e)}
        record["error"] = "unknown"
        _record_exec(run_id, tool_name, "error", error=str(e))

    scratchpad.append(record)
    return {"scratchpad": scratchpad, "plan_index": idx + 1}


def _record_exec(run_id: str, tool_name: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    """记录工具执行状态（关联 run_id，供追踪/指标）。"""
    if not run_id:
        return
    try:
        record_tool_execution(run_id, tool_name, status, result=result, error=error)
    except Exception:
        # 追踪失败不影响主流程
        pass


def _condition_met(condition: dict, scratchpad: list[dict]) -> bool:
    """判断条件分支是否满足。

    condition 形如 {"field": "status", "op": "==", "value": "pending_ship"}。
    从前面查询步骤（query_order/query_logistics）的结果里取 field 值比对。

    找不到对应查询结果 → 返回 False（保守跳过，避免错误执行写操作）。
    """
    field = condition.get("field")
    op = condition.get("op", "==")
    expected = condition.get("value")

    # 从前面的查询结果里找 field 值
    actual = None
    for record in scratchpad:
        result = record.get("result") or {}
        data = result.get("data") if isinstance(result, dict) else None
        if isinstance(data, dict) and field in data:
            actual = data[field]
            break

    if actual is None:
        return False

    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    # 默认：不满足
    return False
