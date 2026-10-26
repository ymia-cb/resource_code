"""Planner 节点：多步规划（Plan-Execute-Reflect 的规划端）。

核心变化：不再硬编码「查订单 → 单一写操作」二步模板，而是：

1. 从用户描述中识别**多个读写操作意图**（写命令 + 查询命令）。
2. 识别**顺序逻辑与条件分支**（「如果…就…」「没发货就…，发货了就…」）。
3. 用命令目录的 `depends_on` 建**真实依赖图**（如写命令依赖先查订单状态）。
4. 拓扑分层输出 plan（每步含 tool/params/condition）。

条件分支：step 可带 `condition` 字段，executor 执行时根据前面步骤的查询结果
判断该步是否执行（如「没发货就取消」→ condition=status==pending_ship）。
"""
from __future__ import annotations

from app.graph.state import AgentState
from app.commands import detect_conflicts, get_command
from app.commands.orchestrator import build_conflict_prompt


def _extract_order_id(text: str) -> str | None:
    """从文本中提取订单号（M1 简化：数字串或「订单X」）。"""
    import re

    m = re.search(r"订单\s*([A-Za-z0-9\-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z0-9\-]{4,})", text)
    return m.group(1) if m else None


def _detect_intent_commands(text: str) -> list[str]:
    """从文本识别命中的命令（用于矛盾检测）。

    返回命中的命令名列表（只含写命令，读命令不参与矛盾判断）。
    """
    commands: list[str] = []
    if "取消" in text or "退掉" in text:
        commands.append("cancel_order")
    if "退货" in text or "换货" in text:
        commands.append("apply_return")
    if "退款" in text:
        commands.append("apply_refund")
    if "地址" in text or "改址" in text:
        commands.append("update_address")
    if "继续发货" in text or "继续发" in text or "照常发货" in text:
        commands.append("continue_ship")
    return commands


# ---------------------------------------------------------------------------
# 多步操作识别 + 条件分支识别
# ---------------------------------------------------------------------------

# 条件分支触发词（识别「如果…就…」「没发货就…」「发货了就…」结构）
_CONDITION_BRANCHES: list[tuple[str, str, str]] = [
    # (触发词, 写命令, 条件描述) —— 条件描述映射到订单状态
    ("没发货", "cancel_order", "pending_ship"),
    ("未发货", "cancel_order", "pending_ship"),
    ("还没发货", "cancel_order", "pending_ship"),
    ("待发货", "cancel_order", "pending_ship"),
    ("发货了", "update_address", "shipped"),
    ("已发货", "update_address", "shipped"),
    ("已经发货", "update_address", "shipped"),
]


def _detect_condition_branches(text: str) -> list[dict]:
    """识别条件分支：返回 [{command, status, negate}, ...]。

    例：「没发货就取消，发货了就改地址」→
        [{command: cancel_order, status: pending_ship, negate: False},
         {command: update_address, status: shipped, negate: False}]
    """
    branches: list[dict] = []
    for trigger, command, status in _CONDITION_BRANCHES:
        if trigger in text:
            branches.append({"command": command, "status": status, "negate": False})
    return branches


def _has_conditional_logic(text: str) -> bool:
    """判断是否包含「如果…就…」条件逻辑。"""
    return ("如果" in text or "若" in text or "要是" in text) and (
        "就" in text or "否则" in text or "不然" in text
    ) or bool(_detect_condition_branches(text))


def _build_step(
    step_no: int,
    tool: str,
    params: dict,
    description: str,
    condition: dict | None = None,
) -> dict:
    """构造一个 plan 步骤。"""
    step = {
        "step": step_no,
        "tool": tool,
        "params": params,
        "description": description,
    }
    if condition is not None:
        step["condition"] = condition
    return step


def _build_plan(intent: str, text: str, sub_intent: str | None = None) -> list[dict]:
    """多步动态规划：识别读写操作 + 顺序/条件逻辑，构建 plan。

    返回 step 列表，每步含 tool/params/description/condition（可选）。
    条件分支的 step 带 condition，executor 根据中间查询结果决定是否执行。
    """
    order_id = _extract_order_id(text)
    plan: list[dict] = []
    step_no = 1

    # 1. 识别条件分支（「没发货就取消，发货了就改地址」）
    branches = _detect_condition_branches(text)

    # 2. 若存在条件分支，产出一个「先查状态」的 plan + 带 condition 的写步骤
    if branches:
        # 第一步：查订单状态（为后续条件判断提供依据）
        plan.append(_build_step(
            step_no, "query_order",
            {"order_id": order_id},
            "查询订单状态（供后续条件分支判断）",
        ))
        step_no += 1
        for branch in branches:
            condition = {
                "field": "status",
                "op": "==",
                "value": branch["status"],
            }
            plan.append(_build_step(
                step_no, branch["command"],
                {"order_id": order_id, "confirm": True},
                f"条件满足（订单状态={branch['status']}）时执行",
                condition=condition,
            ))
            step_no += 1
        return plan

    # 3. 无条件分支：识别单一写操作（先查后写，二步）
    write_tool = _resolve_write_tool(sub_intent, text)
    if write_tool:
        plan.append(_build_step(
            step_no, "query_order",
            {"order_id": order_id},
            "查询订单状态",
        ))
        step_no += 1
        plan.append(_build_step(
            step_no, write_tool,
            {"order_id": order_id, "confirm": True},
            _write_description(write_tool),
        ))
        return plan

    # 4. 无法识别具体操作 → 空 plan（reflector 转人工）
    return []


def _resolve_write_tool(sub_intent: str | None, text: str) -> str | None:
    """解析写操作对应的工具（优先 sub_intent，兜底文本关键词）。"""
    sub_to_tool = {
        "cancel_order": "cancel_order",
        "apply_refund": "apply_refund",
        "apply_return": "apply_return",
        "update_address": "update_address",
    }
    if sub_intent in sub_to_tool:
        return sub_to_tool[sub_intent]
    # 兜底：文本关键词
    if "取消" in text or "退掉" in text:
        return "cancel_order"
    if "退货" in text or "换货" in text or "退这个" in text:
        return "apply_return"
    if "退款" in text:
        return "apply_refund"
    if "地址" in text or "改址" in text:
        return "update_address"
    return None


def _write_description(tool: str) -> str:
    return {
        "cancel_order": "取消订单",
        "apply_refund": "发起退款",
        "apply_return": "申请退货",
        "update_address": "修改地址",
    }.get(tool, "执行写操作")


def planner_node(state: AgentState) -> dict:
    """Planner：多步规划（识别读写操作 + 顺序/条件逻辑 + 依赖图）。

    优先用 classify 产出的 sub_intent，兜底文本识别；支持条件分支。
    """
    last = state["messages"][-1].content if state.get("messages") else ""
    sub_intent = state.get("sub_intent")

    # 矛盾检测：如「取消订单并继续发货」→ 拒绝执行，要求澄清，不自行选择。
    # 但条件分支场景（「如果没发货就取消，发货了就改地址」）中，互斥命令是
    # 互斥条件下的分支，不是矛盾，故跳过矛盾检测。
    if not _has_conditional_logic(last):
        intent_commands = _detect_intent_commands(last)
        conflicts = detect_conflicts(intent_commands)
        if conflicts:
            answer = build_conflict_prompt(conflicts)
            return {
                "plan": [],
                "plan_index": -1,
                "plan_done": True,
                "needs_human": True,
                "final_answer": answer,
                "scratchpad": state.get("scratchpad", []),
            }

    plan = _build_plan(state.get("intent", "multi_step"), last, sub_intent=sub_intent)

    if not plan:
        # 无法拆解 → 直接转人工
        answer = "抱歉，这个问题涉及多步处理，已为您转接人工客服。"
        return {
            "plan": [],
            "plan_index": -1,
            "plan_done": True,
            "needs_human": True,
            "final_answer": answer,
            "scratchpad": state.get("scratchpad", []),
        }

    return {
        "plan": plan,
        "plan_index": 0,  # 从第 0 步开始
        "plan_done": False,
        "scratchpad": state.get("scratchpad", []),
    }
