"""Reflector 节点：审视 executor 结果，决定继续下一步 / 结束 / 转人工。

确定性规则（竖切片阶段，后续换小模型）：
- 任一步 error → 转人工（不重试，M1 简化）
- 金额超阈值 / 写操作被拒 → 转人工
- 所有步骤执行完 → 结束，进 summarize
- 步数超上限（6）→ 转人工（防死循环）
"""
from __future__ import annotations

from app.graph.state import AgentState

# 最大步数（Spec 3.4：步数≤6，超限转人工）
MAX_STEPS = 6


def _check_result(scratchpad: list[dict]) -> tuple[bool, bool]:
    """检查 scratchpad，返回 (是否结束, 是否转人工)。"""
    for record in scratchpad:
        result = record.get("result") or {}
        error = record.get("error")

        if error == "need_approval":
            return True, True
        if error is not None:
            # 执行出错（权限/幂等/白名单/参数）→ 转人工
            return True, True
        # 写操作被业务规则拒绝（success=False 且非查询）
        if result.get("status") == "ok" and isinstance(result.get("data"), dict):
            data = result["data"]
            if data.get("success") is False:
                # 业务规则拒绝（如「已发货不可取消」）→ 转人工引导
                return True, True
    return False, False


def reflector_node(state: AgentState) -> dict:
    """Reflector：判断继续 / 结束 / 转人工。"""
    scratchpad = list(state.get("scratchpad", []))
    plan = state.get("plan", [])
    idx = state.get("plan_index", 0)

    # 步数超限 → 转人工
    if len(scratchpad) > MAX_STEPS:
        return {
            "plan_done": True,
            "needs_human": True,
            "final_answer": "处理步骤过多，已为您转接人工客服。",
            "scratchpad": scratchpad,
        }

    # 任一步出错 → 转人工
    done, needs_human = _check_result(scratchpad)
    if done:
        return {
            "plan_done": True,
            "needs_human": needs_human,
            "scratchpad": scratchpad,
        }

    # 还有剩余步骤 → 继续
    if idx < len(plan):
        return {"plan_done": False, "scratchpad": scratchpad}

    # 所有步骤执行完且无错 → 结束，进 summarize
    return {"plan_done": True, "needs_human": False, "scratchpad": scratchpad}
