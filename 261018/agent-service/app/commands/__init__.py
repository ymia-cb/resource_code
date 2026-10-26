"""命令模型：业务层定义「原子命令」，Agent 层按依赖/风险编排。

核心原则（grill 会话收敛）：
- 模型永不负责拼事务。事务边界由服务端命令定义。
- 命令分三级原子性（见 CommandAtomicity）。
- 聚合命令有自己的幂等键，避免超时重试重复操作。
- 弱关联批量命令允许部分完成，逐项记录结果并告知用户。

命令 vs 工具：
- 「工具」（ToolSpec）是最小执行单元，经 execute_tool 五道闸门。
- 「命令」（Command）是业务语义单元：单操作命令 = 1 个工具；聚合命令 =
  多个子步骤（强关联同事务 / 弱关联逐项）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandAtomicity(str, Enum):
    """命令的原子性级别。"""

    SINGLE = "single"  # 单操作命令（1 个工具，天然原子）
    ATOMIC = "atomic"  # 强关联聚合：同一事务全成全败，单一幂等键
    PARTIAL = "partial"  # 弱关联批量：允许部分完成，逐项幂等 + 逐项告知


class ConflictKind(str, Enum):
    """命令间冲突类型。"""

    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # 互斥（如取消 vs 继续发货）
    SAME_RESOURCE_WRITE = "same_resource_write"  # 同资源写（需串行）
    DEPENDENCY = "dependency"  # 依赖（必须先执行 A 才能执行 B）


@dataclass
class CommandSpec:
    """一个业务命令的定义。"""

    name: str  # 命令名（稳定标识）
    description: str  # 业务语义描述（供提示词/编排）
    atomicity: CommandAtomicity
    tool_names: list[str]  # 命令包含的工具（单操作 = 1 个；聚合 = 多个子步骤）
    skill: str | None = None  # 归属 skill
    idempotent_keys: tuple[str, ...] = ()  # 参与命令级幂等键的业务字段
    depends_on: tuple[str, ...] = ()  # 依赖的其他命令名（拓扑排序用）
    conflicts_with: tuple[str, ...] = ()  # 互斥的其他命令名（矛盾检测用）


# ---------------------------------------------------------------------------
# 命令目录：电商客服已知命令
# ---------------------------------------------------------------------------

# 互斥对（矛盾请求）：这些操作不能同时出现，应拒绝或要求澄清
_MUTUAL_EXCLUSION = {
    # 取消订单 与 继续发货/退款/退货/改址 互斥（订单状态互斥）
    ("cancel_order", "apply_refund"),
    ("cancel_order", "apply_return"),
    ("cancel_order", "update_address"),
    ("cancel_order", "continue_ship"),
    # 退款 与 退货 通常二选一（同一订单售后动作互斥）
    ("apply_refund", "apply_return"),
}

COMMAND_CATALOG: dict[str, CommandSpec] = {
    "query_order": CommandSpec(
        name="query_order",
        description="查询订单状态与详情",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["query_order"],
        skill="order",
        idempotent_keys=("order_id",),
    ),
    "query_coupon": CommandSpec(
        name="query_coupon",
        description="查询优惠券/活动规则",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["query_coupon"],
        skill="order",
        idempotent_keys=("coupon_code",),
    ),
    "query_product": CommandSpec(
        name="query_product",
        description="查询商品信息/库存",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["query_product"],
        skill="product",
        idempotent_keys=("product_id",),
    ),
    "cancel_order": CommandSpec(
        name="cancel_order",
        description="取消订单（内部含：改状态+退款+回补库存+回退优惠券，强关联原子）",
        atomicity=CommandAtomicity.ATOMIC,
        tool_names=["cancel_order"],
        skill="order",
        idempotent_keys=("order_id",),
        depends_on=("query_order",),  # 先查订单状态再取消
        conflicts_with=("continue_ship", "apply_refund", "apply_return", "update_address"),
    ),
    "apply_refund": CommandSpec(
        name="apply_refund",
        description="发起退款（金额超阈值转人工）",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["apply_refund"],
        skill="order",
        idempotent_keys=("order_id",),
        depends_on=("query_order",),  # 先查订单状态再退款
        conflicts_with=("cancel_order", "apply_return"),
    ),
    "apply_return": CommandSpec(
        name="apply_return",
        description="申请退货/换货",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["apply_return"],
        skill="order",
        idempotent_keys=("order_id",),
        depends_on=("query_order",),  # 先查订单状态再退货
        conflicts_with=("cancel_order", "apply_refund"),
    ),
    "update_address": CommandSpec(
        name="update_address",
        description="修改收货地址（仅待发货可改）",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=["update_address"],
        skill="order",
        idempotent_keys=("order_id", "address"),
        depends_on=("query_order",),  # 先查订单状态再改址
        conflicts_with=("cancel_order",),
    ),
    # 继续发货：占位命令（本系统无该工具，仅用于矛盾检测示例）
    "continue_ship": CommandSpec(
        name="continue_ship",
        description="继续发货（与取消订单互斥）",
        atomicity=CommandAtomicity.SINGLE,
        tool_names=[],
        skill="order",
        conflicts_with=("cancel_order",),
    ),
}


def get_command(name: str) -> CommandSpec | None:
    """按名查命令。"""
    return COMMAND_CATALOG.get(name)


def conflicts_between(a: str, b: str) -> bool:
    """判断两个命令是否互斥（对称）。"""
    ca = get_command(a)
    cb = get_command(b)
    if ca is None or cb is None:
        return False
    return b in ca.conflicts_with or a in cb.conflicts_with


def detect_conflicts(command_names: list[str]) -> list[tuple[str, str]]:
    """检测一组命令中的互斥/矛盾对。

    Returns:
        [(cmd_a, cmd_b), ...] 互斥对列表。空列表表示无矛盾。
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(command_names):
        for b in command_names[i + 1 :]:
            if conflicts_between(a, b):
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    pairs.append((a, b))
    return pairs


def is_write_command(name: str) -> bool:
    """判断命令是否含写操作（有副作用）。"""
    spec = get_command(name)
    if spec is None:
        return False
    # 只读命令白名单
    _READONLY = {"query_order", "query_coupon", "query_product"}
    return name not in _READONLY
