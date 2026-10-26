"""语义表 + 策略引擎 + 不变量校验。

五级定域判据（按优先级，命中即定域）：
1. 越过文档边界（发送/签署/写回/付款）→ 动作域
2. 改变受控要素成员或顺序 → 动作域
3. 改变受控要素值 → 事实域
4. 改变呈现且不触碰 1-3 → 形式域
5. 无法归类 → 动作域（默认严）

不变量：controlled_elements(after) ⊇ controlled_elements(before)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ast import ActionLevel, Domain, Node, Op, TypedOp


@dataclass
class SemanticTable:
    """节点语义表：semantic_tag → 动作域级别。

    来源（按可信度）：受控模板库 > 条款库/CLM > 法规标准 > 历史语料。
    受控要素（A2）由模板契约声明，策略引擎据此派生。
    """

    doc_class: str = "contract"
    # semantic_tag → ActionLevel（受控要素标 A2，普通结构 A3，呈现 A4）
    tag_levels: dict[str, ActionLevel] = field(default_factory=dict)

    def level_of(self, semantic_tag: str) -> ActionLevel:
        return self.tag_levels.get(semantic_tag, ActionLevel.A3_STRUCTURAL)


@dataclass
class PolicyDecision:
    """策略引擎的一次判定结果。"""

    domain: Domain
    action_level: ActionLevel | None
    allowed: bool
    reason: str = ""


class PolicyEngine:
    """策略引擎：(doc_class, node_type, tag, op, requester) → domain + gate。"""

    def __init__(self, table: SemanticTable) -> None:
        self._table = table

    def decide(self, typed_op: TypedOp, crosses_boundary: bool = False) -> PolicyDecision:
        tag = typed_op.semantic_tag

        # 判据 1：越过文档边界 → 动作域 A1
        if crosses_boundary:
            return PolicyDecision(Domain.ACTION, ActionLevel.A1_EXTERNAL, False, "越过文档边界需双人授权")

        # 判据 2：改变受控要素成员/顺序（A2 级节点的 add/delete/reorder）
        level = self._table.level_of(tag)
        if level == ActionLevel.A2_SEMANTIC and typed_op.op in (Op.ADD, Op.DELETE, Op.REORDER):
            return PolicyDecision(Domain.ACTION, ActionLevel.A2_SEMANTIC, False, "语义结构变更需显式指令+人工确认")

        # 判据 3：改变值 → 事实域（由 claim 层拦截，这里只标记）
        if typed_op.op == Op.MODIFY and typed_op.node_type in ("value", "field", "cell"):
            return PolicyDecision(Domain.FACT, None, True, "事实变更由 claim 层拦截")

        # 判据 4：改呈现 → 形式域
        if level == ActionLevel.A4_PRESENTATIONAL:
            return PolicyDecision(Domain.FORM, ActionLevel.A4_PRESENTATIONAL, True, "呈现性结构自动放行")

        # 一般结构变更 A3：显式指令即可
        if level == ActionLevel.A3_STRUCTURAL:
            return PolicyDecision(Domain.ACTION, ActionLevel.A3_STRUCTURAL, True, "一般结构变更显式指令即可")

        # 判据 5：无法归类 → 动作域（默认严）
        return PolicyDecision(Domain.ACTION, ActionLevel.A2_SEMANTIC, False, "无法归类默认从严")


def check_invariant(before: list[Node], after: list[Node], table: SemanticTable) -> tuple[bool, str]:
    """不变量校验：受控要素（A2）集合只增不减。"""
    before_controlled = {
        n.semantic_tag for n in before if table.level_of(n.semantic_tag) == ActionLevel.A2_SEMANTIC
    }
    after_controlled = {
        n.semantic_tag for n in after if table.level_of(n.semantic_tag) == ActionLevel.A2_SEMANTIC
    }
    missing = before_controlled - after_controlled
    if missing:
        return False, f"受控要素被删除: {sorted(missing)}"
    return True, ""
