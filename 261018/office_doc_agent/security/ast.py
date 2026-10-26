"""语义 AST + typed op：把文档结构表示为带 semantic_tag 的节点，变更表示为 typed op。

semantic_tag 用于策略引擎判定「这个变更属于动作域 A1-A4 还是形式域/事实域」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Domain(Enum):
    """变更所属域。"""

    ACTION = "action"    # 动作域（结构/边界）
    FACT = "fact"        # 事实域（值）
    FORM = "form"        # 形式域（呈现）


class ActionLevel(Enum):
    """动作域四级分级。"""

    A1_EXTERNAL = "A1"    # 外部副作用：发送/签署/写回
    A2_SEMANTIC = "A2"    # 语义结构变更：改承诺/义务/披露
    A3_STRUCTURAL = "A3"  # 一般结构变更：改组织不改承诺
    A4_PRESENTATIONAL = "A4"  # 呈现性结构：只影响阅读


class Op(Enum):
    """变更算子。"""

    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    REORDER = "reorder"


@dataclass
class Node:
    """语义 AST 节点，带 semantic_tag。"""

    node_id: str
    node_type: str
    semantic_tag: str            # 如 "clause" / "disclaimer" / "signature" / "attachment"
    children: list["Node"] = field(default_factory=list)


@dataclass
class TypedOp:
    """一次变更，表示为 typed op。"""

    op: Op
    node_id: str
    node_type: str
    semantic_tag: str
    requester: str = ""          # 操作者，策略引擎据此判定授权


def controlled_elements(nodes: list[Node]) -> set[str]:
    """受控要素集合（A2 级节点的 semantic_tag 集合）。"""
    tags: set[str] = set()
    for n in nodes:
        tags.add(n.semantic_tag)
        tags.update(controlled_elements(n.children))
    return tags
