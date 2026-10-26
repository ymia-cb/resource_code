"""安全模型：输入源治理 + provenance 回查 + 语义变更策略。

组成：
- claim.py     : 事件模型（claim + span + 目标字段）
- channel.py   : 四通道处置队列（自动放行/阻断/人工/抽样）
- ast.py       : 语义 AST + typed op
- policy.py    : 语义表 + 策略引擎 + 不变量校验
- whitelist.py : C 层白名单归一化
"""
from .claim import Channel, Claim, EventType, Layer
from .channel import ClaimQueue
from .ast import ActionLevel, Domain, Node, Op, TypedOp
from .policy import PolicyEngine, SemanticTable, check_invariant
from .whitelist import check_authorized, extract_numbers, normalize

__all__ = [
    "Channel", "Claim", "EventType", "Layer",
    "ClaimQueue",
    "ActionLevel", "Domain", "Node", "Op", "TypedOp",
    "PolicyEngine", "SemanticTable", "check_invariant",
    "check_authorized", "extract_numbers", "normalize",
]
