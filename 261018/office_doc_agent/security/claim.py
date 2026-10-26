"""claim 事件模型：provenance 回查的统一事件抽象。

每层出口拦截都产出 claim，带 span + 目标字段。人工审核进「规则 × 字段」队列，
不是按文档聚合。处置走四通道：自动放行 / 自动阻断 / 人工裁决 / 抽样复核。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Layer(Enum):
    """拦截发生所在层。"""

    A_SYMBOLIC = "A"
    B_READONLY = "B"
    C_FALLBACK = "C"
    DELIVERY = "delivery"


class Channel(Enum):
    """四通道处置。"""

    AUTO_BLOCK = "auto_block"      # 自动阻断（规则确定 + 高风险）
    AUTO_PASS = "auto_pass"        # 自动放行（规则确定 + 低风险）
    HUMAN = "human"                # 人工裁决（语义/新规则/容差边缘）
    SAMPLE = "sample"              # 抽样复核（监控自动通道本身）


class EventType(Enum):
    """claim 事件类型。"""

    REF_UNRESOLVED = "ref_unresolved"
    REF_FORBIDDEN = "ref_forbidden"
    REF_STALE = "ref_stale"
    LITERAL_IN_FACT_SLOT = "literal_in_fact_slot"
    VALUE_NOT_AUTHORIZED = "value_not_authorized"
    VALUE_NEAR_MISS = "value_near_miss"
    UNIT_MISMATCH = "unit_mismatch"
    MODIFIER_CONFLICT = "modifier_conflict"
    PRECISION_VIOLATION = "precision_violation"


@dataclass
class Claim:
    """一条 provenance 事件。"""

    event_type: EventType
    layer: Layer
    span: tuple[int, int]          # 文本区间（定位到具体 token）
    target_field: str | None       # 目标字段（规则 × 字段 聚合键之一）
    rule: str                      # 触发的规则名（聚合键之二）
    detail: str = ""
    expected: Any = None           # 期望值
    actual: Any = None             # 实际值
    doc_id: str = ""

    @property
    def queue_key(self) -> tuple[str, str]:
        """「规则 × 字段」聚合键。"""
        return (self.rule, self.target_field or "")
