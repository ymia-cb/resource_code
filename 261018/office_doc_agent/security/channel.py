"""四通道队列：把 claim 按事件类型路由到处置通道，按「规则 × 字段」聚合。

- 自动阻断：规则确定 + 高风险 → 不看，但留规则级申诉。
- 自动放行：规则确定 + 低风险 → 不看，抽样复核。
- 人工裁决：语义判断 / 新规则 / 容差边缘 → 必看。
- 抽样复核：监控自动通道本身。

申诉只走规则级：改规则/白名单批量放行，不允许单条豁免。
"""
from __future__ import annotations

from collections import defaultdict

from .claim import Channel, Claim, EventType

# 事件类型 → 通道 的静态映射（类型静态决定，不引入置信度判断）
_EVENT_CHANNEL: dict[EventType, Channel] = {
    EventType.REF_UNRESOLVED: Channel.AUTO_BLOCK,
    EventType.REF_FORBIDDEN: Channel.AUTO_BLOCK,
    EventType.REF_STALE: Channel.AUTO_BLOCK,
    EventType.LITERAL_IN_FACT_SLOT: Channel.HUMAN,       # 先尝试自动映射，失败转人工
    EventType.VALUE_NOT_AUTHORIZED: Channel.AUTO_BLOCK,
    EventType.VALUE_NEAR_MISS: Channel.HUMAN,            # 精确错 vs 合法舍入，必须人判
    EventType.UNIT_MISMATCH: Channel.HUMAN,              # 缺单位声明只能人判
    EventType.MODIFIER_CONFLICT: Channel.HUMAN,          # 纯语义
    EventType.PRECISION_VIOLATION: Channel.AUTO_PASS,    # 大多可自动重排
}


class ClaimQueue:
    """按「规则 × 字段」聚合 claim，路由到四通道。"""

    def __init__(self, auto_pass_sample_rate: float = 0.05, auto_block_sample_rate: float = 0.01) -> None:
        self.auto_pass_sample_rate = auto_pass_sample_rate
        self.auto_block_sample_rate = auto_block_sample_rate
        self._claims: list[Claim] = []
        self._by_channel: dict[Channel, list[Claim]] = defaultdict(list)

    def route(self, claim: Claim) -> Channel:
        """把单条 claim 路由到通道。"""
        channel = _EVENT_CHANNEL.get(claim.event_type, Channel.HUMAN)
        self._claims.append(claim)
        self._by_channel[channel].append(claim)
        return channel

    def aggregate(self) -> dict[tuple[str, str], list[Claim]]:
        """按「规则 × 字段」聚合，供人工审核队列消费。"""
        groups: dict[tuple[str, str], list[Claim]] = defaultdict(list)
        for c in self._claims:
            groups[c.queue_key].append(c)
        return dict(groups)

    def human_queue(self) -> list[Claim]:
        """人工裁决队列（必看）。"""
        return list(self._by_channel[Channel.HUMAN])

    def auto_blocked(self) -> list[Claim]:
        """自动阻断的 claim（留规则级申诉，不单条豁免）。"""
        return list(self._by_channel[Channel.AUTO_BLOCK])

    def auto_passed(self) -> list[Claim]:
        """自动放行的 claim（抽样复核）。"""
        return list(self._by_channel[Channel.AUTO_PASS])

    def has_blocking(self) -> bool:
        """是否存在会阻断交付的事件。"""
        return bool(self._by_channel[Channel.AUTO_BLOCK])
