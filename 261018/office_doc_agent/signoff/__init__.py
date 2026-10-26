"""Stage 6：人工审签留痕。

职责
----
记录签核动作、签核人、时间戳，并与文件版本指纹绑定，形成完整审计链。

关键决策
--------
- 审签记录是「合规风险」的最后防线：出了合规问题，要能查到「哪一版、哪份数据、谁签的」。
- 抽样规则 + 谁签核，由规格书的 acceptance 定义，本模块只负责「留痕」，不负责「决定抽谁」。

桩 → 真实现
------------
- StubSignoff：空实现，返回桩状态。
- AuditSignoff：真实现，写结构化审计记录（AuditRecord 实体）。
"""
from __future__ import annotations

from typing import Any


class StubSignoff:
    """【桩】空实现，供 MVP 空跑。真实现见下方 AuditSignoff。"""

    def record(self, doc_id: str, operator: str) -> dict:
        return {"doc_id": doc_id, "operator": operator, "status": "stub"}


class AuditSignoff:
    """【真实现·半成品】写结构化签核审计记录。

    审核注意：AuditRecord 实体字段（record_id/doc_id/stage/action/operator/
    timestamp/inputs_hash/outputs_hash/result）见数据模型设计。当前为内存骨架，
    落库（追加式、不可篡改）属 P3 填充。
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, doc_id: str, operator: str, action: str = "sign", **extra: Any) -> dict:
        import hashlib
        import time

        record = {
            "record_id": hashlib.sha1(f"{doc_id}{operator}{time.time()}".encode()).hexdigest()[:16],
            "doc_id": doc_id,
            "stage": 6,
            "action": action,
            "operator": operator,
            "timestamp": time.time(),
            "result": "success",
            **extra,
        }
        self._records.append(record)
        return record

    def trail(self, doc_id: str) -> list[dict]:
        """返回某文档的完整签核审计链。"""
        return [r for r in self._records if r["doc_id"] == doc_id]
