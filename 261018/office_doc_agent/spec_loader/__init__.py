"""Stage 1：文档规格书加载与校验。

职责
----
从 YAML 加载文档规格书，校验必填字段，返回类型安全的 DocumentSpec。

接口契约
--------
符合 interfaces.SpecLoader Protocol：
    load(spec_path: Path) -> DocumentSpec

关键决策
--------
- 规格书是「四方签核」的产物（业务/合规/IT/数据），字段级定义口径/精度/时区/空值。
- 本模块只负责「加载 + 结构校验」，不负责「四方签核流程」（那是人工/工作流系统的事）。

桩 → 真实现
------------
- StubSpecLoader：不读文件，返回写死规格，供 MVP 空跑。
- SpecLoader：读 YAML → 校验必填字段 → 返回 DocumentSpec。
  校验失败抛 SPEC-001（缺必填字段）/ SPEC-002（验收标准不明确）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class StubSpecLoader:
    """【桩】返回写死规格，供 MVP 空跑。真实现见下方 SpecLoader。"""

    def load(self, spec_path: Path | None) -> Any:
        # 写死一份最小合法规格，字段与数据模型设计对齐
        return {
            "doc_type": "contract",
            "version": "1.0",
            "generated_at": "2026-10-23T18:00:00+08:00",
            "data_batch": "batch-20261023-001",
        }


class SpecLoader:
    """【真实现·半成品】从 YAML 加载并校验规格书。

    审核注意：当前只做「必填字段存在性」校验，口径/精度/时区/空值的
    字段级校验在数据契约（DataContract）层完成，不在本模块重复。
    """

    REQUIRED_FIELDS = ("doc_type", "version", "field_list", "data_sources", "acceptance")

    def load(self, spec_path: Path) -> Any:
        import yaml  # 延迟导入：仅真实现路径需要

        raw = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))

        missing = [f for f in self.REQUIRED_FIELDS if f not in raw]
        if missing:
            # 错误码 SPEC-001：规格书缺失必填字段
            raise ValueError(f"SPEC-001: 规格书缺失必填字段 {missing}")

        if not raw.get("acceptance"):
            # 错误码 SPEC-002：验收标准不明确
            raise ValueError("SPEC-002: 验收标准不明确")

        return raw
