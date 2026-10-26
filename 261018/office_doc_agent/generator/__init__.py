"""Stage 4a：生成中间表示（IR）。

铁律：
- 纯函数，禁止 import 任何 OOXML 库（python-docx/openpyxl/python-pptx）。
- 同一输入必得字节级一致 JSON（可回放）。
- 公式存表达式字符串，cache_value 恒 null，由 recalc 层回填。
"""
from __future__ import annotations

import json
from typing import Any

from ..interfaces import Generator, IR


class DefaultGenerator:
    """把 spec + template + data 生成中间表示 JSON。"""

    def __init__(self, schema_path: str | None = None) -> None:
        self._schema_path = schema_path

    def generate(self, spec: Any, template: Any, data: Any) -> IR:
        # 从模板提取 block 结构，用 data 填充占位符绑定
        blocks: list[dict] = []
        for block in template.get("blocks", []):
            blocks.append(self._render_block(block, data))

        ir: IR = {
            "meta": {
                "doc_type": spec.get("doc_type", "general"),
                "spec_version": spec.get("version", "1.0"),
                "template_id": template.get("template_id", "unknown"),
                "generated_at": spec.get("generated_at", ""),
                "data_batch": spec.get("data_batch", ""),
            },
            "blocks": blocks,
        }
        return ir

    @staticmethod
    def _render_block(block: dict, data: Any) -> dict:
        btype = block.get("type")
        if btype == "paragraph":
            return {
                "type": "paragraph",
                "text": block.get("text", ""),
                "style": block.get("style", "body"),
                "bindings": dict(block.get("bindings", {})),
            }
        if btype == "table":
            rows = []
            for row in block.get("rows", []):
                cells = []
                for cell in row.get("cells", []):
                    cells.append(
                        {
                            "text": cell.get("text", ""),
                            "formula": cell.get("formula"),
                            "bindings": dict(cell.get("bindings", {})),
                        }
                    )
                rows.append({"cells": cells})
            return {"type": "table", "style": block.get("style", "grid"), "rows": rows}
        if btype == "formula_cell":
            return {
                "type": "formula_cell",
                "formula": block.get("formula", ""),
                "cache_value": None,  # 恒 null，由 recalc 回填
                "bindings": dict(block.get("bindings", {})),
            }
        # 未知 block 类型：原样保留（渲染器/校验器会拒绝）
        return dict(block)

    # ---- 可回放性：序列化保证稳定 ----
    @staticmethod
    def dumps(ir: IR) -> bytes:
        """稳定序列化：sort_keys 保证字节级一致，供回放测试用。"""
        return json.dumps(ir, ensure_ascii=False, sort_keys=True).encode("utf-8")
