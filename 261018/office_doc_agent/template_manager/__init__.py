"""Stage 2：受控模板 + 占位符契约。

职责
----
加载受控模板（.dotx/.xltx/.potx），校验模板占位符与契约一致，返回模板实例。

接口契约
--------
符合 interfaces.TemplateManager Protocol：
    load(template_id: str) -> TemplateInstance

关键决策
--------
- 模板存「受控库」，业务只读、无编辑权限（模板失控是批量失败的根因）。
- 每个模板带版本号 + 内容哈希（SHA-256），改动走变更流程。
- 占位符用契约文件声明（名称/类型/必填/默认/格式），渲染前校验模板占位符与契约一致。

桩 → 真实现
------------
- StubTemplateManager：返回写死模板实例（含 paragraph/table block），供 MVP 空跑。
- TemplateManager：读 .dotx → 提取占位符 → 对照契约校验 → 返回 TemplateInstance。
  校验失败抛 TPL-001（占位符与契约不一致）/ TPL-002（版本号/哈希缺失）。
"""
from __future__ import annotations

from typing import Any


class StubTemplateManager:
    """【桩】返回写死模板实例，供 MVP 空跑。真实现见下方 TemplateManager。"""

    def load(self, template_id: str) -> Any:
        return {
            "template_id": template_id,
            "doc_type": "contract",
            "version": "1.0",
            "content_hash": "stub-hash",  # 真实现应写入真实 SHA-256
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "保单号：{policy_no}",
                    "style": "body",
                    "bindings": {"policy_no": "P202610230001"},
                },
                {
                    "type": "table",
                    "style": "Table Grid",
                    "rows": [
                        {
                            "cells": [
                                {"text": "项目", "bindings": {}},
                                {"text": "金额", "bindings": {}},
                            ]
                        },
                        {
                            "cells": [
                                {"text": "保费", "bindings": {}},
                                {"text": "{premium}", "bindings": {"premium": "¥1,234.56"}},
                            ]
                        },
                    ],
                },
            ],
        }


class TemplateManager:
    """【真实现·半成品】加载模板并校验占位符契约。

    审核注意：当前只做「占位符契约一致性」的骨架校验，真实的 .dotx 解析
    （读 document.xml 提取占位符）尚未接入——那是 python-docx 之外的 XML 层，
    属 P2 填充项。
    """

    def load(self, template_id: str, contract: Any = None) -> Any:
        # 桩数据：真实现应读 template_id 对应的 .dotx
        template = StubTemplateManager().load(template_id)

        if contract is not None:
            declared = {p["name"] for p in contract.get("placeholders", [])}
            used = self._extract_placeholders(template)
            if used != declared:
                # 错误码 TPL-001：模板占位符与契约不一致
                raise ValueError(f"TPL-001: 占位符不一致，模板用 {used}，契约声明 {declared}")

        return template

    @staticmethod
    def _extract_placeholders(template: Any) -> set[str]:
        """从模板 block 提取实际使用的占位符名。"""
        used: set[str] = set()
        for block in template.get("blocks", []):
            if block.get("type") == "paragraph":
                used.update(block.get("bindings", {}).keys())
            elif block.get("type") == "table":
                for row in block.get("rows", []):
                    for cell in row.get("cells", []):
                        used.update(cell.get("bindings", {}).keys())
        return used
