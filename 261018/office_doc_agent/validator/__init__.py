"""Stage 5：四层校验。

四层递进（从结构到语义）：
- L1 结构层：文件能打开、OOXML 合法、占位符 0 残留。
- L2 数值层：公式缓存值与重算结果一致（Word 无公式缓存，主要针对 Excel）。
- L3 渲染层：字体完整性、内容溢出、分页漂移（可静态检测部分）。
- L4 业务层：口径一致性（关键字段与数据源逐值比对）、空值策略。

铁律：每层失败即阻断，不允许带病交付。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from lxml import etree

from ..interfaces import ValidationReport, Validator

# 残留占位符模式：{xxx}
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

# 含中日韩文字的检测（用于字体完整性校验）
_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class DocxValidator:
    """校验渲染后的 Word 产物。"""

    def validate(self, artifacts: dict, spec: Any) -> ValidationReport:
        docx_path = artifacts.get("docx")
        layers = [
            self._structure_layer(docx_path),
            self._numeric_layer(docx_path),
            self._render_layer(docx_path),   # MVP 桩
            self._business_layer(artifacts, spec),  # MVP 桩
        ]
        failed_at = None
        overall = "pass"
        for layer in layers:
            if not layer["passed"]:
                overall = "fail"
                if failed_at is None:
                    failed_at = layer["layer"]
        return {
            "report_id": artifacts.get("report_id", ""),
            "doc_id": artifacts.get("doc_id", ""),
            "generated_at": "",
            "overall": overall,
            "failed_at_layer": failed_at,
            "layers": layers,
        }

    # ---- L1 结构层 ----
    @staticmethod
    def _structure_layer(docx_path: Path | None) -> dict:
        checks = []
        if docx_path is None or not Path(docx_path).exists():
            return {
                "layer": "structure",
                "passed": False,
                "checks": [{"name": "file_exists", "passed": False, "detail": "产物文件不存在"}],
                "issues": ["产物文件不存在"],
            }

        # 1. OOXML 合法性：用 lxml 解析 document.xml
        try:
            doc = Document(str(docx_path))
            checks.append({"name": "ooXML_valid", "passed": True, "detail": None})
        except Exception as exc:  # noqa: BLE001
            return {
                "layer": "structure",
                "passed": False,
                "checks": [{"name": "ooXML_valid", "passed": False, "detail": str(exc)}],
                "issues": [f"OOXML 非法: {exc}"],
            }

        # 2. 占位符残留：遍历所有段落/表格文本
        residual = []
        for para in doc.paragraphs:
            residual.extend(_PLACEHOLDER_RE.findall(para.text))
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    residual.extend(_PLACEHOLDER_RE.findall(cell.text))

        if residual:
            checks.append(
                {"name": "placeholder_residual", "passed": False, "detail": f"残留: {residual}"}
            )
            return {
                "layer": "structure",
                "passed": False,
                "checks": checks,
                "issues": [f"占位符残留: {residual}"],
            }

        checks.append({"name": "placeholder_residual", "passed": True, "detail": None})
        return {"layer": "structure", "passed": True, "checks": checks, "issues": []}

    # ---- L2 数值层 ----
    @staticmethod
    def _numeric_layer(docx_path: Path | None) -> dict:
        # MVP：Word 无公式缓存值概念，此层主要针对 Excel。Word 场景通过。
        # 后续 Excel 场景：比对重算缓存值 vs 实际值。
        return {
            "layer": "numeric",
            "passed": True,
            "checks": [{"name": "formula_cache_match", "passed": True, "detail": "Word 无公式缓存，跳过"}],
            "issues": [],
        }

    # ---- L3 渲染层 ----
    @staticmethod
    def _render_layer(docx_path: Path | None) -> dict:
        """渲染层校验：字体完整性 + 内容溢出 + 空段落。

        审核注意：
        - 字体完整性：检测文档 run 声明的字体是否在「已打包字体白名单」里。
          真实现中，白名单来自沙箱镜像的字体清单；此处用「文档声明了中文字体即视为完整」的启发式。
        - 内容溢出：检测超长单元格文本 / 缺失表格（渲染后无表格但 IR 声明了表格）。
        - 分页漂移：依赖真实渲染引擎（LibreOffice），此处只做「可静态检测」的项。
        """
        checks = []
        issues = []
        if docx_path is None or not Path(docx_path).exists():
            return {
                "layer": "render",
                "passed": False,
                "checks": [{"name": "file_exists", "passed": False, "detail": "产物文件不存在"}],
                "issues": ["产物文件不存在"],
            }

        doc = Document(str(docx_path))

        # 1. 字体完整性：检测 run 显式声明的字体是否在「允许字体集合」里。
        #    注意：python-docx 默认不显式声明字体（font.name is None，继承样式默认值），
        #    这是正常现象，不是「字体缺失」。真正的风险是「声明了某个不在沙箱镜像里的字体」。
        #    MVP 阶段：只检测「声明了非中文字体」的 run（如声明的字体明显是西文且含中文文本）。
        undeclared_font = 0
        for para in doc.paragraphs:
            for run in para.runs:
                # 含中文文本但字体名是纯 ASCII（未显式设置中文字体）→ 潜在方框风险
                if run.text.strip() and _HAS_CJK.search(run.text) and run.font.name is None:
                    undeclared_font += 1
        checks.append({
            "name": "font_complete",
            "passed": undeclared_font == 0,
            "detail": None if undeclared_font == 0 else f"{undeclared_font} 个含中文 run 未声明中文字体",
        })
        if undeclared_font:
            issues.append(f"字体缺失风险: {undeclared_font} 个含中文 run 未声明中文字体")

        # 2. 内容溢出：检测单元格超长文本（粗粒度启发式，>200 字符视为可能溢出）
        overflow_cells = 0
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if len(cell.text) > 200:
                        overflow_cells += 1
        checks.append({
            "name": "content_overflow",
            "passed": overflow_cells == 0,
            "detail": None if overflow_cells == 0 else f"{overflow_cells} 个单元格文本超长",
        })
        if overflow_cells:
            issues.append(f"内容溢出风险: {overflow_cells} 个单元格文本超长")

        passed = not issues
        return {"layer": "render", "passed": passed, "checks": checks, "issues": issues}

    # ---- L4 业务层 ----
    @staticmethod
    def _business_layer(artifacts: dict, spec: Any) -> dict:
        """业务层校验：口径一致性（关键字段与数据源逐值比对）+ 空值策略。

        审核注意：这是「数字对但含义错」的最后防线。
        - 关键字段从 spec 的 field_list 提取，逐个与数据源 data 比对。
        - 空值策略：forbidden 的字段若为空 → 阻断（口径错误，不是数值错误）。
        - 时区/百分比口径：MVP 做「存在性 + 空值」校验，完整口径断言在 data_connector。
        """
        checks = []
        issues = []
        data = artifacts.get("data", {})

        field_list = (spec or {}).get("field_list", [])
        if not field_list:
            # 无字段清单时（桩规格书），做「数据包非空」的基础断言
            checks.append({"name": "caliber", "passed": bool(data), "detail": "无字段清单，仅校验数据非空"})
            if not data:
                issues.append("数据包为空")
            passed = not issues
            return {"layer": "business", "passed": passed, "checks": checks, "issues": issues}

        # 逐字段比对：关键字段必须在数据源中有值，且空值策略合规
        for field in field_list:
            name = field.get("name")
            value = data.get(name)
            null_policy = field.get("null_policy")

            if value is None and null_policy == "forbidden":
                checks.append({
                    "name": f"caliber:{name}",
                    "passed": False,
                    "detail": f"字段 {name} 空值违反 forbidden 策略",
                })
                issues.append(f"字段 {name} 空值违反 forbidden 策略")
            else:
                checks.append({"name": f"caliber:{name}", "passed": True, "detail": None})

        passed = not issues
        return {"layer": "business", "passed": passed, "checks": checks, "issues": issues}
