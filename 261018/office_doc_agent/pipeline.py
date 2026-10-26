"""编排器：把各阶段串起来，MVP 跑通「生成 → 渲染 → 校验」最短闭环。

依赖注入：各阶段通过 interfaces.Protocol 注入，实现可替换（桩/真）。

全链路顺序（安全模型，详见《输入源治理与安全限制设计》）：
    生成（只输出引用与结构，不输出值）
  → 中间表示校验（claim 级：ref 存在/授权/新鲜；裸值拦截）★主拦截
  → 渲染（回填值、格式化）
  → 渲染后门禁（四层校验）
  → provenance 回查（终局断言）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interfaces import Generator, Renderer, Validator
from .security.claim import Channel, Claim, EventType, Layer
from .security.channel import ClaimQueue
from .security.whitelist import check_authorized


@dataclass
class PipelineResult:
    """一次编排的运行结果。"""

    docx_path: Path | None
    report: dict | None
    success: bool
    failed_stage: str | None = None
    # 安全校验产物：claim 队列（审核员可据此定位「哪条规则、哪个字段」出问题）
    claims: list[Claim] = field(default_factory=list)


class Pipeline:
    """编排生成 → 中间表示校验（安全） → 渲染 → 四层校验。"""

    def __init__(
        self,
        generator: Generator,
        renderer: Renderer,
        validator: Validator,
        *,
        # 安全校验的授权值集合：None 表示跳过安全校验（桩阶段）
        authorized_values: set | None = None,
        # 数值型字段名集合：只对这些字段的绑定值做白名单校验。
        # 标识符字段（如 policy_no）含数字但非事实数值，不进白名单。
        # 这个集合来自数据契约（type: decimal/money/number 的字段）。
        numeric_fields: set[str] | None = None,
    ) -> None:
        self._generator = generator
        self._renderer = renderer
        self._validator = validator
        self._authorized_values = authorized_values
        self._numeric_fields = numeric_fields

    def run(
        self,
        spec: Any,
        template: Any,
        data: Any,
        out_dir: Path,
        doc_id: str = "doc-001",
    ) -> PipelineResult:
        # Stage 4a：生成 IR
        ir = self._generator.generate(spec, template, data)

        # ★主拦截：中间表示校验（claim 级 + 白名单）
        claims = self._check_ir(ir, doc_id)
        blocking = [c for c in claims if c.event_type in (
            EventType.VALUE_NOT_AUTHORIZED, EventType.REF_FORBIDDEN, EventType.REF_UNRESOLVED
        )]
        if blocking:
            # 存在未授权值 → 阻断交付（安全模型：值不在集合就是错）
            return PipelineResult(
                docx_path=None,
                report=None,
                success=False,
                failed_stage="security",
                claims=claims,
            )

        # Stage 4b：渲染 OOXML
        docx_path = self._renderer.render(ir, out_dir / f"{doc_id}.docx")

        # Stage 5：四层校验
        artifacts = {
            "docx": docx_path,
            "doc_id": doc_id,
            "report_id": f"report-{doc_id}",
            "data": data,
        }
        report = self._validator.validate(artifacts, spec)

        success = report.get("overall") == "pass"
        failed_stage = None if success else report.get("failed_at_layer")
        return PipelineResult(
            docx_path=docx_path,
            report=report,
            success=success,
            failed_stage=failed_stage,
            claims=claims,
        )

    def _check_ir(self, ir: dict, doc_id: str) -> list[Claim]:
        """中间表示校验：扫描 IR 里的裸数字，做白名单匹配（C 层兜底）。

        审核注意：这是「数字只能来自结构化数据」的机器强制点。
        - IR 里的事实位本应只有 {{ref}} 引用（B 层），出现裸数字 = literal_in_fact_slot。
        - 裸数字归一后不在授权值集合 = value_not_authorized（自动阻断）。
        """
        if self._authorized_values is None:
            return []  # 桩阶段：未配置授权值，跳过安全校验

        claims: list[Claim] = []
        for idx, block in enumerate(ir.get("blocks", [])):
            # 事实值在 bindings 里（text 是占位符 {{ref}}），所以扫描 bindings 的值
            # —— 这正是「B 层只读」的体现：数字只能通过绑定注入，不能裸写在文本里。
            # 只校验「数值型字段」：标识符字段（如 policy_no）含数字但非事实数值。
            for field, value in self._block_bindings(block).items():
                if self._numeric_fields is not None and field not in self._numeric_fields:
                    continue  # 非数值型字段，跳过白名单校验
                unauthorized = check_authorized(str(value), self._authorized_values)
                for token in unauthorized:
                    claims.append(Claim(
                        event_type=EventType.VALUE_NOT_AUTHORIZED,
                        layer=Layer.C_FALLBACK,
                        span=(idx, idx),
                        target_field=field,
                        rule="value_whitelist",
                        detail=f"块 {idx} 字段 {field} 出现未授权数字 {token}",
                        actual=token,
                        doc_id=doc_id,
                    ))
        return claims

    @staticmethod
    def _block_bindings(block: dict) -> dict:
        """提取 block 里所有 bindings 的字段值（事实值的承载处）。"""
        bindings: dict = {}
        btype = block.get("type")
        if btype == "paragraph":
            bindings.update(block.get("bindings", {}))
        elif btype == "table":
            for row in block.get("rows", []):
                for cell in row.get("cells", []):
                    bindings.update(cell.get("bindings", {}))
        elif btype == "formula_cell":
            bindings.update(block.get("bindings", {}))
        return bindings
