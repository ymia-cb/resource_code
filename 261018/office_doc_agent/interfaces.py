"""核心抽象：所有模块通过 Protocol 定义接口，实现可替换。

这是「先骨架」的关键 —— 先定义接口，桩实现也能跑通全链路。
纯逻辑与副作用分离：Generator 是纯函数，Renderer/Recalculator 是副作用（进沙箱）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---- 领域实体（用 Any 占位，正式实现时换 pydantic 模型）----
DocumentSpec = Any
TemplateInstance = Any
DataContract = Any
Dataset = Any
IR = dict
ValidationReport = dict


@runtime_checkable
class SpecLoader(Protocol):
    """Stage 1：加载并校验文档规格书。"""

    def load(self, spec_path: Path) -> DocumentSpec:
        ...


@runtime_checkable
class TemplateManager(Protocol):
    """Stage 2：加载受控模板并校验占位符契约。"""

    def load(self, template_id: str) -> TemplateInstance:
        ...


@runtime_checkable
class DataConnector(Protocol):
    """Stage 3：取数，接入层做口径断言（精度/时区/空值）。"""

    def fetch(self, contract: DataContract) -> Dataset:
        ...


@runtime_checkable
class Generator(Protocol):
    """Stage 4a：生成中间表示。纯函数，禁止 import OOXML 库，可回放。"""

    def generate(self, spec: DocumentSpec, template: TemplateInstance, data: Dataset) -> IR:
        ...


@runtime_checkable
class Renderer(Protocol):
    """Stage 4b：IR → OOXML。副作用，进沙箱。禁止手拼 OOXML 字符串。"""

    def render(self, ir: IR, out_path: Path) -> Path:
        ...


@runtime_checkable
class Recalculator(Protocol):
    """Stage 4c：公式重算，回填缓存值。副作用，进沙箱。"""

    def recalc(self, xlsx_path: Path) -> Path:
        ...


@runtime_checkable
class Validator(Protocol):
    """Stage 5：四层校验，产出校验报告。"""

    def validate(self, artifacts: dict, spec: DocumentSpec) -> ValidationReport:
        ...


@runtime_checkable
class SandboxExecutor(Protocol):
    """沙箱执行层：环境敏感操作必须经此层。"""

    def run(self, image: str, code: str, files: dict[str, bytes]) -> dict:
        ...
