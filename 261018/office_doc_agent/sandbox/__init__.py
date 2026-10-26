"""沙箱执行层：环境敏感操作的隔离执行。

职责
----
承载 renderer / recalc / exporter 等依赖 LibreOffice、字体、第三方库的「环境敏感」操作，
提供隔离 + 可复现（固定镜像 + 版本哈希）。generator / data_connector 等纯逻辑留在宿主机。

接口契约
--------
符合 interfaces.SandboxExecutor Protocol：
    run(image: str, code: str, files: dict[str, bytes]) -> dict

关键决策
--------
- 选型 agent-sandbox（E2B 协议兼容），Python 侧走 e2b-code-interpreter SDK。
- 固定镜像（office-render:1.0），内置 Python + python-docx/openpyxl + LibreOffice + 中文字体，
  消除「字体缺失 → PDF 方框」和「环境不一致 → 不可复现」。
- 沙箱生命周期：单文档一次 with 块，用完即销毁，杜绝跨文档状态污染。

桩 → 真实现
------------
- HostSandboxExecutor：宿主机直跑，无隔离，仅 MVP 阶段用。
- E2BSandboxExecutor：真实现，走 E2B Sandbox.create/run_code/files。
"""
from __future__ import annotations

from typing import Any


class HostSandboxExecutor:
    """【桩】宿主机直跑，无隔离。仅 MVP 阶段使用，生产必须换 E2BSandboxExecutor。"""

    def run(self, image: str, code: str, files: dict[str, bytes]) -> dict:
        # 审核注意：这里不真正执行 code，只返回空结果。
        # 真实现应调用 E2B Sandbox.run_code 在隔离环境执行。
        return {"image": image, "status": "host_stub", "files": list(files.keys())}


class E2BSandboxExecutor:
    """【真实现·半成品】通过 agent-sandbox / E2B 隔离执行。

    审核注意：依赖 e2b-code-interpreter 包 + E2B_API_KEY 环境变量。
    当前为骨架，未实际 import e2b（避免硬依赖），接入时取消注释即可。
    """

    def __init__(self, template: str = "office-render", timeout: int = 600) -> None:
        self._template = template
        self._timeout = timeout

    def run(self, image: str, code: str, files: dict[str, bytes]) -> dict:
        # 真实现（接入 e2b-code-interpreter 后）：
        #   from e2b_code_interpreter import Sandbox
        #   with Sandbox.create(template=self._template, timeout=self._timeout) as sbx:
        #       for name, data in files.items():
        #           sbx.files.write(name, data)
        #       result = sbx.run_code(code)
        #       return {"status": "ok", "result": result.logs.stdout}
        return {"image": image, "status": "e2b_stub", "files": list(files.keys())}
