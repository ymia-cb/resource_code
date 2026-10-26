"""Stage 4c：公式重算，回填缓存值。

职责
----
对含公式的 OOXML（尤其 xlsx）做重算，回填公式缓存值。
这是解决「openpyxl 写公式不写缓存值 → 客户打开看到 0」的关键层。

接口契约
--------
符合 interfaces.Recalculator Protocol：
    recalc(xlsx_path: Path) -> Path   # 返回缓存值已回填的文件

关键决策
--------
- 公式在 IR 中存「表达式字符串」，cache_value 恒 null（见 generator），
  由本层统一回填——生成层绝不手写缓存值。
- 重算靠 LibreOffice headless 打开+保存触发；重算后逐公式比对缓存值，不一致报 REN-002。
- 本操作依赖 LibreOffice/字体，属「环境敏感」，生产环境进沙箱（见 sandbox 模块）。

桩 → 真实现
------------
- StubRecalculator：直接返回输入文件（无公式重算），供 MVP 空跑。
- LibreOfficeRecalculator：真实现，调 LibreOffice headless 重算。
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class StubRecalculator:
    """【桩】直接返回输入文件，不做重算。真实现见下方 LibreOfficeRecalculator。"""

    def recalc(self, xlsx_path: Path) -> Path:
        return xlsx_path


class LibreOfficeRecalculator:
    """【真实现·半成品】用 LibreOffice headless 重算并回填缓存值。

    审核注意：依赖系统安装 LibreOffice（soffice 在 PATH）。当前只做
    「调用 soffice 重新保存」的骨架，逐公式比对缓存值尚未实现（需解析 xlsx 的
    <v> 缓存节点，属 P2 填充）。
    """

    def __init__(self, soffice_bin: str = "soffice") -> None:
        self._soffice = soffice_bin

    def recalc(self, xlsx_path: Path) -> Path:
        out_dir = xlsx_path.parent / "recalc"
        out_dir.mkdir(parents=True, exist_ok=True)
        # LibreOffice headless 打开并重新保存，触发公式重算写缓存值
        subprocess.run(
            [self._soffice, "--headless", "--calc", "--convert-to", "xlsx",
             "--outdir", str(out_dir), str(xlsx_path)],
            check=True,
        )
        result = out_dir / xlsx_path.name
        # 审核注意：此处应逐公式比对 <v> 缓存值，不一致抛 REN-002
        return result
