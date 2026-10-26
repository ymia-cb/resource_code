"""Stage 7：分发归档（PDF/A + 水印 + 权限 + 留存）。

职责
----
把 OOXML 导出为 PDF/A（可长期归档的 ISO 标准格式），加水印、设权限，
归档时记录文件指纹，保证「留痕、可追溯」，防「泄密或找不到原件」。

关键决策
--------
- 统一导出 PDF/A：这是长期归档标准，避免「文件打开不了/版式漂移」。
- 加水印 + 权限控制：防泄露；归档记指纹（内容哈希），原件可随时按指纹追溯。
- 依赖 LibreOffice + 字体，属环境敏感，进沙箱（见 sandbox 模块）。

桩 → 真实现
------------
- StubExporter：空实现。
- PdfExporter：真实现，LibreOffice 导出 PDF/A + 水印 + 指纹。
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class StubExporter:
    """【桩】空实现。真实现见下方 PdfExporter。"""

    def export(self, docx_path: str) -> dict:
        return {"docx_path": docx_path, "status": "stub"}


class PdfExporter:
    """【真实现·半成品】导出 PDF/A + 记录归档指纹。

    审核注意：水印与权限控制需额外工具（如 pikepdf/PyPDF2），当前只做
    「LibreOffice 导出 PDF + 指纹」骨架，水印/权限属 P3 填充。
    """

    def __init__(self, soffice_bin: str = "soffice") -> None:
        self._soffice = soffice_bin

    def export(self, docx_path: str, out_dir: str | None = None) -> dict:
        src = Path(docx_path)
        out = Path(out_dir) if out_dir else src.parent / "pdf"
        out.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [self._soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(out), str(src)],
            check=True,
        )
        pdf_path = out / (src.stem + ".pdf")

        # 归档指纹：内容哈希，出问题能追溯原件
        fingerprint = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        return {"pdf_path": str(pdf_path), "fingerprint": fingerprint, "status": "ok"}
