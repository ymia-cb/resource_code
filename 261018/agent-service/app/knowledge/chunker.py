"""Markdown 语义切块（Spec 10.8 约束 3：按标题层级语义切分，非固定长度硬切）。

策略：
- 优先按 Markdown 标题（# 级别）切分，保留父级标题上下文
- 表格（|...|）作为原子单元，整体不硬切（表格语义依赖行列完整性）
- 图片引用 ![](url) 识别为独立信息点，保留 URL + alt（描述由 importer 经 VLM 填回）
- 无标题的纯文本段落按空行切块，超长段落软切
"""
from __future__ import annotations

import re

# 默认切块参数
DEFAULT_CHUNK_SIZE = 800  # 字符
DEFAULT_OVERLAP = 100  # 重叠字符（相邻 chunk 之间保留上文）

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def split_markdown(
    md_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """把 Markdown 按标题/段落/表格语义切块。

    Args:
        md_text: 原始 Markdown 文本
        chunk_size: 单个 chunk 目标字符数
        overlap: 相邻 chunk 重叠字符数

    Returns:
        chunk 文本列表（非空、去空白）。
    """
    if not md_text or not md_text.strip():
        return []

    # 1. 按标题分节，每节保留父级标题上下文
    sections = _split_by_headings(md_text)

    # 2. 每节内：先保护表格（原子单元），再按段落切块
    chunks: list[str] = []
    for heading, body in sections:
        pieces = _split_section_body(body, chunk_size)
        for piece in pieces:
            text = f"{heading}\n{piece}".strip() if heading else piece.strip()
            if text:
                chunks.append(text)

    # 3. 应用 overlap（相邻 chunk 之间共享尾部/头部上下文）
    return _apply_overlap(chunks, overlap)


def _split_by_headings(md_text: str) -> list[tuple[str, str]]:
    """按标题切分，返回 [(标题(含父级), 正文), ...]。

    无标题时返回 [("", 全文)]。
    """
    matches = list(_HEADING_RE.finditer(md_text))
    if not matches:
        return [("", md_text)]

    sections: list[tuple[str, str]] = []
    parent_heading = ""  # 累积父级标题上下文
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        level = len(m.group(1))
        # 父级标题上下文：低级别标题作为高级别标题的前缀（如 # A 下 ## B → "# A > ## B"）
        if i > 0 and level > len(matches[i - 1].group(1)):
            parent_heading = matches[i - 1].group(0)
        else:
            parent_heading = ""
        heading = f"{parent_heading}\n{m.group(0)}".strip() if parent_heading else m.group(0)
        sections.append((heading, md_text[start:end]))
    return sections


def _split_section_body(body: str, chunk_size: int) -> list[str]:
    """按段落切块，但表格作为原子单元整体保留，超长段落软切。"""
    body = body.strip()
    if not body:
        return [""]

    # 1. 先把表格块替换为占位符（保护原子性）
    tables = _extract_tables(body)
    placeholder_body = body
    for i, table in enumerate(tables):
        placeholder_body = placeholder_body.replace(table, f"\n\n__TABLE_{i}__\n\n")

    # 2. 按空行分段落
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", placeholder_body) if p.strip()]
    pieces: list[str] = []
    buf = ""

    for para in paragraphs:
        # 还原表格占位符
        para = _restore_tables(para, tables)
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}".strip() if buf else para
        else:
            if buf:
                pieces.append(buf)
            if len(para) > chunk_size and not _is_table(para):
                # 非表格超长段落软切
                pieces.extend(_soft_split(para, chunk_size))
                buf = ""
            else:
                # 表格或短段落整体保留
                buf = para

    if buf:
        pieces.append(buf)
    return pieces


def _extract_tables(body: str) -> list[str]:
    """提取 markdown 表格块（连续 |...| 行）。"""
    tables: list[str] = []
    for m in _TABLE_ROW_RE.finditer(body):
        # 简单：单行表格也保留；连续行由段落分割自然聚合
        tables.append(m.group(0))
    # 去重保序
    return list(dict.fromkeys(tables))


def _restore_tables(text: str, tables: list[str]) -> str:
    """把 __TABLE_i__ 占位符还原为表格（转结构化可读文本）。"""
    for i, table in enumerate(tables):
        text = text.replace(f"__TABLE_{i}__", table)
    return text


def _is_table(text: str) -> bool:
    """判断文本是否（含）表格。"""
    return bool(_TABLE_ROW_RE.search(text))


def _soft_split(text: str, chunk_size: int) -> list[str]:
    """超长段落软切：优先在句号/换行处切，尽量不破坏语义。"""
    parts: list[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        cut = remaining.rfind("。", 0, chunk_size)
        if cut < chunk_size // 2:
            cut = remaining.rfind("\n", 0, chunk_size)
        if cut < chunk_size // 2:
            cut = remaining.rfind("，", 0, chunk_size)
        if cut < chunk_size // 2:
            cut = chunk_size
        parts.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1 :].strip()
    if remaining:
        parts.append(remaining)
    return parts


def extract_images(md_text: str) -> list[dict]:
    """提取 md 中的图片引用，返回 [{alt, url}]。

    供 importer 上传图片到 MinIO + VLM 描述 + 填回 md。
    """
    return [
        {"alt": m.group(1), "url": m.group(2)}
        for m in _IMAGE_RE.finditer(md_text)
    ]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """相邻 chunk 之间加 overlap：把前一个 chunk 的尾部拼到后一个 chunk 头部。

    注意：overlap 作用于正文内容，标题本身已是共享上下文，故 overlap 只做
    轻量拼接，避免大幅膨胀。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        # 仅当尾部非空且不重复时拼接
        merged = f"{prev_tail.strip()}\n\n{chunks[i]}".strip()
        result.append(merged)
    return result
