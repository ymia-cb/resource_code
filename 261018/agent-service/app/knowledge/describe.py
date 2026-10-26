"""图片描述（VLM）+ 文档命名（大模型）。

- ``describe_image``：用 VLM 把图片转成文字描述，供 RAG 检索图片语义。
  无 VLM key 时降级返回 alt 文本占位（保证离线可跑）。
- ``generate_filename``：用大模型为文档生成规范文件名（防用户随意命名）。
  无 key 时降级用内容摘要 + 时间戳。
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import router


def describe_image(image_url: str, alt_text: str = "") -> str:
    """用 VLM 把图片转成一段文字描述。

    Args:
        image_url: 图片 URL（MinIO 或 file://）
        alt_text: 图片的 alt 文本（md 里 ![](url) 的方括号内容），作兜底

    Returns:
        文字描述（VLM 生成；无 key 降级返回 alt 占位）。
    """
    prompt = (
        "请用一句话中文描述这张图片的内容，聚焦关键信息（商品/表格/流程/数据等），"
        "不要输出其他内容。"
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    message = HumanMessage(content=content)
    raw = router.llm("vlm", [message])

    desc = str(raw).strip()
    # VLM 降级 mock 时返回的是 prompt 文本或空，此时用 alt 兜底
    if not desc or desc == prompt or desc.startswith("你是"):
        return alt_text or "（图片）"
    return desc


def generate_filename(md_content: str, original_name: str = "") -> str:
    """用大模型为文档生成规范文件名（防用户随意命名）。

    返回不含非法字符、长度受控的文件名（不含扩展名）。
    无 key 降级：取首行标题/摘要 + 时间戳。
    """
    # 提取候选标题（首个 markdown 标题或首行）
    title = _extract_title(md_content) or "文档"

    provider = router._build_provider("summarize") if hasattr(router, "_build_provider") else None
    if provider is not None:
        try:
            prompt = (
                "根据以下文档内容，生成一个简洁、规范的中文文件名（不含扩展名，"
                "不超过 20 个字符，只输出文件名本身，不要标点、不要引号）：\n\n"
                f"{md_content[:2000]}"
            )
            name = router.llm("summarize", [SystemMessage(content="你是文档命名助手"), HumanMessage(content=prompt)])
            name = str(name).strip().strip('"').strip("'")
            if name and name != prompt:
                return _sanitize_filename(name)
        except Exception:
            pass  # 降级到确定性命名

    # 降级：标题 + 时间戳
    import time

    ts = time.strftime("%Y%m%d")
    return _sanitize_filename(f"{title}-{ts}")


def _extract_title(md_content: str) -> str | None:
    """提取首个 markdown 标题作为标题候选。"""
    m = re.search(r"^#\s+(.+)$", md_content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    first_line = md_content.strip().splitlines()[0] if md_content.strip() else ""
    return first_line[:30] if first_line else None


def _sanitize_filename(name: str) -> str:
    """清洗文件名：去掉非法字符，限制长度。"""
    name = re.sub(r'[\\/:*?"<>|\s]+', "-", name).strip("-")
    name = name[:40] or "文档"
    return name
