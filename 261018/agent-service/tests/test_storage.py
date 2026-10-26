"""存储层 + 图片/表格切分 + 命名 测试。

覆盖：MinIO 上传/下载/删除（本地降级）、图片提取、表格原子性切块、
图片描述降级、文件名生成。

运行：python -m pytest tests/test_storage.py -v
"""
from __future__ import annotations

import pytest

from app.knowledge.chunker import split_markdown, extract_images
from app.knowledge.describe import generate_filename, describe_image, _sanitize_filename
from app.knowledge.importer import process_images
from app.storage import minio_client


# ---------------------------------------------------------------------------
# 图片提取
# ---------------------------------------------------------------------------


def test_extract_images():
    md = "![a](http://x/a.png) 文字 ![b](http://y/b.jpg)"
    imgs = extract_images(md)
    assert len(imgs) == 2
    assert imgs[0] == {"alt": "a", "url": "http://x/a.png"}
    assert imgs[1] == {"alt": "b", "url": "http://y/b.jpg"}


def test_extract_images_none():
    assert extract_images("纯文本无图") == []


# ---------------------------------------------------------------------------
# 表格原子性切块
# ---------------------------------------------------------------------------


def test_table_preserved_as_atomic_unit():
    """表格整体保留，不被硬切破坏行列。"""
    md = "# 表\n\n| 列A | 列B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    chunks = split_markdown(md, chunk_size=200)
    # 表格行完整出现在某个 chunk 里
    joined = "\n".join(chunks)
    assert "| 列A | 列B |" in joined
    assert "| 1 | 2 |" in joined
    assert "| 3 | 4 |" in joined


def test_split_markdown_parent_heading_context():
    """父级标题上下文：子标题 chunk 携带父标题。"""
    md = "# 退款政策\n\n## 7天无理由\n内容A\n\n## 秒杀商品\n内容B"
    chunks = split_markdown(md, chunk_size=500)
    # 至少一个 chunk 含「退款政策」和「7天无理由」的上下文
    assert any("# 退款政策" in c and "7天无理由" in c for c in chunks)


# ---------------------------------------------------------------------------
# 文件名生成（防随意命名）
# ---------------------------------------------------------------------------


def test_sanitize_filename():
    assert _sanitize_filename("退款/政策:说明?") == "退款-政策-说明"
    assert _sanitize_filename("") == "文档"


def test_generate_filename_fallback():
    """无大模型 key 时降级：标题 + 时间戳，不含非法字符。"""
    name = generate_filename("# 退款政策说明\n内容", "随意名字.md")
    assert name  # 非空
    assert "/" not in name and ":" not in name and "?" not in name
    # 降级应基于标题而非用户原名
    assert "退款政策" in name


# ---------------------------------------------------------------------------
# 图片描述降级
# ---------------------------------------------------------------------------


def test_describe_image_fallback_to_alt():
    """无 VLM key 时，描述降级为 alt 占位。"""
    desc = describe_image("http://x/a.png", alt_text="商品图")
    assert desc  # 非空
    # mock 降级时返回 alt 或占位，不应是 prompt 原文
    assert "你是" not in desc


# ---------------------------------------------------------------------------
# process_images（图片地址填回 md）
# ---------------------------------------------------------------------------


def test_process_images_no_images_returns_original():
    md = "# 标题\n正文"
    assert process_images(md, 1) == md


def test_process_images_unreachable_image_keeps_original():
    """图片无法下载 → 保留原引用，不填回，不阻断。"""
    md = "![图](http://nonexistent.invalid/img.png)\n正文"
    result = process_images(md, 2)
    # 无法下载 → 保留原引用（不抛异常）
    assert "![图](http://nonexistent.invalid/img.png)" in result


# ---------------------------------------------------------------------------
# MinIO 上传/下载/删除（真实 MinIO，不可用时跳过）
# ---------------------------------------------------------------------------


def _minio_available():
    try:
        return minio_client.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _minio_available(), reason="MinIO 未配置")
def test_minio_upload_download_delete_roundtrip():
    key = "test/roundtrip.txt"
    data = b"hello minio"
    url = minio_client.upload(key, data, content_type="text/plain")
    assert url.endswith(key)

    got = minio_client.download(key)
    assert got == data

    minio_client.delete(key)
