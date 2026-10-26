"""知识入库：把 chunk 向量化后写入 Milvus。

当前阶段灌内置 FAQ；后续接 MinerU 文档导入（chunker → embedding → index_to_milvus）。
"""
from __future__ import annotations

from app.knowledge import milvus_client
from app.knowledge.embedding import embed
from app.knowledge.retriever import get_faq


def index_faq() -> int:
    """把内置 FAQ 向量化灌入 Milvus。返回插入条数。"""
    faq = get_faq()
    vectors = [embed(d["text"]) for d in faq]
    return milvus_client.insert(faq, vectors)


def index_chunks(chunks: list[dict]) -> int:
    """把任意 chunk 列表向量化灌入 Milvus。

    Args:
        chunks: [{chunk_id, text, category, source}, ...]

    Returns:
        插入条数。
    """
    if not chunks:
        return 0
    vectors = [embed(c["text"]) for c in chunks]
    return milvus_client.insert(chunks, vectors)


def ensure_indexed() -> bool:
    """确保 Milvus 已灌库（空则灌内置 FAQ）。返回是否已灌库。"""
    if milvus_client.count() > 0:
        return True
    try:
        index_faq()
        return True
    except Exception:
        return False
