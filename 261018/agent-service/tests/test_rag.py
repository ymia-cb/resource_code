"""RAG / Milvus 向量检索测试。

覆盖：embedding 确定性、哈希向量归一化、索引灌库、count、检索、
category 标量过滤、无 embedding key 时的阈值策略、降级关键词检索。

运行：python -m pytest tests/test_rag.py -v
"""
from __future__ import annotations

import pytest

from app.knowledge.embedding import embed, embedding_dim, _hash_embed
from app.knowledge.retriever import retrieve, _keyword_retrieve, get_faq


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_embed_deterministic():
    """同一文本两次 embed 结果一致。"""
    assert embed("退款政策") == embed("退款政策")


def test_embed_dimension():
    """向量维度与 embedding_dim() 对齐。"""
    v = embed("测试")
    assert len(v) == embedding_dim()


def test_hash_embed_normalized():
    """哈希向量 L2 归一化（模长 ≈ 1）。"""
    v = _hash_embed("任意文本")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_embed_different_texts_differ():
    """不同文本向量不同。"""
    assert embed("退款") != embed("发货")


# ---------------------------------------------------------------------------
# 降级关键词检索（不依赖 Milvus，保证离线可跑）
# ---------------------------------------------------------------------------


def test_keyword_retrieve_refund():
    hits = _keyword_retrieve("我要退款", category=None, top_k=3)
    assert hits, "退款关键词应命中 FAQ"
    assert all("退" in h["text"] or "退款" in h["text"] for h in hits)


def test_keyword_retrieve_category_filter():
    hits = _keyword_retrieve("退款", category="coupon", top_k=3)
    assert all(h["category"] == "coupon" for h in hits)


def test_keyword_retrieve_empty_when_no_match():
    hits = _keyword_retrieve("今天天气怎么样", category=None, top_k=3)
    assert hits == []


def test_get_faq_shape():
    faq = get_faq()
    assert len(faq) == 4
    for d in faq:
        assert {"chunk_id", "text", "category", "source"} <= set(d)


# ---------------------------------------------------------------------------
# Milvus 检索（需要真实 Milvus 运行时；不可用时跳过）
# ---------------------------------------------------------------------------


def _milvus_available() -> bool:
    try:
        from app.knowledge import milvus_client

        return milvus_client.count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _milvus_available(), reason="Milvus 未运行或未灌库")
def test_milvus_count():
    from app.knowledge import milvus_client

    # 至少内置 FAQ 的 4 条已灌库（测试文档可能额外写入，故用 >= 而非 ==）
    assert milvus_client.count() >= 4


@pytest.mark.skipif(not _milvus_available(), reason="Milvus 未运行或未灌库")
def test_milvus_retrieve_returns_hits():
    hits = retrieve("退款政策是什么")
    assert hits, "Milvus 已灌库时检索应命中"


@pytest.mark.skipif(not _milvus_available(), reason="Milvus 未运行或未灌库")
def test_milvus_retrieve_category_filter():
    hits = retrieve("退货", category="coupon", top_k=5)
    assert all(h["category"] == "coupon" for h in hits)


@pytest.mark.skipif(not _milvus_available(), reason="Milvus 未运行或未灌库")
def test_milvus_retrieve_top_k():
    hits = retrieve("退款", top_k=2)
    assert len(hits) <= 2
