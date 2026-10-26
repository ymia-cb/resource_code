"""知识检索：优先 Milvus 向量检索，降级内存关键词检索。

- Milvus 可用且已灌库 → 向量检索（语义相似度 + category 标量过滤）
- Milvus 不可用/未灌库 → 降级内存关键词检索（保证离线可跑）

检索结果统一为 [{chunk_id, text, category, source}]，供 RAG 生成 + 溯源。
"""
from __future__ import annotations

from app.config import get_embedding_config
from app.knowledge import milvus_client
from app.knowledge.embedding import embed


# 内置 FAQ（作为 Milvus 未灌库时的降级数据源 + indexer 灌库的种子）
_FAQ: list[dict] = [
    {
        "chunk_id": "faq-refund-0",
        "text": "7 天无理由退货：签收后 7 天内、商品未使用、不影响二次销售，可申请退货退款。",
        "category": "refund_policy",
        "source": "faq/refund.md",
    },
    {
        "chunk_id": "faq-refund-1",
        "text": "秒杀商品、定制商品、生鲜食品不支持无理由退货。",
        "category": "refund_policy",
        "source": "faq/refund.md",
    },
    {
        "chunk_id": "faq-ship-0",
        "text": "订单付款后 48 小时内发货，偏远地区顺延 24 小时。",
        "category": "shipping",
        "source": "faq/shipping.md",
    },
    {
        "chunk_id": "faq-coupon-0",
        "text": "优惠券在订单取消后自动退回账户，7 天内有效。",
        "category": "coupon",
        "source": "faq/coupon.md",
    },
]


def _keyword_retrieve(query: str, category: str | None, top_k: int) -> list[dict]:
    """内存关键词检索（降级路径）。"""
    hits = [d for d in _FAQ if category is None or d["category"] == category]
    keywords = _keywords(query)
    scored: list[tuple[int, dict]] = []
    for d in hits:
        n = sum(1 for w in keywords if w in d["text"])
        if n > 0:
            scored.append((n, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def retrieve(query: str, category: str | None = None, top_k: int = 3) -> list[dict]:
    """检索：优先 Milvus 向量检索，降级内存关键词。

    无命中返回空列表，由上层触发转人工，禁止硬塞不相关内容（防幻觉）。

    阈值策略：
    - 真实 embedding 启用时，COSINE 相似度有意义，过滤掉 score < 0.5 的低质命中
    - 哈希向量兜底时，向量近似正交、相似度无意义，只做 category 过滤 + top_k，
      不做 score 阈值过滤（否则会误杀全部命中、退回关键词检索）
    """
    cfg = get_embedding_config()
    use_semantic_threshold = cfg.enabled

    # 优先 Milvus
    try:
        if milvus_client.count() > 0:
            query_vec = embed(query)
            hits = milvus_client.search(query_vec, top_k=top_k, category=category)
            if use_semantic_threshold:
                hits = [h for h in hits if h.get("score", 0) > 0.5]
            return hits
    except Exception:
        # Milvus 不可用 → 降级内存
        pass

    return _keyword_retrieve(query, category, top_k)


def _keywords(query: str) -> list[str]:
    """从 query 提取检索关键词（降级路径用）。"""
    domain_terms = [
        "退款", "退货", "退", "秒杀", "定制", "生鲜", "发货", "偏远",
        "优惠券", "运费", "售后", "换货", "无理由",
    ]
    return [t for t in domain_terms if t in query]


def get_faq() -> list[dict]:
    """返回内置 FAQ（供 indexer 灌库用）。"""
    return list(_FAQ)
