"""Embedding 向量化。

有 embedding API 配置时走 OpenAI 兼容 /embeddings 端点；
未配置时用确定性哈希向量兜底（同一文本 → 相同向量，保证离线可跑、检索可复现）。

哈希向量的局限：无法体现语义相似度，仅作为无 embedding key 时的占位，
生产环境必须配真实 embedding 模型（如 bge-small-zh / text-embedding-3）。
"""
from __future__ import annotations

import hashlib

import httpx

from app.config import get_embedding_config

# 哈希向量维度（与 Milvus collection 的 dim 对齐）
HASH_DIM = 384


def _hash_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """确定性哈希向量：用 SHA-256 展开成 dim 维浮点向量。

    同一文本永远得到相同向量；不同文本近似正交（哈希分散性）。
    仅作无 embedding key 的降级占位，不表达语义。
    """
    vec = [0.0] * dim
    # 对文本的每个字节位置做哈希，撒到 dim 个桶
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(dim):
        # 用 (i, digest[i % len(digest)]) 扰动，得到 [-1, 1] 的伪随机值
        h = hashlib.sha256((str(i) + text).encode("utf-8")).digest()
        vec[i] = (h[0] / 127.5) - 1.0
    # L2 归一化（便于余弦相似度）
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _api_embed(text: str, dim: int) -> list[float]:
    """调 OpenAI 兼容 /embeddings 端点。"""
    cfg = get_embedding_config()
    resp = httpx.post(
        f"{cfg.base_url.rstrip('/')}/embeddings",
        json={"model": cfg.model, "input": text},
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    vec = data["data"][0]["embedding"]
    # 若维度不匹配，截断或补零
    if len(vec) < dim:
        vec = vec + [0.0] * (dim - len(vec))
    elif len(vec) > dim:
        vec = vec[:dim]
    return vec


def embed(text: str) -> list[float]:
    """文本向量化。有 embedding key 走 API，否则哈希向量兜底。"""
    cfg = get_embedding_config()
    dim = cfg.dim if cfg.enabled else HASH_DIM
    if cfg.enabled:
        try:
            return _api_embed(text, dim)
        except Exception:
            # API 失败降级哈希
            return _hash_embed(text, dim)
    return _hash_embed(text, dim)


def embedding_dim() -> int:
    """返回当前 embedding 维度（供 Milvus collection 建表对齐）。"""
    cfg = get_embedding_config()
    return cfg.dim if cfg.enabled else HASH_DIM
