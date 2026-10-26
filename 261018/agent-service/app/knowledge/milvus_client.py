"""Milvus 客户端封装（MilvusClient API，pymilvus 3.x 推荐）。

职责：连接、collection 创建、向量插入、向量检索 + category 标量过滤。
"""
from __future__ import annotations

from app.config import get_milvus_config
from app.knowledge.embedding import embedding_dim


def _get_client():
    """懒加载 MilvusClient（进程内复用）。"""
    from pymilvus import MilvusClient

    cfg = get_milvus_config()
    return MilvusClient(uri=f"http://{cfg.host}:{cfg.port}")


_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "AUTOINDEX",
}


def ensure_collection() -> None:
    """确保 collection 存在、向量索引已建、且已加载到内存。

    - 建 collection（varchar 主键 + vector + 标量字段）
    - 建向量索引（COSINE，缺失索引会导致 load_collection 失败、search 返回空）
    - load_collection 到内存（未加载时 count 恒为 0，search 也无法执行）
    """
    from pymilvus import MilvusClient, DataType

    cfg = get_milvus_config()
    dim = embedding_dim()
    client = _get_client()

    if not client.has_collection(cfg.collection):
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        client.create_collection(
            collection_name=cfg.collection,
            schema=schema,
            metric_type="COSINE",
        )
        # 建向量索引（AUTOINDEX 自动选 HNSW，适配 COSINE）
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=_INDEX_PARAMS["index_type"],
            metric_type=_INDEX_PARAMS["metric_type"],
        )
        client.create_index(collection_name=cfg.collection, index_params=index_params)

    # 幂等加载：无论新建还是已存在，都确保 collection 已 load
    try:
        client.load_collection(cfg.collection)
    except Exception:
        # 已加载过会抛异常，忽略即可（幂等）
        pass


def insert(chunks: list[dict], vectors: list[list[float]]) -> int:
    """批量插入 chunk（text/category/source + vector）。

    Args:
        chunks: [{chunk_id, text, category, source}, ...]
        vectors: 与 chunks 对齐的向量

    Returns:
        插入条数。
    """
    cfg = get_milvus_config()
    ensure_collection()
    client = _get_client()

    data = []
    for chunk, vec in zip(chunks, vectors):
        data.append({
            "chunk_id": chunk["chunk_id"],
            "vector": vec,
            "text": chunk["text"],
            "category": chunk.get("category", ""),
            "source": chunk.get("source", ""),
        })
    result = client.insert(collection_name=cfg.collection, data=data)
    return len(data)


def search(
    query_vec: list[float],
    top_k: int = 5,
    category: str | None = None,
) -> list[dict]:
    """向量检索，返回 [{chunk_id, text, category, source, score}]。

    category 非空时加标量过滤（混合查询）。
    """
    cfg = get_milvus_config()
    client = _get_client()

    filter_expr = f'category == "{category}"' if category else None

    results = client.search(
        collection_name=cfg.collection,
        data=[query_vec],
        limit=top_k,
        filter=filter_expr,
        output_fields=["text", "category", "source"],
    )

    hits = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        hits.append({
            "chunk_id": hit.get("chunk_id") or hit.get("id"),
            "text": entity.get("text", ""),
            "category": entity.get("category", ""),
            "source": entity.get("source", ""),
            "score": hit.get("distance", 0.0),
        })
    return hits


def delete_by_chunk_ids(chunk_ids: list[str]) -> None:
    """按 chunk_id 精确删除（文档删除联动删向量）。"""
    from pymilvus import MilvusClient

    cfg = get_milvus_config()
    client = _get_client()
    if not chunk_ids:
        return
    client.delete(collection_name=cfg.collection, ids=chunk_ids)


def delete_by_prefix(prefix: str) -> None:
    """按 chunk_id 前缀删除（删除某文档的全部 chunk）。

    用 ``chunk_id like 'prefix%'`` 过滤表达式删除，避免先查 id 再删的两步操作。
    Milvus 的 VARCHAR 字段支持 ``like`` 前缀匹配。
    """
    from pymilvus import MilvusClient

    cfg = get_milvus_config()
    client = _get_client()
    if not client.has_collection(cfg.collection):
        return
    client.delete(
        collection_name=cfg.collection,
        filter=f'chunk_id like "{prefix}%"',
    )


def count() -> int:
    """返回 collection 当前实体数。

    注意：Milvus 的 get_collection_stats()['row_count'] 在部分版本对
    FLOAT_VECTOR + 未 flush 的数据返回 0（统计延迟），用 query 计数更可靠。
    用 query 拉主键计数，避免误判「未灌库」导致检索被跳过。
    """
    cfg = get_milvus_config()
    client = _get_client()
    if not client.has_collection(cfg.collection):
        return 0
    try:
        # query 不带 filter 时用恒真表达式拉主键，limit 足够大即可
        res = client.query(
            collection_name=cfg.collection,
            filter="chunk_id != \"\"",
            output_fields=["chunk_id"],
            limit=16384,
        )
        return len(res)
    except Exception:
        # query 失败（未加载等）退回 stats
        stats = client.get_collection_stats(cfg.collection)
        return int(stats.get("row_count", 0))
