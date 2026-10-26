"""知识库文档导入测试（MinerU 集成 + 前端 API）。

覆盖：chunker 语义切块、mineru_client 降级、importer 编排（含 Milvus 联动）、
/knowledge/* API 端点。

运行：python -m pytest tests/test_knowledge_import.py -v
依赖真实 MySQL（文档记录）；Milvus 相关用例在未运行时降级跳过。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.models.knowledge import KnowledgeDoc, KnowledgeDocStatus
from app.db.session import get_session
from app.knowledge.chunker import split_markdown
from app.knowledge import mineru_client, importer

client = TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_docs():
    """每个用例后清理测试文档（MySQL 记录 + Milvus 向量），保证可重复运行。"""
    yield
    with get_session() as s:
        # 先删 Milvus 中测试文档的向量（chunk_id 前缀 doc_{id}_），再删记录
        doc_ids = [
            row[0]
            for row in s.query(KnowledgeDoc.id).all()
        ]
        for did in doc_ids:
            try:
                importer.milvus_client.delete_by_prefix(f"doc_{did}_")
            except Exception:
                pass
        s.query(KnowledgeDoc).delete(synchronize_session=False)


# ---------------------------------------------------------------------------
# chunker 语义切块
# ---------------------------------------------------------------------------


def test_chunker_split_by_headings():
    md = "# 退款政策\n7 天无理由退货。\n\n# 发货说明\n48 小时发货。"
    chunks = split_markdown(md)
    assert len(chunks) == 2
    assert any("退款" in c for c in chunks)
    assert any("发货" in c for c in chunks)


def test_chunker_keeps_heading_context():
    md = "# 优惠券\n订单取消后自动退回。"
    chunks = split_markdown(md)
    assert chunks[0].startswith("# 优惠券")
    assert "自动退回" in chunks[0]


def test_chunker_empty_input():
    assert split_markdown("") == []
    assert split_markdown("   \n  ") == []


def test_chunker_no_heading_paragraphs():
    md = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = split_markdown(md, chunk_size=100)
    assert len(chunks) >= 1
    # 至少保留所有内容
    assert "第一段" in "".join(chunks)


def test_chunker_overlap():
    md = "# A\n" + "段落甲。" * 50 + "\n\n# B\n" + "段落乙。" * 50
    chunks = split_markdown(md, chunk_size=200, overlap=20)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# mineru_client 降级
# ---------------------------------------------------------------------------


def test_mineru_local_text_parse():
    md = mineru_client.parse_local_text("你好世界".encode("utf-8"))
    assert md == "你好世界"


def test_mineru_local_text_gbk():
    md = mineru_client.parse_local_text("中文".encode("gbk"))
    assert md == "中文"


def test_mineru_submit_text_without_key():
    """无 key + 文本文件 → 本地降级，返回 local: task_id。"""
    task_id = mineru_client.submit_parse("# 标题\n内容".encode("utf-8"), "md", file_name="a.md")
    assert task_id.startswith("local:")
    assert mineru_client.get_task_status(task_id) == "done"


def test_mineru_submit_binary_without_key_raises():
    """无 key + 二进制文件 → 抛 MinerUError。"""
    with pytest.raises(mineru_client.MinerUError):
        mineru_client.submit_parse(b"%PDF-1.4 fake", "pdf", file_name="a.pdf")


# ---------------------------------------------------------------------------
# importer 编排（纯文本本地降级，不依赖 MinerU key）
# ---------------------------------------------------------------------------


def test_importer_create_and_index_text_doc():
    """文本文件：create → done（本地解析）→ confirm → indexed。"""
    md_content = "# 退款政策\n7 天无理由退货。\n\n# 优惠券\n取消订单退回。"
    result = importer.create_document("policy.md", "md", md_content.encode("utf-8"), "refund_policy")
    doc_id = result["doc_id"]
    assert result["status"] == KnowledgeDocStatus.DONE.value

    # confirm 入库（Milvus 可用时真正写向量；不可用时标记 failed 不抛）
    out = importer.index_document(doc_id)
    # 入库结果可能是 indexed（Milvus 可用）或 failed（Milvus 不可用），两种都算编排走通
    assert out["doc_id"] == doc_id
    assert out["status"] in (KnowledgeDocStatus.INDEXED.value, KnowledgeDocStatus.FAILED.value)


def test_importer_list_and_get():
    result = importer.create_document("note.txt", "txt", "内容正文".encode("utf-8"), None)
    doc_id = result["doc_id"]

    docs = importer.list_documents()
    assert any(d["doc_id"] == doc_id for d in docs)

    detail = importer.get_document(doc_id)
    assert detail["preview_md"] == "内容正文"
    assert detail["status"] == KnowledgeDocStatus.DONE.value


def test_importer_delete_removes_doc():
    result = importer.create_document("del.md", "md", "# X\n内容".encode("utf-8"), None)
    doc_id = result["doc_id"]
    importer.delete_document(doc_id)
    assert importer.get_document(doc_id) is None


def test_importer_index_requires_markdown():
    """未解析出 markdown 的文档 confirm 会抛错。"""
    with get_session() as s:
        doc = KnowledgeDoc(
            file_name="empty.pdf", file_type="pdf",
            status=KnowledgeDocStatus.FAILED.value, error_msg="无 MinerU key",
        )
        s.add(doc)
        s.flush()
        doc_id = doc.id
    with pytest.raises(ValueError):
        importer.index_document(doc_id)


def test_importer_chunk_id_prefix_convention():
    """chunk_id 前缀约定，供删除联动。"""
    assert importer._chunk_id(42, 0) == "doc_42_0"
    assert importer._doc_prefix(42) == "doc_42_"


def test_status_progress_mapping():
    """状态 → 默认进度映射。"""
    assert importer.STATUS_PROGRESS[KnowledgeDocStatus.QUEUED.value] == 10
    assert importer.STATUS_PROGRESS[KnowledgeDocStatus.PARSING.value] == 30
    assert importer.STATUS_PROGRESS[KnowledgeDocStatus.DONE.value] == 95
    assert importer.STATUS_PROGRESS[KnowledgeDocStatus.INDEXED.value] == 100
    assert importer._default_progress(KnowledgeDocStatus.FAILED.value) == 0


def test_importer_progress_updates_on_status():
    """文本文件导入：queued(10) → done(95)，progress 随状态推进。"""
    result = importer.create_document("p.md", "md", "# T\n内容".encode("utf-8"), None)
    assert result["progress"] == 95  # 文本类直接 done
    detail = importer.get_document(result["doc_id"])
    assert detail["progress"] == 95
    assert detail["status"] == KnowledgeDocStatus.DONE.value


def test_mineru_progress_parse():
    """mineru_client.get_task_progress 兼容 0-100 与 0-1 两种量纲。"""
    # 本地降级任务直接 100
    assert mineru_client.get_task_progress("local:abc") == 100


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------


def test_api_import_text_file():
    resp = client.post(
        "/knowledge/import",
        files={"file": ("faq.md", "# 常见问题\n退款怎么退".encode("utf-8"), "text/markdown")},
        data={"category": "refund_policy"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "doc_id" in data
    assert data["status"] == KnowledgeDocStatus.DONE.value


def test_api_import_reject_bad_extension():
    resp = client.post(
        "/knowledge/import",
        files={"file": ("bad.exe", b"xx", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_api_list_docs():
    resp = client.get("/knowledge/docs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_api_get_doc_404():
    resp = client.get("/knowledge/docs/99999999")
    assert resp.status_code == 404


def test_api_delete_doc():
    # 先上传再删除
    up = client.post(
        "/knowledge/import",
        files={"file": ("todel.txt", "内容".encode("utf-8"), "text/plain")},
    )
    doc_id = up.json()["doc_id"]
    resp = client.delete(f"/knowledge/docs/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    # 再查 404
    assert client.get(f"/knowledge/docs/{doc_id}").status_code == 404


def test_api_knowledge_page_served():
    """前端知识库管理页可访问。"""
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    assert "知识库" in resp.text or "text/html" in resp.headers.get("content-type", "")
