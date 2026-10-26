"""知识库路由：文档导入（MinerU 集成）+ 文档管理。

端点（Spec 10.5）：
- POST   /knowledge/import             上传文档发起导入
- GET    /knowledge/docs               文档列表
- GET    /knowledge/docs/{doc_id}      文档详情 + 预览
- POST   /knowledge/docs/{doc_id}/confirm     确认入库（切块向量化写 Milvus）
- POST   /knowledge/docs/{doc_id}/reparse     重新解析
- DELETE /knowledge/docs/{doc_id}      删除文档（含向量）
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.knowledge import importer, mineru_client
from app.db.models.knowledge import KnowledgeDocStatus

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# 支持的文件类型（Spec 10.1）
_ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "xlsx", "pptx", "md", "txt", "markdown"}
_MAX_SIZE = 50 * 1024 * 1024  # 50MB


def _ext_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/import")
async def import_document(
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
):
    """上传文档发起导入。

    文本类（md/txt）本地降级解析直接 done；二进制走 MinerU（无 key 标记 failed）。
    """
    filename = file.filename or "unnamed"
    ext = _ext_of(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: .{ext}")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="空文件")
    if len(content) > _MAX_SIZE:
        raise HTTPException(status_code=400, detail=f"文件超过 {_MAX_SIZE // 1024 // 1024}MB 限制")

    result = importer.create_document(filename, ext, content, category)

    # 文本类已 done；二进制尝试提交 MinerU（无 key 或失败标记 failed）
    if result["status"] == KnowledgeDocStatus.QUEUED.value:
        doc_id = result["doc_id"]
        try:
            task_id = mineru_client.submit_parse(content, ext, file_name=filename)
            # 更新 mineru_task + parsing 状态（文本本地降级不在此）
            importer._update_status(doc_id, KnowledgeDocStatus.PARSING.value, mineru_task=task_id)
            result["status"] = KnowledgeDocStatus.PARSING.value
        except mineru_client.MinerUError as e:
            importer._update_status(doc_id, KnowledgeDocStatus.FAILED.value, error_msg=str(e)[:500])
            result["status"] = KnowledgeDocStatus.FAILED.value
            result["error"] = str(e)

    return result


@router.get("/docs")
def list_docs():
    """文档列表（不含正文）。"""
    return importer.list_documents()


@router.get("/docs/{doc_id}")
def get_doc(doc_id: int):
    """文档详情 + Markdown 预览。"""
    doc = importer.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@router.post("/docs/{doc_id}/confirm")
def confirm_doc(doc_id: int):
    """确认入库：切块 → 向量化 → 写 Milvus → 标记 indexed。"""
    doc = importer.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not doc["preview_md"]:
        raise HTTPException(status_code=400, detail="文档尚未解析出内容，无法入库")
    try:
        return importer.index_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/docs/{doc_id}/reparse")
def reparse_doc(doc_id: int):
    """重新解析（幂等：先删旧向量，重新走解析 → 待确认入库）。"""
    doc = importer.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    # 回到 queued，等待再次解析/确认
    importer._update_status(doc_id, KnowledgeDocStatus.QUEUED.value, error_msg=None, chunk_count=None)
    return {"doc_id": doc_id, "status": KnowledgeDocStatus.QUEUED.value}


@router.delete("/docs/{doc_id}")
def delete_doc(doc_id: int):
    """删除文档（含 Milvus 向量）。"""
    doc = importer.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return importer.delete_document(doc_id)
