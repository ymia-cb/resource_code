"""文档导入编排（Spec 10.3 / 10.7）：解析 → 切块 → 向量化 → 入库 Milvus → 更新状态。

chunk_id 约定：``doc_{doc_id}_{idx}``，删除文档时按 ``doc_{doc_id}_`` 前缀联动删向量。

幂等性（Spec 10.8 约束 5）：
- 重新解析/确认入库前，先按前缀删除旧 chunk 向量，再插入新向量，保证不重复。
"""
from __future__ import annotations

from app.db.session import get_session
from app.db.models.knowledge import KnowledgeDoc, KnowledgeDocStatus
from app.knowledge import milvus_client, mineru_client
from app.knowledge.chunker import split_markdown, extract_images
from app.knowledge.embedding import embed
from app.knowledge.describe import describe_image, generate_filename
from app.storage import minio_client


# 状态 → 进度百分比（前端进度条用）。parsing 为区间起点，可由 MinerU 真实进度覆盖。
STATUS_PROGRESS: dict[str, int] = {
    KnowledgeDocStatus.UPLOADED.value: 5,
    KnowledgeDocStatus.QUEUED.value: 10,
    KnowledgeDocStatus.PARSING.value: 30,
    KnowledgeDocStatus.DONE.value: 95,
    KnowledgeDocStatus.INDEXED.value: 100,
    # failed 不固定，保留当前值；若需重置见 _update_status 调用处
}


def _default_progress(status: str) -> int:
    """状态 → 默认进度。failed 返回 0（前端红色展示）。"""
    if status == KnowledgeDocStatus.FAILED.value:
        return 0
    return STATUS_PROGRESS.get(status, 10)


def _chunk_id(doc_id: int, idx: int) -> str:
    return f"doc_{doc_id}_{idx}"


def _doc_prefix(doc_id: int) -> str:
    return f"doc_{doc_id}_"


def _get_doc(doc_id: int) -> KnowledgeDoc:
    with get_session() as s:
        doc = s.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if doc is None:
            raise ValueError(f"文档不存在: {doc_id}")
        # 脱离 session 后仍可读（expire_on_commit=False）
        return doc


def _update_status(doc_id: int, status: str, **fields) -> None:
    """更新文档状态与可选字段。

    自动附带状态对应的默认进度（除非调用方显式传 progress）。
    """
    with get_session() as s:
        doc = s.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if doc is None:
            return
        doc.status = status
        # 未显式指定 progress 时，按状态映射默认值（failed 保留原值避免进度倒退到 0）
        if "progress" not in fields and status != KnowledgeDocStatus.FAILED.value:
            doc.progress = _default_progress(status)
        for k, v in fields.items():
            setattr(doc, k, v)


def create_document(file_name: str, file_type: str, file_bytes: bytes, category: str | None) -> dict:
    """创建导入任务（落库 + 落盘元数据）。

    返回 {doc_id, status}。文件内容由调用方暂存；M1 简化：文本内容直接入
    markdown 字段，二进制文件仅记录元数据（需真实 MinerU 解析）。
    """
    with get_session() as s:
        doc = KnowledgeDoc(
            file_name=file_name,
            file_type=file_type,
            category=category,
            source=f"uploads/{file_name}",
            status=KnowledgeDocStatus.QUEUED.value,
            progress=_default_progress(KnowledgeDocStatus.QUEUED.value),
        )
        s.add(doc)
        s.flush()  # 拿到自增 id
        doc_id = doc.id
        # 纯文本文件：本地降级直接解析，无需 MinerU
        ext = file_type.lower().lstrip(".")
        if ext in {"md", "txt", "markdown", "text"}:
            try:
                doc.markdown = mineru_client.parse_local_text(file_bytes)
                doc.status = KnowledgeDocStatus.DONE.value
                doc.progress = STATUS_PROGRESS[KnowledgeDocStatus.DONE.value]
                # 用大模型生成规范文件名（防用户随意命名），失败降级保留原名
                try:
                    doc.file_name = f"{generate_filename(doc.markdown, file_name)}.{ext}"
                except Exception:
                    pass
            except Exception:
                doc.status = KnowledgeDocStatus.FAILED.value
                doc.progress = 0
                doc.error_msg = "本地文本解码失败"
        return {"doc_id": doc_id, "status": doc.status, "progress": doc.progress}


def parse_document(doc_id: int) -> dict:
    """对已建文档执行 MinerU 解析（异步任务入口）。

    流程：提交解析 → （同步阶段：文本类已 done 直接跳过）→ 拿 Markdown 落库。
    二进制文档需真实 MinerU：提交任务后进入 parsing 状态，由外部轮询。
    """
    doc = _get_doc(doc_id)
    ext = doc.file_type.lower().lstrip(".")

    # 纯文本已 done，跳过
    if doc.markdown and doc.status == KnowledgeDocStatus.DONE.value:
        return {"doc_id": doc_id, "status": doc.status}

    try:
        # 提交 MinerU 解析任务（无 key 且二进制 → 抛 MinerUError）
        # 注意：submit_parse 需要文件字节，这里从 markdown 字段回退读取已不可行，
        # 二进制文件字节由 create_document 阶段无法暂存 → 需在 import 端点直接持有。
        # 简化约定：parse_document 由 import 端点内联调用（见 routes），
        # 此函数保留给「重新解析」入口，用 mineru_task 状态轮询。
        task_id = doc.mineru_task
        if not task_id:
            raise mineru_client.MinerUError("文档未提交 MinerU 解析任务")
        status = mineru_client.get_task_status(task_id)
        if status == "done":
            md = mineru_client.get_result(task_id)
            _update_status(doc_id, KnowledgeDocStatus.DONE.value, markdown=md, error_msg=None)
            return {"doc_id": doc_id, "status": KnowledgeDocStatus.DONE.value}
        if status == "failed":
            _update_status(doc_id, KnowledgeDocStatus.FAILED.value, error_msg="MinerU 解析失败")
            return {"doc_id": doc_id, "status": KnowledgeDocStatus.FAILED.value}
        # parsing 中：透传 MinerU 真实进度（映射到 30~95 区间）
        progress = mineru_client.get_task_progress(task_id)
        if progress is not None:
            # MinerU 进度 0-100 → 导入进度 30-95（parsing 区间）
            mapped = 30 + int(progress * 0.65)
            _update_status(doc_id, KnowledgeDocStatus.PARSING.value, progress=mapped)
        else:
            _update_status(doc_id, KnowledgeDocStatus.PARSING.value)
        return {"doc_id": doc_id, "status": KnowledgeDocStatus.PARSING.value}
    except mineru_client.MinerUError as e:
        _update_status(doc_id, KnowledgeDocStatus.FAILED.value, error_msg=str(e)[:500])
        return {"doc_id": doc_id, "status": KnowledgeDocStatus.FAILED.value, "error": str(e)}


def process_images(md: str, doc_id: int) -> str:
    """处理 md 中的图片：上传 MinIO + VLM 描述 + 地址填回 md。

    流程：
    1. 提取 md 中的图片引用 ![](url)
    2. 下载图片字节（http 可访问时）
    3. 上传 MinIO 拿稳定 URL
    4. VLM 生成文字描述
    5. 把新 URL + 描述填回 md（替换原引用，描述附在图片下方）

    任何一步失败都不阻断入库：图片无法下载/上传时保留原引用，描述降级为 alt。
    """
    images = extract_images(md)
    if not images:
        return md

    result_md = md
    for i, img in enumerate(images):
        alt = img["alt"] or f"图片{i + 1}"
        old_url = img["url"]

        # 1. 下载图片字节
        data = _fetch_image_bytes(old_url)
        if data is None:
            # 无法获取，保留原引用（不填回）
            continue

        # 2. 上传 MinIO 拿稳定 URL
        ext = _guess_image_ext(old_url, data)
        object_key = f"images/doc_{doc_id}_{i}.{ext}"
        try:
            new_url = minio_client.upload(object_key, data, content_type=_content_type(ext))
        except Exception:
            continue  # 上传失败，保留原引用

        # 3. VLM 描述（降级为 alt）
        try:
            desc = describe_image(new_url, alt_text=alt)
        except Exception:
            desc = alt

        # 4. 填回 md：替换原图片引用，附加描述
        old_ref = f"![{img['alt']}]({old_url})"
        new_ref = f"![{alt}]({new_url})"
        result_md = result_md.replace(old_ref, f"{new_ref}\n> 图片内容：{desc}")

    return result_md


def _fetch_image_bytes(url: str) -> bytes | None:
    """下载图片字节。http/https 可下载；file:// 读本地；其他返回 None。"""
    import httpx

    if url.startswith(("http://", "https://")):
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None
    if url.startswith("file://"):
        try:
            from pathlib import Path

            return Path(url[7:]).read_bytes()
        except Exception:
            return None
    return None


def _guess_image_ext(url: str, data: bytes) -> str:
    """根据 URL 后缀或魔数猜图片扩展名。"""
    import re

    m = re.search(r"\.(png|jpe?g|gif|webp|bmp)(?:\?|$)", url, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # 魔数判断
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"GIF8"):
        return "gif"
    return "png"


def _content_type(ext: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


def index_document(doc_id: int, category: str | None = None) -> dict:
    """确认入库：把已解析的 Markdown 切块 → 向量化 → 写 Milvus → 标记 indexed。

    幂等：先按前缀删旧向量再插入。
    """
    doc = _get_doc(doc_id)
    if not doc.markdown:
        raise ValueError("文档尚未解析出 Markdown，无法入库")

    # 图片处理：提取 → 上传 MinIO → VLM 描述 → 填回 md
    md = process_images(doc.markdown, doc_id)

    chunks = split_markdown(md)
    if not chunks:
        _update_status(doc_id, KnowledgeDocStatus.FAILED.value, error_msg="切块结果为空")
        return {"doc_id": doc_id, "status": KnowledgeDocStatus.FAILED.value, "chunk_count": 0}

    cat = category or doc.category or "general"
    source = doc.source or f"uploads/{doc.file_name}"

    # 幂等：删除旧向量
    try:
        milvus_client.delete_by_prefix(_doc_prefix(doc_id))
    except Exception:
        pass  # 首次入库无旧向量，忽略

    records = [
        {
            "chunk_id": _chunk_id(doc_id, i),
            "text": c,
            "category": cat,
            "source": source,
        }
        for i, c in enumerate(chunks)
    ]
    vectors = [embed(c) for c in chunks]
    try:
        milvus_client.insert(records, vectors)
    except Exception as e:
        _update_status(doc_id, KnowledgeDocStatus.FAILED.value, error_msg=f"入库 Milvus 失败: {e}"[:500])
        return {"doc_id": doc_id, "status": KnowledgeDocStatus.FAILED.value, "chunk_count": 0}

    _update_status(
        doc_id,
        KnowledgeDocStatus.INDEXED.value,
        chunk_count=len(chunks),
        category=cat,
        source=source,
    )
    return {"doc_id": doc_id, "status": KnowledgeDocStatus.INDEXED.value, "chunk_count": len(chunks)}


def delete_document(doc_id: int) -> dict:
    """删除文档（含 Milvus 向量）与 MySQL 记录。"""
    # 先删向量（联动），再删记录
    try:
        milvus_client.delete_by_prefix(_doc_prefix(doc_id))
    except Exception:
        pass
    with get_session() as s:
        doc = s.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if doc is None:
            raise ValueError(f"文档不存在: {doc_id}")
        s.delete(doc)
    return {"doc_id": doc_id, "status": "deleted"}


def list_documents() -> list[dict]:
    """文档列表（不含 markdown 正文，减少传输）。"""
    with get_session() as s:
        docs = s.query(KnowledgeDoc).order_by(KnowledgeDoc.id.desc()).all()
        return [
            {
                "doc_id": d.id,
                "file_name": d.file_name,
                "file_type": d.file_type,
                "category": d.category,
                "source": d.source,
                "status": d.status,
                "progress": d.progress,
                "chunk_count": d.chunk_count,
                "error_msg": d.error_msg,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]


def get_document(doc_id: int) -> dict | None:
    """文档详情（含 markdown 预览）。"""
    with get_session() as s:
        d = s.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if d is None:
            return None
        return {
            "doc_id": d.id,
            "file_name": d.file_name,
            "file_type": d.file_type,
            "category": d.category,
            "source": d.source,
            "status": d.status,
            "progress": d.progress,
            "chunk_count": d.chunk_count,
            "preview_md": d.markdown,
            "error_msg": d.error_msg,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
