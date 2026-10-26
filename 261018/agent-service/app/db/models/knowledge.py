"""知识库文档表（KnowledgeDoc）。

支撑 RAG 文档导入（Spec 10.4）：前端上传文档 → MinerU 解析 → 切块向量化 → 入库 Milvus。
状态机：

    uploaded → queued → parsing → done → (确认) → indexed
                      ↘ failed → 可重新解析
"""
from __future__ import annotations

from enum import Enum

from sqlalchemy import (
    Column,
    String,
    BigInteger,
    Integer,
    Text,
    DateTime,
    func,
    Index,
)

from app.db.base import Base


class KnowledgeDocStatus(str, Enum):
    """文档导入状态。"""

    UPLOADED = "uploaded"  # 已上传（文件已落盘）
    QUEUED = "queued"  # 已入队
    PARSING = "parsing"  # MinerU 解析中
    DONE = "done"  # 解析完成（待确认入库）
    INDEXED = "indexed"  # 已入库（向量已写 Milvus）
    FAILED = "failed"  # 解析失败（可重新解析）


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"
    __table_args__ = (
        Index("ix_knowledge_docs_status", "status"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(16), nullable=False)  # pdf/image/docx/xlsx/pptx/md/txt
    category = Column(String(64), nullable=True)  # 人工指定分类
    source = Column(String(255), nullable=True)  # 溯源路径
    status = Column(
        String(16),
        nullable=False,
        default=KnowledgeDocStatus.QUEUED.value,
    )
    mineru_task = Column(String(64), nullable=True)  # MinerU 任务 id
    chunk_count = Column(Integer, nullable=True)  # 切块数量
    progress = Column(Integer, nullable=False, default=0)  # 导入进度 0-100（前端进度条）
    markdown = Column(Text, nullable=True)  # 解析出的 Markdown（预览用）
    error_msg = Column(String(512), nullable=True)  # 失败原因
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
