"""对象存储层：MinIO 客户端 + 本地磁盘降级。

职责：上传/下载/删除文件（图片/附件），返回可引用的 URL 或 object key。

降级策略（对齐现有「无 key 降级」思路）：
- 配置了 MinIO（endpoint + access_key + secret_key）→ 走真实 MinIO
- 未配置 → 降级本地磁盘目录（``data/uploads/``），返回本地相对路径
"""
from __future__ import annotations

import io
from pathlib import Path

from app.config import get_minio_config

# 本地降级目录（项目根下）
_LOCAL_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"


def _get_client():
    """懒加载 Minio 客户端（进程内复用）。"""
    from minio import Minio

    cfg = get_minio_config()
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def _ensure_bucket() -> None:
    """确保 bucket 存在，不存在则创建。"""
    from minio import Minio

    cfg = get_minio_config()
    client = _get_client()
    if not client.bucket_exists(cfg.bucket):
        client.make_bucket(cfg.bucket)


def upload(object_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """上传文件到对象存储，返回可引用的 URL。

    - 有 MinIO → 上传到 bucket，返回 ``http://endpoint/bucket/object_key``
    - 无 MinIO → 降级写本地磁盘，返回 ``file://...`` 或相对路径

    Args:
        object_key: 对象键（如 ``images/abc123.png``）
        data: 文件字节
        content_type: 内容类型（用于浏览器预览）
    """
    cfg = get_minio_config()

    if not cfg.enabled:
        # 本地降级
        path = _LOCAL_DIR / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"file://{path.as_posix()}"

    from minio import Minio

    _ensure_bucket()
    client = _get_client()
    client.put_object(
        cfg.bucket,
        object_key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    scheme = "https" if cfg.secure else "http"
    return f"{scheme}://{cfg.endpoint}/{cfg.bucket}/{object_key}"


def download(object_key: str) -> bytes:
    """下载对象内容。"""
    cfg = get_minio_config()
    if not cfg.enabled:
        path = _LOCAL_DIR / object_key
        return path.read_bytes()

    _ensure_bucket()
    client = _get_client()
    resp = client.get_object(cfg.bucket, object_key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def delete(object_key: str) -> None:
    """删除对象。"""
    cfg = get_minio_config()
    if not cfg.enabled:
        path = _LOCAL_DIR / object_key
        if path.exists():
            path.unlink()
        return

    client = _get_client()
    if client.bucket_exists(cfg.bucket):
        client.remove_object(cfg.bucket, object_key)


def is_available() -> bool:
    """对象存储是否可用（有 MinIO 或本地降级目录可写）。"""
    return get_minio_config().enabled
