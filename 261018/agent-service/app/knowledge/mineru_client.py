"""MinerU 在线 API 客户端（Spec 10.2 / 10.7）。

接入方式：纯网络请求，调 MinerU 官方在线 API（base_url + token 可配置），
后续数据合规可平滑切换到私有化部署（改 base_url 即可）。

降级策略（离线可跑）：
- 配置了 MINERU_API_KEY → 走真实 MinerU /tasks 异步解析
- 未配置 → 本地降级解析器：纯文本类（md/txt）直接读内容，
  pdf/docx/xlsx/pptx 等二进制无法本地解析 → 抛 MinerUError 标记 failed
"""
from __future__ import annotations

import httpx

from app.config import get_mineru_config

# 纯文本类扩展名（本地降级可直接解析）
_TEXT_EXTENSIONS = {"md", "txt", "markdown", "text"}


class MinerUError(Exception):
    """MinerU 解析失败（网络/超时/无 key 解析二进制等）。"""


def submit_parse(file_bytes: bytes, file_type: str, file_name: str = "doc") -> str:
    """提交解析任务，返回 task_id。

    有 key 走真实 MinerU /tasks；无 key 且为纯文本类 → 走本地降级（task_id 用
    ``local:<hash>`` 占位），无 key 且为二进制 → 抛 MinerUError。
    """
    cfg = get_mineru_config()
    ext = file_type.lower().lstrip(".")

    if not cfg.enabled:
        if ext in _TEXT_EXTENSIONS:
            return _local_text_task(file_bytes)
        raise MinerUError(
            f"未配置 MINERU_API_KEY，无法解析 .{ext} 二进制文件；"
            "请配置 MinerU token 或改用文本类文件（md/txt）"
        )

    resp = httpx.post(
        f"{cfg.base_url.rstrip('/')}/tasks",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        files={"files": (file_name, file_bytes, _content_type(ext))},
        data={"return_md": "true", "lang_list": "ch"},
        timeout=cfg.timeout,
    )
    if resp.status_code >= 400:
        raise MinerUError(f"MinerU 提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    task_id = data.get("task_id") or data.get("data", {}).get("task_id")
    if not task_id:
        raise MinerUError(f"MinerU 响应缺少 task_id: {resp.text[:200]}")
    return task_id


def get_task_status(task_id: str) -> str:
    """查询解析任务状态。返回 queued/parsing/done/failed。

    本地降级任务（local: 前缀）直接返回 done。
    """
    if task_id.startswith("local:"):
        return "done"

    cfg = get_mineru_config()
    resp = httpx.get(
        f"{cfg.base_url.rstrip('/')}/tasks/{task_id}",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        timeout=cfg.timeout,
    )
    if resp.status_code >= 400:
        raise MinerUError(f"MinerU 查询状态失败 HTTP {resp.status_code}")
    data = resp.json()
    return _map_status(data)


def get_result(task_id: str) -> str:
    """获取解析结果 Markdown。

    本地降级任务返回占位（实际内容已在 submit 阶段落库，此路径不触发）。
    """
    if task_id.startswith("local:"):
        return ""

    cfg = get_mineru_config()
    resp = httpx.get(
        f"{cfg.base_url.rstrip('/')}/tasks/{task_id}/result",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        timeout=cfg.timeout,
    )
    if resp.status_code >= 400:
        raise MinerUError(f"MinerU 获取结果失败 HTTP {resp.status_code}")
    data = resp.json()
    # MinerU 结果字段：markdown 或 content
    md = (
        data.get("markdown")
        or data.get("content")
        or (data.get("data") or {}).get("markdown")
        or (data.get("data") or {}).get("content")
    )
    if not md:
        # 部分实现返回 md 字段
        md = data.get("md") or (data.get("data") or {}).get("md") or ""
    return md


def parse_local_text(file_bytes: bytes) -> str:
    """本地降级：把纯文本文件字节解码为文本（utf-8 优先，退 gbk）。"""
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _local_text_task(file_bytes: bytes) -> str:
    """为纯文本文件生成本地降级 task_id（内容由调用方直接落库）。"""
    import hashlib

    digest = hashlib.sha1(file_bytes).hexdigest()[:16]
    return f"local:{digest}"


def get_task_progress(task_id: str) -> int | None:
    """查询解析进度（0-100）。MinerU 未返回进度字段时返回 None。

    本地降级任务直接返回 100。
    """
    if task_id.startswith("local:"):
        return 100

    cfg = get_mineru_config()
    resp = httpx.get(
        f"{cfg.base_url.rstrip('/')}/tasks/{task_id}",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        timeout=cfg.timeout,
    )
    if resp.status_code >= 400:
        return None
    data = resp.json()
    raw = (
        data.get("progress")
        or data.get("percent")
        or (data.get("data") or {}).get("progress")
        or (data.get("data") or {}).get("percent")
    )
    if raw is None:
        return None
    try:
        p = float(raw)
        # 兼容 0-1 与 0-100 两种量纲
        if p <= 1.0:
            p *= 100
        return max(0, min(100, int(round(p))))
    except (TypeError, ValueError):
        return None


def _map_status(data: dict) -> str:
    """把 MinerU 状态响应映射到标准状态。"""
    state = (
        data.get("status")
        or data.get("state")
        or (data.get("data") or {}).get("status")
        or (data.get("data") or {}).get("state")
        or ""
    ).lower()
    # MinerU 状态取值可能是 pending/running/success/failed 等，做宽泛映射
    if state in ("done", "success", "succeeded", "finished", "completed"):
        return "done"
    if state in ("failed", "error", "cancelled", "canceled"):
        return "failed"
    if state in ("running", "processing", "parsing"):
        return "parsing"
    return "queued"


def _content_type(ext: str) -> str:
    """按扩展名返回 content-type（MinerU 上传用）。"""
    return {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "md": "text/markdown",
        "txt": "text/plain",
    }.get(ext, "application/octet-stream")
