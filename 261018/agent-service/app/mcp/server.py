"""MCP Streamable HTTP 端点（手写轻量实现，不依赖 mcp SDK）。

遵循 MCP Streamable HTTP 传输（2026-07-28 修订）：
- 单端点 /mcp，每个消息独立 POST
- JSON-RPC 2.0 消息格式（jsonrpc/id/method/params）
- 支持方法：initialize / tools/list / tools/call / ping
- 响应为单个 JSON（result 或 error）
- 鉴权：Bearer token（MCP_API_TOKEN 配置，未配置则不鉴权）

工具统一经 ``execute_tool`` 五道闸门执行，与内部工具层共用同一安全边界。
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.config import get_mcp_config
from app.tools.registry import TOOL_WHITELIST, execute_tool
from app.tools.errors import ToolError, NeedApproval

router = APIRouter(tags=["mcp"])

# 协议版本（与请求体 _meta 保持一致）
PROTOCOL_VERSION = "2026-07-28"
SERVER_NAME = "ecommerce-customer-service-agent"
SERVER_VERSION = "0.1.0"

# JSON-RPC 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _jsonrpc_result(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# 工具 schema 生成（tools/list 用）
# ---------------------------------------------------------------------------


def _tool_to_mcp_schema(name: str, spec) -> dict:
    """把 ToolSpec 转成 MCP 工具定义（含 inputSchema）。"""
    schema = spec.schema.model_json_schema()
    return {
        "name": name,
        "description": spec.description,
        "inputSchema": schema,
    }


# ---------------------------------------------------------------------------
# JSON-RPC 方法分发
# ---------------------------------------------------------------------------


def _handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _handle_tools_list(params: dict) -> dict:
    tools = [_tool_to_mcp_schema(name, spec) for name, spec in TOOL_WHITELIST.items()]
    return {"tools": tools}


def _handle_tools_call(params: dict) -> dict:
    """执行工具调用（经 execute_tool 五道闸门）。

    params 含 name（工具名）与 arguments（参数）。thread_id 从 arguments 或
    _meta 中取（MCP 调用方需传 thread_id 以做权限隔离）。
    """
    name = params.get("name")
    arguments = params.get("arguments") or {}

    # thread_id：从 arguments 提取（业务约定），缺失则用 "mcp" 兜底
    thread_id = arguments.pop("thread_id", None) or "mcp"

    try:
        result = execute_tool(name, arguments, thread_id)
    except NeedApproval as e:
        return {
            "content": [{"type": "text", "text": f"需人工审批: {e}"}],
            "isError": True,
        }
    except ToolError as e:
        return {
            "content": [{"type": "text", "text": f"{e.code}: {e}"}],
            "isError": True,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "content": [{"type": "text", "text": f"执行失败: {e}"}],
            "isError": True,
        }

    # 结果序列化为文本（MCP tool result 约定 content 数组）
    return {
        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
        "isError": False,
    }


def _dispatch(method: str, params: dict | None) -> tuple[Any, bool]:
    """分发 JSON-RPC 方法。返回 (result, is_error)。"""
    params = params or {}
    if method == "initialize":
        return _handle_initialize(params), False
    if method == "ping":
        return {}, False
    if method == "tools/list":
        return _handle_tools_list(params), False
    if method == "tools/call":
        return _handle_tools_call(params), False
    if method == "notifications/initialized":
        # 通知，无需响应
        return None, False
    return None, True  # 未知方法


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP Streamable HTTP 端点。"""
    # 鉴权（Bearer token）
    cfg = get_mcp_config()
    if cfg.auth_enabled:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {cfg.api_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    # 解析 JSON-RPC 请求体
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            _jsonrpc_error(None, PARSE_ERROR, "Parse error"),
            status_code=400,
        )

    # 校验 JSON-RPC 基本结构
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        return JSONResponse(
            _jsonrpc_error(body.get("id") if isinstance(body, dict) else None,
                           INVALID_REQUEST, "Invalid Request"),
            status_code=400,
        )

    method = body.get("method")
    params = body.get("params")
    req_id = body.get("id")

    result, is_error = _dispatch(method, params)
    if is_error:
        return JSONResponse(
            _jsonrpc_error(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        )
    if method == "notifications/initialized":
        # 通知：返回 202 Accepted，无响应体
        return JSONResponse({}, status_code=202)

    return JSONResponse(_jsonrpc_result(req_id, result))
