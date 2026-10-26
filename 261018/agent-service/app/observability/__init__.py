"""可观测性基础：错误分类 + 稳定错误码 + 结构化日志。

职责（Spec 约定）：
- 把底层异常归为稳定的错误分类（timeout / rate_limited / server_error /
  auth_error / config_error / client_error / network / tool_error / unknown）
- 提供面向用户的稳定错误码（不回传堆栈，只回错误码 + 安全提示）
- 结构化日志：异常详情（原始异常类型/状态码/堆栈/provider/model/耗时/重试
  次数/run_id/分类）只进受控日志，不进入面向用户的响应

设计原则（「保留并分类底层原因，对外只回稳定无敏感信息的错误码」）：
- **内部**：保留底层原因的最细粒度（原始异常类型 + HTTP 状态码 + 供应商 +
  模型 + 耗时 + 重试次数），写进受控日志，供事后诊断/告警/指标聚合。
- **对外**：只暴露稳定的错误码 + 安全提示，绝不回传堆栈、原始异常、供应商
  内部错误信息、api_key 等敏感字段。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

# 受控日志 logger（异常详情、可观测性字段都走这里，统一格式）
logger = logging.getLogger("agent.observability")


class ErrorCategory(str, Enum):
    """错误分类（稳定枚举，供指标聚合）。"""

    TIMEOUT = "timeout"  # 连接/读超时
    RATE_LIMITED = "rate_limited"  # 429 限流
    SERVER_ERROR = "server_error"  # 5xx 模型服务端错误
    AUTH_ERROR = "auth_error"  # 401/403 鉴权失败（api_key 无效/过期/无权限）
    CONFIG_ERROR = "config_error"  # 配置错误（缺 key/模型名错误/base_url 错误/400 参数错误）
    CLIENT_ERROR = "client_error"  # 其余 4xx（非 401/403/429）客户端错误
    NETWORK = "network"  # 连接失败/网络抖动
    TOOL_ERROR = "tool_error"  # 工具执行失败
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    """面向用户的稳定错误码（不暴露内部异常）。"""

    MODEL_UNAVAILABLE = "E_MODEL_UNAVAILABLE"  # 模型服务不可用/熔断
    RATE_LIMITED = "E_RATE_LIMITED"  # 限流
    TIMEOUT = "E_TIMEOUT"  # 超时
    TOOL_FAILED = "E_TOOL_FAILED"  # 工具执行失败
    NEED_RETRY = "E_NEED_RETRY"  # 建议稍后重试
    AUTH_FAILED = "E_AUTH_FAILED"  # 鉴权/配置失败（对用户稳定提示，不暴露具体供应商）
    CONFIG_ERROR = "E_CONFIG_ERROR"  # 服务配置错误（内部问题，对用户稳定提示）
    INTERNAL = "E_INTERNAL"  # 内部错误（兜底）


# 错误码 → 面向用户的安全提示（不含堆栈/内部信息）
SAFE_MESSAGE: dict[ErrorCode, str] = {
    ErrorCode.MODEL_UNAVAILABLE: "服务暂时不可用，请稍后重试或转接人工客服。",
    ErrorCode.RATE_LIMITED: "请求过于频繁，请稍后重试。",
    ErrorCode.TIMEOUT: "请求超时，请稍后重试。",
    ErrorCode.TOOL_FAILED: "操作未能完成，已为您转接人工客服。",
    ErrorCode.NEED_RETRY: "当前无法可靠完成您的请求，请稍后重试或转接人工客服。",
    ErrorCode.AUTH_FAILED: "服务暂时不可用，请稍后重试或转接人工客服。",
    ErrorCode.CONFIG_ERROR: "服务暂时不可用，请稍后重试或转接人工客服。",
    ErrorCode.INTERNAL: "服务异常，已为您转接人工客服。",
}


@dataclass
class ObservedError:
    """可观测性错误记录（只进日志，不进用户响应）。

    ``detail`` / ``exception_type`` / ``status_code`` 是**内部诊断信息**，
    仅写入受控日志，绝不进入面向用户的响应（用户只见 code + 安全提示）。
    """

    category: ErrorCategory
    code: ErrorCode
    provider: str | None = None  # 供应商（如 openai/deepseek/qwen）
    model: str | None = None
    elapsed_ms: float | None = None  # 耗时
    retries: int = 0  # 重试次数
    run_id: str | None = None
    tool_name: str | None = None  # 关联工具调用
    detail: str | None = None  # 异常详情（堆栈/原始错误，仅日志）
    exception_type: str | None = None  # 原始异常类名（仅日志，诊断用）
    status_code: int | None = None  # 底层 HTTP 状态码（仅日志，诊断用）
    extra: dict[str, Any] = field(default_factory=dict)

    def log(self, level: int = logging.ERROR) -> None:
        """把结构化错误写入受控日志（不含在面向用户的响应里）。"""
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["code"] = self.code.value
        logger.log(level, "observability_error %s", payload)


def classify_exception(exc: Exception) -> tuple[ErrorCategory, int | None]:
    """把底层异常归为稳定错误分类，并提取 HTTP 状态码（供诊断日志）。

    识别顺序：超时 → 限流 → 鉴权 → 配置 → 客户端错误 → 服务端错误 → 网络 → 未知。
    返回 (category, status_code)，status_code 仅在 httpx.HTTPStatusError 时非 None。

    分类语义：
    - 401/403 → AUTH_ERROR（api_key 无效/过期/无权限）
    - 400/404/422 → CONFIG_ERROR（模型名错误、base_url 路径错误等配置类问题）
    - 其余 4xx → CLIENT_ERROR
    - 429 → RATE_LIMITED
    - 5xx → SERVER_ERROR
    """
    import httpx

    # httpx 超时
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException)):
        return ErrorCategory.TIMEOUT, None

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return ErrorCategory.RATE_LIMITED, status
        if status in (401, 403):
            return ErrorCategory.AUTH_ERROR, status
        if status in (400, 404, 422):
            return ErrorCategory.CONFIG_ERROR, status
        if 400 <= status < 500:
            return ErrorCategory.CLIENT_ERROR, status
        return ErrorCategory.SERVER_ERROR, status

    # 连接/网络类错误
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError, OSError)):
        return ErrorCategory.NETWORK, None

    return ErrorCategory.UNKNOWN, None


def category_to_code(category: ErrorCategory) -> ErrorCode:
    """错误分类 → 面向用户稳定错误码。"""
    return {
        ErrorCategory.TIMEOUT: ErrorCode.TIMEOUT,
        ErrorCategory.RATE_LIMITED: ErrorCode.RATE_LIMITED,
        ErrorCategory.SERVER_ERROR: ErrorCode.MODEL_UNAVAILABLE,
        ErrorCategory.AUTH_ERROR: ErrorCode.AUTH_FAILED,
        ErrorCategory.CONFIG_ERROR: ErrorCode.CONFIG_ERROR,
        ErrorCategory.CLIENT_ERROR: ErrorCode.INTERNAL,
        ErrorCategory.NETWORK: ErrorCode.MODEL_UNAVAILABLE,
        ErrorCategory.TOOL_ERROR: ErrorCode.TOOL_FAILED,
        ErrorCategory.UNKNOWN: ErrorCode.INTERNAL,
    }[category]


def is_retryable(category: ErrorCategory) -> bool:
    """仅可重试错误才重试（4xx 客户端/鉴权/配置错误立即失败）。"""
    return category in {
        ErrorCategory.NETWORK,
        ErrorCategory.TIMEOUT,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.RATE_LIMITED,
    }
