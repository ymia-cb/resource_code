"""OpenAI 兼容接口 Provider（纯 httpx，不依赖 openai SDK）。

适配所有 OpenAI 兼容的 /chat/completions 端点（OpenAI / DeepSeek / Qwen /
Moonshot / 本地 vLLM/Ollama 等），通过 base_url + api_key + model 配置切换。

健壮性设计（对应当前 harness 要求）：
- **连接超时 / 读超时分开**：connect_timeout 短（及时失败），读超时用 timeout。
- **退避重试 + 随机抖动**：仅对可重试错误（网络抖动 / 超时 / 5xx / 429）做
  少量指数退避重试，带随机抖动避免惊群；**4xx（非 429）立即失败不重试**
  （401/400 重试无意义）。
- **熔断**：请求前经 CircuitBreaker 放行，连续失败达阈值打开熔断直接降级。
- **结构化日志**：耗时/重试次数/错误分类/供应商/模型 进受控日志。
- 失败抛 ``ProviderError``（携带 category/code），由 router 层统一降级 mock。
"""
from __future__ import annotations

import random
import time

import httpx

from langchain_core.messages import BaseMessage

from app.llm.providers.base import Provider
from app.observability import (
    ErrorCategory,
    ErrorCode,
    classify_exception,
    category_to_code,
    is_retryable,
)
from app.observability.breaker import breaker, record_request, record_breaker_open


class ProviderError(Exception):
    """模型调用失败（携带错误分类与稳定错误码，供上层降级/转人工）。

    ``detail`` / ``exception_type`` / ``status_code`` 是**内部诊断信息**，
    只进受控日志，不进入面向用户的响应。
    """

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        code: ErrorCode | None = None,
        retries: int = 0,
        exception_type: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.code = code or category_to_code(category)
        self.retries = retries
        self.exception_type = exception_type
        self.status_code = status_code


# 可重试的错误分类（网络抖动/超时/服务端错误/限流）——见 observability.is_retryable
_RETRYABLE_CATEGORIES = {
    ErrorCategory.NETWORK,
    ErrorCategory.TIMEOUT,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.RATE_LIMITED,
}


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    """把 langchain BaseMessage 列表转成 OpenAI chat 消息格式。

    支持多模态 content（content 为 list[dict] 时原样透传，如
    [{"type":"text",...},{"type":"image_url",...}]）。
    """
    result = []
    for m in messages:
        role = getattr(m, "type", "user")
        role = {"ai": "assistant", "human": "user"}.get(role, role)
        content = getattr(m, "content", "")
        result.append({"role": role, "content": content})
    return result


def _should_retry(category: ErrorCategory) -> bool:
    """仅可重试错误才重试（4xx 客户端/鉴权/配置错误立即失败）。"""
    return is_retryable(category)


class OpenAICompatProvider(Provider):
    """OpenAI 兼容 chat provider。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        provider_name: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.temperature = temperature
        # 供应商名（供熔断/指标按 provider 维度聚合）
        self.provider_name = provider_name or self._infer_provider()

    def _infer_provider(self) -> str:
        """从 base_url 推断供应商名（用于熔断/指标维度）。"""
        url = self.base_url.lower()
        for name in ("deepseek", "qwen", "moonshot", "openai", "ollama", "vllm", "gemini"):
            if name in url:
                return name
        return "openai_compat"

    def chat(self, messages: list[BaseMessage]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # 连接超时与读超时分开
        timeout_cfg = httpx.Timeout(
            timeout=self.timeout,
            connect=self.connect_timeout,
        )

        # 熔断前置判断：打开则直接降级
        if not breaker.allow(self.provider_name):
            record_breaker_open(self.provider_name)
            raise ProviderError(
                f"熔断中，模型服务暂不可用（{self.model}）",
                category=ErrorCategory.SERVER_ERROR,
                code=ErrorCode.MODEL_UNAVAILABLE,
                retries=0,
            )

        start = time.monotonic()
        last_error: Exception | None = None
        last_category = ErrorCategory.UNKNOWN
        last_status_code: int | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_cfg,
                )
                resp.raise_for_status()
                data = resp.json()
                elapsed = (time.monotonic() - start) * 1000
                breaker.record_success(self.provider_name)
                record_request(self.provider_name, None, attempt, elapsed, degraded=False)
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last_error = e
                last_category, last_status_code = classify_exception(e)
                # 4xx（非 429，含鉴权/配置错误）→ 不可重试，立即失败
                if not _should_retry(last_category):
                    break
                if attempt < self.max_retries:
                    # 指数退避 + 随机抖动（避免惊群）
                    backoff = (2**attempt) * 0.5 + random.uniform(0, 0.5)
                    time.sleep(backoff)
                    continue

        elapsed = (time.monotonic() - start) * 1000
        retries = attempt
        # 记录熔断失败
        opened = breaker.record_failure(self.provider_name)
        record_request(self.provider_name, last_category, retries, elapsed, degraded=True)

        code = category_to_code(last_category)
        raise ProviderError(
            f"OpenAI 兼容接口调用失败（{self.model}）: {last_error}",
            category=last_category,
            code=code,
            retries=retries,
            exception_type=type(last_error).__name__ if last_error else None,
            status_code=last_status_code,
        )
