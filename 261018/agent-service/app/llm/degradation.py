"""降级策略：统一管理「真实模型不可用时怎么办」。

设计原则（回应「完善降级策略」）：
- **降级可观测**：每次降级都记录原因（未配置 / 调用失败 / 熔断）+ 供应商 + 模型 +
  角色，写结构化日志 + 累加指标，绝不静默降级。
- **降级分级**：不同 role 降级后果不同——
  - 可安全降级的 role（classify / rag / reflector）：降级到确定性规则/mock，
    仍能给出可用答案。
  - 不可安全降级的 role（planner / summarize / vlm）：降级会丢失关键能力或
    产生假答案，应明确标记，由上层决定「转人工」而非硬塞降级结果。
- **降级标记**：降级结果对外可感知（`degraded` 字段），用户/下游能区分
  「真实模型答案」与「降级答案」。

与 router.llm() 的关系：
- router.llm() 是唯一模型入口，降级在这里统一发生并记录；
- 本模块提供 ``DegradeReason`` / ``record_degradation`` / ``should_degrade_safely``
  三个基础件，供 router 与业务节点使用。
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger("agent.observability")


class DegradeReason(str, Enum):
    """降级原因（稳定枚举，供指标聚合与告警）。"""

    NOT_CONFIGURED = "not_configured"  # 未配置 api_key/model（预期离线/未接入）
    PROVIDER_FAILED = "provider_failed"  # 模型调用失败（超时/限流/鉴权/配置/服务端错误）
    BREAKER_OPEN = "breaker_open"  # 熔断打开，直接降级


# 可安全降级到 mock 的 role（降级后仍能给出可用答案）
_SAFE_DEGRADE_ROLES = {"classify", "rag", "reflector"}


def should_degrade_safely(role: str) -> bool:
    """判断某 role 降级到 mock 是否安全（不会产生假答案）。

    - classify / rag / reflector：确定性规则或基于检索拼接，降级安全。
    - planner / summarize / vlm：降级会丢失推理/视觉能力，不安全，应转人工。
    """
    return role in _SAFE_DEGRADE_ROLES


def record_degradation(
    role: str,
    reason: DegradeReason,
    *,
    provider: str | None = None,
    model: str | None = None,
    retries: int = 0,
    run_id: str | None = None,
    detail: str | None = None,
) -> None:
    """记录一次降级（结构化日志 + 指标），保证降级可观测、可告警。

    这是所有降级路径的**唯一记录入口**，业务代码不要绕过。
    """
    # 指标：按 provider 维度累加降级次数（供超时率/降级率大盘）
    try:
        from app.observability.breaker import metrics

        key = f"{provider or 'mock'}:degraded"
        metrics.incr(key)
        metrics.incr(f"{provider or 'mock'}:degraded:{reason.value}")
    except Exception:  # noqa: BLE001
        pass

    logger.warning(
        "llm_degraded role=%s reason=%s provider=%s model=%s retries=%d run_id=%s detail=%s",
        role,
        reason.value,
        provider,
        model,
        retries,
        run_id,
        detail,
    )
