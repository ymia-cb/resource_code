"""自研极简熔断器（进程内）+ 指标计数器。

设计（低预算，不引入 Prometheus/OpenTelemetry）：
- 熔断器：按 provider 维护连续失败计数，连续失败达阈值 → 打开（open），
  冷却期内直接拒绝（降级 mock），冷却期后进入半开（half-open）放行一个探测请求。
- 指标：进程内计数器 + 结构化日志（供事后聚合），统计超时/错误/熔断/降级。

线程安全：用 threading.Lock 保护计数。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

from app.observability import ErrorCategory, logger

# 熔断阈值与冷却（秒）
FAILURE_THRESHOLD = 3  # 连续失败 3 次 → 熔断
COOLDOWN_SECONDS = 30.0  # 冷却 30s 后半开


class CircuitState:
    CLOSED = "closed"  # 正常
    OPEN = "open"  # 熔断中（直接拒绝）
    HALF_OPEN = "half_open"  # 半开（放行一个探测）


class CircuitBreaker:
    """按 provider 维度熔断。"""

    def __init__(self, threshold: int = FAILURE_THRESHOLD, cooldown: float = COOLDOWN_SECONDS):
        self.threshold = threshold
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._state: dict[str, str] = defaultdict(lambda: CircuitState.CLOSED)
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}

    def allow(self, provider: str) -> bool:
        """请求前判断是否放行。熔断打开且未过冷却期 → 拒绝。"""
        with self._lock:
            state = self._state[provider]
            if state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at.get(provider, 0.0)
                if elapsed < self.cooldown:
                    return False
                # 冷却期过 → 半开，放行一个探测
                self._state[provider] = CircuitState.HALF_OPEN
            return True

    def record_success(self, provider: str) -> None:
        """请求成功 → 复位。"""
        with self._lock:
            self._state[provider] = CircuitState.CLOSED
            self._consecutive_failures[provider] = 0

    def record_failure(self, provider: str) -> bool:
        """请求失败 → 累加；达阈值则打开熔断，返回是否触发熔断。"""
        with self._lock:
            self._consecutive_failures[provider] += 1
            if self._consecutive_failures[provider] >= self.threshold:
                self._state[provider] = CircuitState.OPEN
                self._opened_at[provider] = time.monotonic()
                return True
            return False

    def state(self, provider: str) -> str:
        with self._lock:
            return self._state[provider]


class Metrics:
    """进程内指标计数器（按 provider + 维度）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)

    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counters[key] += n

    def count(self, key: str) -> int:
        with self._lock:
            return self._counters[key]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


# 进程级单例
breaker = CircuitBreaker()
metrics = Metrics()


def _metric_key(provider: str, kind: str) -> str:
    return f"{provider}:{kind}"


def record_request(provider: str, category: ErrorCategory | None, retries: int, elapsed_ms: float, degraded: bool) -> None:
    """记录一次模型请求到指标 + 结构化日志。

    - category 为 None 表示成功
    - degraded 表示最终降级到 mock
    """
    metrics.incr(f"{provider}:total")
    if category is None:
        metrics.incr(f"{provider}:success")
    else:
        metrics.incr(f"{provider}:error")
        metrics.incr(f"{provider}:error:{category.value}")
        if category == ErrorCategory.TIMEOUT:
            metrics.incr(f"{provider}:timeout")
    if degraded:
        metrics.incr(f"{provider}:degraded")
    if retries > 0:
        metrics.incr(f"{provider}:retried")

    logger.info(
        "model_request provider=%s category=%s retries=%d elapsed_ms=%.1f degraded=%s",
        provider,
        category.value if category else "ok",
        retries,
        elapsed_ms,
        degraded,
    )


def record_breaker_open(provider: str) -> None:
    """熔断打开时记录指标。"""
    metrics.incr(f"{provider}:breaker_open")
    logger.warning("circuit_breaker_open provider=%s", provider)
