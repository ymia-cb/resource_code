"""工具调用超时中间件。

在 execute_tool 外层统一收口超时：工具调用超过阈值 → 返回稳定错误结果 + 记录，
避免单个工具挂死拖垮整个 Agent Run。

超时阈值按业务类型差异化（见 app/skills.TypeDefinition.timeout_seconds）：
- 只读查询 10s、写操作 15s、多模态 VLM 20s。
"""
from __future__ import annotations

import concurrent.futures

from app.middleware import Middleware, ToolContext
from app.skills import tool_type, TYPE_DEFINITIONS


class ToolTimeoutMiddleware(Middleware):
    """工具调用超时控制。"""

    def __init__(self, default_timeout: float = 10.0):
        self.default_timeout = default_timeout

    def timeout_for(self, tool_name: str) -> float:
        """按工具的业务类型查超时阈值。"""
        bt = tool_type(tool_name)
        if bt is not None and bt in TYPE_DEFINITIONS:
            return TYPE_DEFINITIONS[bt].timeout_seconds
        return self.default_timeout

    def after_tool(self, ctx: ToolContext, result: dict) -> dict:
        """工具调用后处理（此处不做阻塞，超时由 executor 用 run_with_timeout 包裹）。

        中间件本身不阻塞执行；真正的超时在 executor 通过
        ``run_tool_with_timeout`` 包裹。此处保留钩子以便记录/改写超时结果。
        """
        return result


def run_tool_with_timeout(
    fn,
    timeout: float,
    *args,
    **kwargs,
):
    """在超时保护下执行工具函数。

    Args:
        fn: 工具函数。
        timeout: 超时秒数。
    Returns:
        (是否超时, 结果或异常)。
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return False, result
        except concurrent.futures.TimeoutError:
            future.cancel()
            return True, None
        except Exception as e:  # noqa: BLE001
            return False, e
