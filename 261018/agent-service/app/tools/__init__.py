"""工具层包：统一执行框架 + 各工具实现。

- ``order``：订单工具（query_order / check_cancelable / cancel_order）
- ``registry``：ToolSpec + TOOL_WHITELIST + execute_tool（五道闸门）
- ``errors``：工具层异常与错误码
- ``idempotency``：幂等键生成/校验/记录
- ``permission``：thread→user 权限解析
"""
from app.tools.order import (
    check_cancelable,
    cancel_order,
    query_order,
    apply_refund,
    apply_return,
    update_address,
    CancelOrderOutput,
    REFUND_APPROVAL_THRESHOLD,
)
from app.tools.vision import (
    image_understand,
    build_vision_prompt,
    parse_structured,
    CONFIDENCE_THRESHOLD,
)
from app.tools.registry import (
    ToolSpec,
    TOOL_WHITELIST,
    execute_tool,
)
from app.tools import errors

__all__ = [
    "check_cancelable",
    "cancel_order",
    "query_order",
    "apply_refund",
    "apply_return",
    "update_address",
    "CancelOrderOutput",
    "REFUND_APPROVAL_THRESHOLD",
    "image_understand",
    "build_vision_prompt",
    "parse_structured",
    "CONFIDENCE_THRESHOLD",
    "ToolSpec",
    "TOOL_WHITELIST",
    "execute_tool",
    "errors",
]
