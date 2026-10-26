"""工具层异常与错误码（对齐 Spec 4.6）。

五道闸门失败时抛出对应异常，由上层（graph 节点 / API 层）转人工或返回修正。
"""
from __future__ import annotations


# 错误码常量
TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"  # 工具不在白名单
PARAM_INVALID = "PARAM_INVALID"  # 参数校验失败
PERMISSION_DENIED = "PERMISSION_DENIED"  # 越权访问
DUPLICATE_REQUEST = "DUPLICATE_REQUEST"  # 幂等冲突
NEED_APPROVAL = "NEED_APPROVAL"  # 需人工审批


class ToolError(Exception):
    """工具层异常基类。"""

    code: str = "TOOL_ERROR"

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class ToolNotAllowed(ToolError):
    """闸门①：工具不在白名单。"""

    code = TOOL_NOT_ALLOWED


class ParamInvalid(ToolError):
    """闸门②：参数校验失败（pydantic）。"""

    code = PARAM_INVALID


class PermissionDenied(ToolError):
    """闸门③：越权访问（thread → user 隔离失败）。"""

    code = PERMISSION_DENIED


class DuplicateRequest(ToolError):
    """闸门④：幂等冲突（重复请求，返回已有结果不重复执行）。"""

    code = DUPLICATE_REQUEST

    def __init__(self, message: str, existing_result: dict | None = None):
        super().__init__(message)
        self.existing_result = existing_result


class NeedApproval(ToolError):
    """高风险操作：金额超阈值，转人工审批。"""

    code = NEED_APPROVAL
