"""Agent Run 追踪服务：run 生命周期 + 工具执行状态记录。

轻量版：只做 Run 级追踪与工具调用关联，不做全节点断点续跑。
写操作的「恢复」靠幂等键（idempotency_records）天然实现——重复请求命中已有
结果、跳过已执行写操作，不重放副作用。
"""
from __future__ import annotations

import uuid

from app.db.session import get_session
from app.db.models.agent_run import AgentRun, ToolExecution, RunStatus
from app.observability import ErrorCategory, ErrorCode


def new_run_id() -> str:
    """生成 run_id（uuid hex）。"""
    return uuid.uuid4().hex


def start_run(run_id: str, thread_id: str, session_factory=None) -> None:
    """创建 run 记录（running 状态）。

    ``session_factory`` 可注入（默认 get_session），测试可替换为 SQLite 内存库。
    """
    factory = session_factory or get_session
    with factory() as s:
        s.add(AgentRun(run_id=run_id, thread_id=thread_id, status=RunStatus.RUNNING.value))


def finish_run(run_id: str, status: RunStatus, error_code: str | None = None, error_category: str | None = None, session_factory=None) -> None:
    """结束 run（success / failed）。

    ``session_factory`` 可注入（默认 get_session），测试可替换为 SQLite 内存库。
    """
    factory = session_factory or get_session
    with factory() as s:
        run = s.query(AgentRun).filter(AgentRun.run_id == run_id).first()
        if run is None:
            return
        run.status = status.value if isinstance(status, RunStatus) else status
        run.error_code = error_code
        run.error_category = error_category


def record_tool_execution(
    run_id: str,
    tool_name: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """记录一次工具执行状态（关联 run_id）。"""
    with get_session() as s:
        s.add(
            ToolExecution(
                run_id=run_id,
                tool_name=tool_name,
                status=status,
                result=result,
                error=error,
            )
        )
