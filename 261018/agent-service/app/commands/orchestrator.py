"""命令编排器：按依赖与风险编排命令，支持只读并发 + 写串行。

核心规则（确定性，不交给模型）：
- 依赖：命令声明 depends_on，编排器做拓扑排序。
- 风险：只读命令（无副作用）可并发执行；写命令必须串行。
- 互斥：同批次命令中出现互斥对 → 拒绝执行，返回矛盾提示（要求澄清），
  绝不自行选择一种结果。
- 合并：只读并发结果聚合为一份 merged_tool_result；写命令逐项记录，
  部分完成逐项告知。

「开启并行」的落地：无依赖只读命令用 asyncio.gather 并发；写命令顺序执行。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from app.commands import (
    CommandSpec,
    detect_conflicts,
    get_command,
    is_write_command,
)


@dataclass
class OrchestratedStep:
    """编排后的一个执行步骤。"""

    command: str
    params: dict
    level: int  # 拓扑层级（同层可并行）
    write: bool  # 是否写命令


@dataclass
class ConflictResult:
    """矛盾检测结果。"""

    has_conflict: bool
    pairs: list[tuple[str, str]] = field(default_factory=list)


def detect_conflicts_in_plan(commands: list[str]) -> ConflictResult:
    """检测计划命令中的矛盾。"""
    pairs = detect_conflicts(commands)
    return ConflictResult(has_conflict=bool(pairs), pairs=pairs)


def topo_sort(command_names: list[str]) -> list[list[str]]:
    """按依赖拓扑分层。

    Returns:
        [[cmd,...], [cmd,...], ...] 每层内部可并行，层间串行。
    无依赖命令都在第 0 层。
    """
    specs = {name: get_command(name) for name in command_names}
    # 计算入度（依赖数）
    indegree = {name: 0 for name in command_names}
    dependents: dict[str, list[str]] = {name: [] for name in command_names}
    for name in command_names:
        spec = specs.get(name)
        if spec is None:
            continue
        for dep in spec.depends_on:
            if dep in command_names:
                indegree[name] += 1
                dependents[dep].append(name)

    levels: list[list[str]] = []
    remaining = list(command_names)
    while remaining:
        # 当前入度为 0 的节点
        ready = sorted([n for n in remaining if indegree[n] == 0])
        if not ready:
            # 环（本系统依赖树简单，不应出现；防御性：全部塞一层）
            levels.append(sorted(remaining))
            break
        levels.append(ready)
        for n in ready:
            remaining.remove(n)
            for dep in dependents[n]:
                indegree[dep] -= 1
    return levels


def orchestrate(command_names: list[str], params_map: dict[str, dict]) -> list[OrchestratedStep]:
    """把一组命令编排成「可并行/串行」的执行步骤。

    Returns:
        按执行顺序排列的步骤列表；同 level 且均为只读的步骤可并行。
    """
    levels = topo_sort(command_names)
    steps: list[OrchestratedStep] = []
    for level_idx, level in enumerate(levels):
        for name in level:
            steps.append(
                OrchestratedStep(
                    command=name,
                    params=params_map.get(name, {}),
                    level=level_idx,
                    write=is_write_command(name),
                )
            )
    return steps


def _run_parallel_reads(steps: list[OrchestratedStep], execute: Callable) -> list[dict]:
    """并发执行同层只读命令。execute 返回 {tool, data} 或抛异常。"""
    async def _run():
        async def _one(step: OrchestratedStep):
            return await asyncio.to_thread(execute, step.command, step.params)

        return await asyncio.gather(*(_one(s) for s in steps))

    # 兼容无事件循环环境
    try:
        return asyncio.run(_run())
    except RuntimeError:
        # 已在事件循环中 → 用同步顺序兜底（本服务线程模型下一般走 asyncio.run）
        return [execute(s.command, s.params) for s in steps]


def execute_plan(
    steps: list[OrchestratedStep],
    execute: Callable[[str, dict], dict],
) -> dict:
    """执行编排后的步骤，返回合并结果。

    execute(command, params) -> dict（单命令执行结果，含 tool 名与 data）。

    Returns:
        {
          "all_success": bool,
          "results": [ {tool, success, data, error}, ... ],
          "failed": [tool, ...],
        }
    """
    results: list[dict] = []
    # 按 level 分组，逐层执行
    by_level: dict[int, list[OrchestratedStep]] = {}
    for s in steps:
        by_level.setdefault(s.level, []).append(s)

    for level in sorted(by_level):
        group = by_level[level]
        # 同层全只读 → 并发；否则串行
        all_read = all(not s.write for s in group)
        if all_read and len(group) > 1:
            group_results = _run_parallel_reads(group, execute)
        else:
            group_results = [execute(s.command, s.params) for s in group]
        results.extend(group_results)

    all_success = all(r.get("success") for r in results)
    failed = [r.get("tool") for r in results if not r.get("success")]
    return {
        "all_success": all_success,
        "results": results,
        "failed": failed,
    }


def build_conflict_prompt(pairs: list[tuple[str, str]]) -> str:
    """生成矛盾请求的澄清提示（拒绝执行，要求用户澄清）。"""
    desc = "、".join(f"「{a}」和「{b}」" for a, b in pairs)
    return (
        f"您的请求包含相互冲突的操作：{desc}。"
        "这些操作无法同时执行，请明确您希望进行哪一项，我将为您单独处理。"
    )
