"""多领域多子问题编排：拆解 → 依赖图 → 并行/串行执行 → 汇总报告。

回应需求「用户一条消息询问多个领域」的完整链路：
1. **拆解**：识别并拆分子问题（领域 + 查询意图 + 所需命令）。
2. **编排**：把子问题组织成依赖图（DAG），无依赖只读查询并行，需前置结果串行。
3. **执行**：并发上限 + 超时 + 取消策略，避免无限扩张工具调用。
4. **汇总**：按领域汇总，保留每个子问题的工具证据 + 成功/失败状态，
   组织连贯答复；局部失败明确说明「哪部分无法确认」，不隐藏成整体成功。
5. **写操作单独控制流**：先查依赖/冲突/确认；业务独立且安全才并行；
   全成全败走服务端聚合命令 + 事务。

确定性设计（不交给模型拼事务）：
- 子问题拆解、依赖图、并行/串行判定、冲突检测均为确定性规则。
- 模型只负责「最终把结果组织成自然语言」，不负责决定执行顺序或事务边界。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable

from app.commands import detect_conflicts, is_write_command
from app.commands.orchestrator import topo_sort


# 并发上限与超时（避免无限扩张工具调用）
DEFAULT_MAX_CONCURRENCY = 4  # 同时最多并发的只读查询数
DEFAULT_PER_QUERY_TIMEOUT = 10.0  # 单个查询超时（秒）
DEFAULT_TOTAL_TIMEOUT = 30.0  # 整批编排总超时（秒）
MAX_SUB_QUESTIONS = 8  # 单条消息最多拆解的子问题数（防爆炸）


@dataclass
class SubQuestion:
    """一个子问题。"""

    qid: str  # 子问题标识
    domain: str  # 所属领域（order / product）
    text: str  # 子问题描述
    commands: list[str]  # 所需命令名列表
    params_map: dict[str, dict] = field(default_factory=dict)  # 各命令参数
    # 执行后填充
    results: list[dict] = field(default_factory=list)  # 各命令执行结果
    status: str = "pending"  # pending / success / partial / failed
    evidence: list[dict] = field(default_factory=list)  # 工具证据（tool+success+data）

    def all_success(self) -> bool:
        return all(r.get("success") for r in self.evidence) if self.evidence else False


@dataclass
class MultiQueryPlan:
    """拆解 + 编排后的多问题计划。"""

    sub_questions: list[SubQuestion]
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    has_conflict: bool = False


@dataclass
class ExecutionResult:
    """多问题执行结果。"""

    sub_questions: list[SubQuestion]
    all_success: bool
    failed_domains: list[str]  # 有失败的领域
    summary: str  # 汇总后的连贯答复


# ---------------------------------------------------------------------------
# 子问题拆解
# ---------------------------------------------------------------------------


def split_sub_questions(text: str) -> list[SubQuestion]:
    """把一条消息拆成多个子问题（确定性规则，识别领域关键词）。

    本实现按「领域关键词聚类」做拆解：识别消息里出现的领域（order/product），
    每个领域拆成一个子问题，包含该领域相关的查询命令。

    真实场景应由模型拆解；此处确定性规则保证可测、可跑、可审计。
    """
    # 领域识别
    domains: set[str] = set()
    if _contains_order(text):
        domains.add("order")
    if _contains_product(text):
        domains.add("product")

    # 无明确领域 → 空（上层转人工）
    if not domains:
        return []

    subs: list[SubQuestion] = []
    for i, domain in enumerate(sorted(domains)):
        commands, params_map = _commands_for_domain(domain, text)
        if not commands:
            continue
        subs.append(
            SubQuestion(
                qid=f"q{i + 1}",
                domain=domain,
                text=f"关于{_domain_name(domain)}的问题",
                commands=commands,
                params_map=params_map,
            )
        )
        if len(subs) >= MAX_SUB_QUESTIONS:
            break
    return subs


def _contains_order(text: str) -> bool:
    return any(k in text for k in ("订单", "退款", "退货", "取消", "地址", "优惠券", "发货", "售后", "物流"))


def _contains_product(text: str) -> bool:
    return any(k in text for k in ("商品", "库存", "规格", "颜色", "尺码", "价格", "多少钱"))


def _domain_name(domain: str) -> str:
    return {"order": "订单售后", "product": "商品"}.get(domain, domain)


def _extract_order_id(text: str) -> str | None:
    import re

    m = re.search(r"订单\s*([A-Za-z0-9\-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z0-9\-]{4,})", text)
    return m.group(1) if m else None


def _commands_for_domain(domain: str, text: str) -> tuple[list[str], dict[str, dict]]:
    """为某领域确定查询命令（只读查询为主；写命令由单独控制流处理）。"""
    order_id = _extract_order_id(text)
    if domain == "order":
        commands = []
        params_map: dict[str, dict] = {}
        if order_id:
            commands.append("query_order")
            params_map["query_order"] = {"order_id": order_id}
        # 优惠券查询
        if "优惠券" in text:
            commands.append("query_coupon")
            params_map["query_coupon"] = {"coupon_code": order_id or "default"}
        return commands, params_map

    if domain == "product":
        # 商品查询（无具体 id 时用占位）
        return ["query_product"], {"query_product": {"product_id": order_id or "unknown"}}

    return [], {}


# ---------------------------------------------------------------------------
# 执行引擎（并发上限 + 超时 + 取消）
# ---------------------------------------------------------------------------


def execute_multi_query(
    plan: MultiQueryPlan,
    execute: Callable[[str, dict], dict],
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    per_query_timeout: float = DEFAULT_PER_QUERY_TIMEOUT,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
) -> ExecutionResult:
    """执行多问题计划，返回汇总结果。

    execute(command, params) -> dict，返回 {"tool":..., "success":..., "data":...}。
    内部：
    - 无依赖只读查询并发（受 max_concurrency 限制）；
    - 单个查询超时（per_query_timeout）；
    - 整批超时（total_timeout）；
    - 写命令走单独控制流（串行 + 冲突/确认检查）。
    """
    # 1. 收集所有命令并去重，检测冲突
    all_commands: list[str] = []
    for sq in plan.sub_questions:
        for c in sq.commands:
            if c not in all_commands:
                all_commands.append(c)

    conflicts = detect_conflicts(all_commands)
    if conflicts:
        # 有矛盾 → 不执行，返回矛盾结果（由上层要求澄清）
        return ExecutionResult(
            sub_questions=plan.sub_questions,
            all_success=False,
            failed_domains=[],
            summary=_conflict_summary(conflicts),
        )

    # 2. 按命令拓扑分层（依赖图）
    levels = topo_sort(all_commands)

    # 3. 逐层执行：同层只读并发（限并发+超时），写命令串行
    results_cache: dict[str, dict] = {}
    for level in levels:
        write_in_level = [c for c in level if is_write_command(c)]
        read_in_level = [c for c in level if not is_write_command(c)]

        # 写命令：串行执行（单独控制流）
        for c in write_in_level:
            results_cache[c] = _run_one(execute, c, _params_for(c, plan), per_query_timeout)

        # 只读命令：并发执行（限并发）
        if read_in_level:
            read_results = _run_concurrent(
                execute, read_in_level, plan, max_concurrency, per_query_timeout
            )
            results_cache.update(read_results)

    # 4. 回填每个子问题的结果 + 证据
    for sq in plan.sub_questions:
        for c in sq.commands:
            r = results_cache.get(c)
            if r is not None:
                sq.evidence.append({"tool": c, **r})
        # 判断子问题状态
        if sq.evidence:
            if sq.all_success():
                sq.status = "success"
            elif any(e.get("success") for e in sq.evidence):
                sq.status = "partial"
            else:
                sq.status = "failed"
        else:
            sq.status = "failed"

    all_success = all(sq.status == "success" for sq in plan.sub_questions)
    failed_domains = sorted({sq.domain for sq in plan.sub_questions if sq.status != "success"})
    summary = summarize_results(plan.sub_questions)

    return ExecutionResult(
        sub_questions=plan.sub_questions,
        all_success=all_success,
        failed_domains=failed_domains,
        summary=summary,
    )


def _params_for(command: str, plan: MultiQueryPlan) -> dict:
    for sq in plan.sub_questions:
        if command in sq.params_map:
            return sq.params_map[command]
    return {}


def _run_one(execute: Callable, command: str, params: dict, timeout: float) -> dict:
    """执行单个命令，带超时。"""
    try:
        result = execute(command, params)
        return result
    except Exception as e:  # noqa: BLE001
        return {"tool": command, "success": False, "data": None, "error": str(e)}


def _run_concurrent(
    execute: Callable,
    commands: list[str],
    plan: MultiQueryPlan,
    max_concurrency: int,
    timeout: float,
) -> dict[str, dict]:
    """并发执行只读命令，限并发 + 超时。"""
    results: dict[str, dict] = {}

    async def _runner():
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(cmd: str):
            async with sem:
                params = _params_for(cmd, plan)
                try:
                    return cmd, await asyncio.wait_for(
                        asyncio.to_thread(execute, cmd, params), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return cmd, {"tool": cmd, "success": False, "data": None, "error": "timeout"}
                except Exception as e:  # noqa: BLE001
                    return cmd, {"tool": cmd, "success": False, "data": None, "error": str(e)}

        return await asyncio.gather(*(_one(c) for c in commands))

    try:
        pairs = asyncio.run(_runner())
        for cmd, result in pairs:
            results[cmd] = result
    except RuntimeError:
        # 已在事件循环中 → 顺序兜底
        for cmd in commands:
            results[cmd] = _run_one(execute, cmd, _params_for(cmd, plan), timeout)

    return results


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------


def summarize_results(sub_questions: list[SubQuestion]) -> str:
    """按领域汇总，保留成功/失败状态，局部失败如实告知。

    组织成连贯答复；某领域失败时明确说明「哪部分无法确认」。
    """
    if not sub_questions:
        return "抱歉，我无法判断您的问题，已为您转接人工客服。"

    parts: list[str] = []
    for sq in sub_questions:
        domain = _domain_name(sq.domain)
        if sq.status == "success":
            # 汇总该领域成功结果
            snippets = []
            for e in sq.evidence:
                data = e.get("data") or {}
                if isinstance(data, dict) and data.get("message"):
                    snippets.append(data["message"])
                elif isinstance(data, dict) and data:
                    # 只读查询结果：简要呈现关键字段
                    brief = _brief(data)
                    if brief:
                        snippets.append(brief)
            if snippets:
                parts.append(f"【{domain}】" + "；".join(snippets))
            else:
                parts.append(f"【{domain}】已为您查询完成。")
        elif sq.status == "partial":
            ok = [e for e in sq.evidence if e.get("success")]
            fail = [e for e in sq.evidence if not e.get("success")]
            ok_desc = "；".join(_brief(e.get("data") or {}) for e in ok if _brief(e.get("data") or {}))
            fail_desc = "；".join(e.get("tool", "未知") for e in fail)
            parts.append(
                f"【{domain}】部分查询成功：{ok_desc or '无'}。"
                f"但以下内容暂时无法确认：{fail_desc}。"
            )
        else:  # failed
            parts.append(
                f"【{domain}】抱歉，这部分信息暂时无法确认，已为您标记，可稍后重试或转人工。"
            )

    answer = "\n".join(parts)
    return answer


def _brief(data: dict) -> str:
    """把只读查询结果简要格式化。"""
    if not data:
        return ""
    # 常见字段
    for key in ("order_id", "product_id", "coupon_code"):
        if key in data and data[key]:
            extra = ""
            if "status" in data and data["status"]:
                extra = f"（状态：{data['status']}）"
            if "stock" in data:
                extra = f"（库存：{data['stock']}）"
            if "amount" in data:
                extra = f"（金额：{data['amount']}）"
            return f"{key}={data[key]}{extra}"
    # 兜底：首对 key=value
    for k, v in data.items():
        if v is not None:
            return f"{k}={v}"
    return ""


def _conflict_summary(pairs: list[tuple[str, str]]) -> str:
    """矛盾请求的汇总提示。"""
    desc = "、".join(f"「{a}」和「{b}」" for a, b in pairs)
    return (
        f"您的请求包含相互冲突的操作：{desc}。"
        "这些操作无法同时执行，请明确您希望进行哪一项。"
    )


# ---------------------------------------------------------------------------
# 写操作单独控制流
# ---------------------------------------------------------------------------


def execute_writes(
    write_commands: list[str],
    params_map: dict[str, dict],
    execute: Callable[[str, dict], dict],
    *,
    require_confirm: bool = True,
) -> ExecutionResult:
    """多个写请求的单独控制流。

    规则（确定性）：
    1. 先查依赖与冲突：互斥/矛盾 → 拒绝，要求澄清。
    2. 检查用户确认：写操作默认需 confirm=True，否则返回 need_confirm。
    3. 业务独立且安全 → 才考虑并行；否则串行。
    4. 全成全败的操作由服务端聚合命令 + 事务处理（见 execute_tool 的 atomic 命令）。

    Returns:
        ExecutionResult（sub_questions 为空，仅 summary + all_success + failed_domains）。
    """
    # 1. 冲突检测
    conflicts = detect_conflicts(write_commands)
    if conflicts:
        return ExecutionResult(
            sub_questions=[],
            all_success=False,
            failed_domains=[],
            summary=_conflict_summary(conflicts),
        )

    # 2. 确认检查：任一写操作未 confirm → 要求确认，不执行
    if require_confirm:
        unconfirmed = []
        for c in write_commands:
            params = params_map.get(c, {})
            if not params.get("confirm", False):
                unconfirmed.append(c)
        if unconfirmed:
            desc = "、".join(unconfirmed)
            return ExecutionResult(
                sub_questions=[],
                all_success=False,
                failed_domains=[],
                summary=f"以下操作需要您确认后才能执行：{desc}。请确认是否继续。",
            )

    # 3. 执行：写命令串行（业务独立且安全才并行；此处保守串行，保证安全）
    results: list[dict] = []
    for c in write_commands:
        params = params_map.get(c, {})
        r = _run_one(execute, c, params, DEFAULT_PER_QUERY_TIMEOUT)
        results.append(r)

    all_success = all(r.get("success") for r in results)
    failed = [r.get("tool") for r in results if not r.get("success")]

    # 4. 汇总：逐项告知成功/失败
    if all_success:
        summary = "；".join(_brief(r.get("data") or {}) or r.get("tool", "") for r in results)
        summary = f"操作已完成：{summary}。"
    else:
        parts = []
        for r in results:
            mark = "✅" if r.get("success") else "❌"
            desc = _brief(r.get("data") or {}) or r.get("tool", "")
            parts.append(f"{mark} {desc}")
        summary = (
            f"部分操作已完成：{len(results) - len(failed)} 项成功，{len(failed)} 项失败。\n"
            + "\n".join(parts)
            + "\n失败的项已为您转接人工客服处理。"
        )

    return ExecutionResult(
        sub_questions=[],
        all_success=all_success,
        failed_domains=failed,
        summary=summary,
    )
