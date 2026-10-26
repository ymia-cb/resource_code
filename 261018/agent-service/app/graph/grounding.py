"""事实校验（Grounding）：用工具成功结果校验回答中的业务事实。

核心原则（防幻觉）：
- 大模型回答中出现的业务实体（订单号/金额/状态/优惠券码等），必须**来自工具成功
  返回的结果**（或知识库溯源），不允许凭空捏造。
- 用 JSON Schema 定义实体结构，提示词约束大模型输出「回答 + 实体 + 引用」的结构化
  格式；生成后提取实体，逐一核对是否在「已核验事实源」中，不在 → 判定幻觉。

设计：校验层独立于生成层，方便对大模型输出做确定性校验（不依赖模型自觉）。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 业务实体 JSON Schema
# ---------------------------------------------------------------------------


class GroundedEntity(BaseModel):
    """回答中出现的业务实体（供事实校验）。

    type 枚举限定了允许出现的实体类型，value 是具体取值。
    只有出现在「工具成功结果」或「知识库溯源」里的实体才允许进入回答。
    """

    type: Literal["order_id", "amount", "status", "coupon_code", "refund_amount", "product"]
    value: str | float


class GroundedAnswer(BaseModel):
    """大模型输出的结构化回答（提示词约束的格式）。

    answer：面向用户的自然语言
    entities：回答中出现的业务实体列表（供事实校验）
    """

    answer: str
    entities: list[GroundedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 实体提取（从回答文本/JSON 提取业务实体）
# ---------------------------------------------------------------------------


def extract_entities(raw: str | dict | GroundedAnswer) -> list[GroundedEntity]:
    """从大模型输出提取业务实体。

    支持：
    - 已经是 GroundedAnswer / dict（含 answer + entities）→ 直接取 entities
    - 纯文本回答 → 用正则提取订单号/金额等常见实体（兜底）
    """
    import re

    if isinstance(raw, GroundedAnswer):
        return list(raw.entities)
    if isinstance(raw, dict):
        entities = raw.get("entities") or []
        return [GroundedEntity(**e) if isinstance(e, dict) else e for e in entities]

    text = str(raw)
    entities: list[GroundedEntity] = []

    # 订单号：常见「订单号 1234」「订单 1234」「订单号：ABC-123」
    for m in re.finditer(r"订单号?\s*[:：]?\s*([A-Za-z0-9\-]{4,})", text):
        entities.append(GroundedEntity(type="order_id", value=m.group(1)))

    # 金额：¥100.00 / 100.00 元 / 100元
    for m in re.finditer(r"[¥￥]\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*元", text):
        amount = m.group(1) or m.group(2)
        entities.append(GroundedEntity(type="amount", value=float(amount)))

    # 优惠券码：券码 XXX / 优惠券 XXX
    for m in re.finditer(r"(?:券码|优惠券)\s*[:：]?\s*([A-Za-z0-9\-]+)", text):
        entities.append(GroundedEntity(type="coupon_code", value=m.group(1)))

    return entities


# ---------------------------------------------------------------------------
# 事实源构建（从工具成功结果提取「已核验事实」）
# ---------------------------------------------------------------------------


def build_fact_source(scratchpad: list[dict], sources: list[dict] | None = None) -> set[tuple[str, str]]:
    """从工具成功结果 + 知识库溯源构建「已核验事实源」。

    返回 {(type, value_str), ...}，供 verify_entities 核对。
    只收「成功」的写操作结果与查询结果（success=True 或 status=ok），
    失败/转人工的结果不作为事实源。
    """
    facts: set[tuple[str, str]] = set()

    for record in scratchpad:
        result = record.get("result") or {}
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            continue
        # 只收成功结果
        if data.get("success") is False:
            continue
        if result.get("status") not in ("ok", "pending_human"):
            # need_approval 等不带可靠事实
            if result.get("status") != "ok":
                continue

        # 提取已知字段
        if data.get("order_id"):
            facts.add(("order_id", str(data["order_id"])))
        if data.get("refund_amount") is not None:
            facts.add(("refund_amount", str(data["refund_amount"])))
        if data.get("amount") is not None:
            facts.add(("amount", str(data["amount"])))
        if data.get("status"):
            facts.add(("status", str(data["status"])))

    # 知识库溯源里的 chunk_id 也作为事实源（允许引用）
    for s in sources or []:
        if s.get("chunk_id"):
            facts.add(("chunk_id", str(s["chunk_id"])))

    return facts


# ---------------------------------------------------------------------------
# 事实校验
# ---------------------------------------------------------------------------


def verify_entities(
    entities: list[GroundedEntity],
    fact_source: set[tuple[str, str]],
) -> tuple[list[GroundedEntity], list[GroundedEntity]]:
    """校验实体是否在事实源中。

    Returns:
        (valid_entities, hallucinated_entities)
        hallucinated = 出现在回答中但不在工具成功结果/知识库溯源里的实体（疑似幻觉）。
    """
    valid: list[GroundedEntity] = []
    hallucinated: list[GroundedEntity] = []

    for e in entities:
        key = (e.type, str(e.value))
        if key in fact_source:
            valid.append(e)
        else:
            hallucinated.append(e)

    return valid, hallucinated


def verify_grounding(
    declared_entities: list[GroundedEntity],
    answer_text: str,
    fact_source: set[tuple[str, str]],
) -> list[GroundedEntity]:
    """统一分层事实校验入口（主通道 + 兜底）。

    主通道：模型按 JSON Schema 自报的 entities → 与事实源核对。
    兜底：仅当 entities 为空（模型未自报/未按格式输出）时，才从 answer 文本
    正则提取实体做核对——抓「自报干净但正文夹带幻觉」的漏网之鱼。

    Returns:
        hallucinated_entities（疑似幻觉的实体列表，非空 → 上层转人工）。
    """
    if declared_entities:
        # 主通道：模型已自报实体，直接核对
        _valid, hallucinated = verify_entities(declared_entities, fact_source)
        return hallucinated

    # 兜底：entities 为空 → 从正文提取实体核对
    text_entities = extract_entities(answer_text)
    _valid, hallucinated = verify_entities(text_entities, fact_source)
    return hallucinated


# ---------------------------------------------------------------------------
# 提示词约束（让大模型输出结构化格式）
# ---------------------------------------------------------------------------


def build_grounding_prompt(task_hint: str, facts: set[tuple[str, str]]) -> str:
    """构造提示词，约束大模型输出「回答 + 实体」结构化格式。

    关键约束：
    - 只输出 JSON（answer + entities）
    - entities 里的业务实体必须来自给定事实源，不允许凭空捏造
    """
    fact_list = "\n".join(f"- {t}: {v}" for t, v in sorted(facts))
    return (
        f"{task_hint}\n\n"
        "请严格按以下 JSON 格式输出，不要输出任何其他内容：\n"
        '{"answer": "<面向用户的自然语言回答>", "entities": [{"type": "<实体类型>", "value": "<取值>"}]}\n\n'
        "约束：\n"
        "1. entities 里只能包含「业务实体」（order_id/amount/status/coupon_code/refund_amount），"
        "不要放无关词。\n"
        "2. 回答中出现的任何订单号、金额、优惠券码、状态，必须来自下面给出的「已核验事实」，"
        "严禁凭空编造。\n"
        "3. 若某个事实无法从下面确认，不要在回答中提及该具体数值，改为模糊表述或提示转人工。\n\n"
        "已核验事实（只能引用这些）：\n"
        f"{fact_list if fact_list else '（无）'}"
    )
