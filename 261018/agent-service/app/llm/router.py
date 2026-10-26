"""模型路由：所有 LLM 调用的唯一入口。

按 role 选择模型：
- classify / rag / reflector → 小模型（SMALL_LLM_*）
- planner / summarize → 大模型（LARGE_LLM_*）
- 未配置对应 api_key/model → 降级 mock（保证离线可跑）

降级策略（见 app/llm/degradation.py）：
- 所有降级路径统一记录原因（未配置/调用失败/熔断）+ 供应商 + 模型 + 角色，
  写结构化日志 + 累加指标，绝不静默降级。
- 可安全降级的 role（classify/rag/reflector）降级到 mock；不可安全降级的
  role（planner/summarize/vlm）降级会丢失关键能力，由 llm_with_status 暴露
  降级标记，供上层决定转人工。

业务节点禁止直连模型，一律走 llm() / classify() / llm_with_status()。
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage

from app.config import get_llm_settings
from app.llm.providers.mock import mock_classify, mock_vlm
from app.llm.providers.openai_compat import OpenAICompatProvider, ProviderError
from app.observability import ObservedError
from app.llm.degradation import DegradeReason, record_degradation, should_degrade_safely


def _build_provider(role: str):
    """按 role 构建 provider；未配置则返回 None（降级 mock）。"""
    settings = get_llm_settings()
    cfg = settings.provider_for(role)
    if cfg is None:
        return None
    return OpenAICompatProvider(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout=cfg.timeout,
        connect_timeout=cfg.connect_timeout,
        max_retries=cfg.max_retries,
        temperature=cfg.temperature,
    )


def classify(text: str) -> tuple[str, str | None, str | None, float]:
    """意图分类 + 技能匹配（细粒度五类 + 写操作细分）。

    返回 (intent, skill, sub_intent, confidence)。
    真实小模型配置后走模型分类（需解析 JSON 输出）；
    未配置走 mock 确定性规则。
    """
    provider = _build_provider("classify")
    if provider is None:
        record_degradation("classify", DegradeReason.NOT_CONFIGURED, detail="未配置 SMALL_LLM_API_KEY/MODEL")
        return mock_classify(text)

    # 真实模型分类：让模型输出 intent/skill/sub_intent/confidence 的 JSON
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        "你是电商客服的意图分类器。请把用户消息分类为以下之一：\n"
        "- knowledge：政策/平台规则等知识库问答（退款政策、运费规则、发货时效等），走知识库直答\n"
        "- query：物流/订单状态等一步查询（查物流、查订单状态），单次查询即可，无需多步推理\n"
        "- multi_step：多步读写混合且含顺序/条件逻辑（取消、退款、退货、改地址，或「如果没发货就取消、发货了就改地址」这类条件分支），需要规划多步执行\n"
        "- clarify：信息不足需澄清（有操作意图但缺关键信息，如没给订单号），反问用户补充\n"
        "- human：无法判断或需要人工\n"
        "【重要约束】\n"
        "1. 如果用户请求包含相互矛盾的操作（如「取消订单并继续发货」「退款又退货」），"
        "必须分类为 human，不要自行选择其中一种结果。\n"
        "2. 多步操作（multi_step）时，sub_intent 填细分命令：cancel_order / apply_refund / apply_return / update_address；"
        "条件分支类请求 sub_intent 填 null。\n"
        "只输出 JSON：{\"intent\": \"knowledge|query|multi_step|clarify|human\", "
        "\"skill\": \"order|product|null\", \"sub_intent\": \"cancel_order|apply_refund|apply_return|update_address|null\", "
        "\"confidence\": 0.0-1.0}"
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text),
    ]
    try:
        raw = provider.chat(messages)
        import json

        data = json.loads(raw)
        return (
            data.get("intent", "human"),
            data.get("skill"),
            data.get("sub_intent"),
            float(data.get("confidence", 0.0)),
        )
    except (ProviderError, json.JSONDecodeError, ValueError) as e:
        # 模型失败/解析失败 → 降级 mock
        _log_provider_error(provider, e)
        record_degradation(
            "classify",
            DegradeReason.PROVIDER_FAILED,
            provider=provider.provider_name,
            model=provider.model,
            detail=str(e),
        )
        return mock_classify(text)


def llm(role: str, messages: list[BaseMessage]) -> str:
    """统一模型调用入口。按 role 选 provider，失败降级 mock。

    role 取值：classify / rag / reflector / planner / summarize / vlm
    """
    text, _degraded = llm_with_status(role, messages)
    return text


def llm_with_status(role: str, messages: list[BaseMessage], active_skill: str | None = None) -> tuple[str, bool]:
    """带降级标记的统一模型调用入口。

    Returns:
        (text, degraded)：degraded=True 表示本次返回的是降级结果（非真实模型输出），
        上层可据此决定「转人工」而非直接采用降级答案（尤其 planner/summarize/vlm）。
    """
    # 中间件：模型调用前（skill 渐进披露提示词注入）
    from app.middleware import get_default_chain, LLMContext

    chain = get_default_chain()
    ctx = chain.before_llm(LLMContext(role=role, messages=list(messages), active_skill=active_skill))
    messages = ctx.messages

    provider = _build_provider(role)
    if provider is None:
        # 未配置 → 降级。记录降级原因，消除静默降级。
        record_degradation(role, DegradeReason.NOT_CONFIGURED, detail=f"未配置 {role} 对应模型 key/model")
        return _mock_fallback(role, messages), True

    try:
        result = provider.chat(messages)
        result = chain.after_llm(ctx, result)
        return result, False
    except ProviderError as e:
        # 模型失败 → 记结构化日志（底层原因/异常类型/状态码只进日志）→ 降级 mock
        _log_provider_error(provider, e)
        record_degradation(
            role,
            DegradeReason.PROVIDER_FAILED,
            provider=provider.provider_name,
            model=provider.model,
            retries=e.retries,
            detail=str(e),
        )
        return _mock_fallback(role, messages), True


def _mock_fallback(role: str, messages: list[BaseMessage]) -> str:
    """按 role 返回降级兜底。"""
    if role == "vlm":
        return mock_vlm(messages)
    last = messages[-1].content if messages else ""
    return str(last)


def _log_provider_error(provider: OpenAICompatProvider, e: ProviderError) -> None:
    """记录 ProviderError 到结构化日志（底层原因/异常类型/状态码只进日志）。"""
    ObservedError(
        category=e.category,
        code=e.code,
        provider=provider.provider_name,
        model=provider.model,
        retries=e.retries,
        detail=str(e),
        exception_type=e.exception_type,
        status_code=e.status_code,
    ).log()
