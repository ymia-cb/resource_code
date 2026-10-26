"""Mock 模型 Provider。

当未配置真实 LLM key 时，用确定性规则兜底，保证竖切片在任何环境都能跑通。
真实 Provider 接入后，替换 llm() 的 provider 映射即可，业务代码不动。
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage

from app.llm.providers.base import Provider


def mock_classify(text: str) -> tuple[str, str | None, str | None, float]:
    """确定性分类兜底：细粒度五类意图 + skill + sub_intent。

    返回 (intent, skill, sub_intent, confidence)。
    - knowledge：政策/平台规则 → RAG + LLM
    - query：物流/订单状态一步查 → query 节点单步
    - multi_step：多步读写混合 + 顺序/条件逻辑 → planner 重推理
    - clarify：信息不足（有操作意图但缺关键参数）→ 反问澄清
    - human：无法判断 / 矛盾请求 → 转人工

    真实场景应换成小模型判断，此处规则仅保证链路可跑、可测。
    """
    # 知识库 / 政策类关键词（RAG + LLM 直答）
    knowledge_keywords = [
        "退款政策", "退货政策", "换货政策", "售后政策", "优惠券规则", "优惠券使用",
        "发货时效", "运费", "运费险", "无理由", "偏远", "秒杀", "定制", "生鲜",
        "怎么退", "怎么退款", "怎么退货", "能不能退", "可以退吗", "支持",
        "规则", "政策", "规定", "时效", "多久", "几天", "保证", "保修",
    ]
    # 一步查询类关键词（物流/订单状态，单次工具调用即可）
    query_keywords = [
        "物流", "到哪", "快递", "单号", "配送", "到货", "签收",
        "订单状态", "查订单", "我的订单", "订单详情", "什么时候发货", "发货了吗",
    ]
    # 多步操作关键词（写操作 / 条件分支，需要 planner 重推理规划）
    multi_step_keywords = [
        "我要退这个订单", "帮我取消", "取消订单", "退掉", "我要退款", "我要退货",
        "退款订单", "退货订单", "申请退款", "申请退货", "我要改地址", "改地址",
        "修改地址",
    ]
    # 澄清关键词（有操作意图但缺关键参数）
    clarify_keywords = ["退款", "退货", "取消", "改地址", "换货"]

    # 1. 条件分支 / 多步读写（最具体，优先匹配）：「如果…就…」「没发货就…，发货了就…」
    if _has_conditional_logic(text):
        return ("multi_step", "order", None, 0.9)

    # 2. 多步写操作
    if any(k in text for k in multi_step_keywords):
        sub = _detect_sub_intent(text)
        return ("multi_step", "order", sub, 0.9)

    # 3. 一步查询
    if any(k in text for k in query_keywords):
        return ("query", "order", None, 0.9)

    # 4. 知识库问答
    if any(k in text for k in knowledge_keywords):
        return ("knowledge", None, None, 0.9)

    # 5. 澄清：有「退款/退货/取消/改址」等动作词但没带订单号 → 反问
    if any(k in text for k in clarify_keywords):
        return ("clarify", "order", None, 0.9)

    # 6. 兜底转人工
    return ("human", None, None, 0.3)


def _has_conditional_logic(text: str) -> bool:
    """判断是否包含「如果…就…」「没发货就…，发货了就…」等条件分支。"""
    has_if = any(k in text for k in ["如果", "若", "要是", "假如"])
    has_then = any(k in text for k in ["就", "否则", "不然", "那么"])
    # 「没发货就取消」「发货了就改地址」这类状态条件
    status_conditions = ["没发货", "未发货", "还没发货", "待发货", "发货了", "已发货", "已经发货"]
    if has_if and has_then:
        return True
    return any(k in text for k in status_conditions) and has_then


def _detect_sub_intent(text: str) -> str | None:
    """从文本识别写操作的细分命令（供 planner 精准拆解）。"""
    if "取消" in text or "退掉" in text:
        return "cancel_order"
    if "退货" in text or "换货" in text:
        return "apply_return"
    if "退款" in text:
        return "apply_refund"
    if "地址" in text or "改址" in text:
        return "update_address"
    return None


def mock_generate(messages: list[BaseMessage], sources: list[dict]) -> str:
    """基于检索结果拼答案的兜底生成。

    取最相关的一条（retrieve 已按相关性排序）。真实场景由小模型基于
    多条 retrieved 综合生成，此处仅保证竖切片可跑、可测。
    """
    if not sources:
        return "抱歉，我暂时无法回答这个问题，已为您转接人工客服。"
    top = sources[0]
    return f"根据知识库：{top['text']}"


def mock_vlm(messages: list[BaseMessage]) -> str:
    """VLM 兜底：确定性返回一个结构化 JSON（含 confidence）。

    真实场景接 Qwen-VL / Gemini 等视觉模型。此处根据提示词里的 type 返回对应结构。
    """
    import json

    last = messages[-1].content if messages else ""
    # 从提示词文本判断图片类型
    text = json.dumps(last, ensure_ascii=False) if not isinstance(last, str) else last

    if "screenshot" in text:
        return json.dumps({
            "type": "screenshot",
            "error_code": "E5001",
            "issue": "支付超时",
            "suggestion": "重试支付",
            "confidence": 0.9,
        }, ensure_ascii=False)
    if "receipt" in text:
        return json.dumps({
            "type": "receipt",
            "order_id": "20261021001",
            "amount": 199.0,
            "date": "2026-10-21",
            "confidence": 0.95,
        }, ensure_ascii=False)
    # product 兜底
    return json.dumps({
        "type": "product",
        "category": "鞋子",
        "attributes": {"color": "红", "size": "42"},
        "confidence": 0.92,
    }, ensure_ascii=False)


class MockProvider(Provider):
    """统一 provider 接口，chat() 与真实 provider 对齐。"""

    def chat(self, messages: list[BaseMessage]) -> str:
        last = messages[-1].content if messages else ""
        return str(last)
