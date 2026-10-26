"""load_skill 节点：渐进披露的「加载技能」环节。

当分类后 active_skill 为空（模型/规则未确定 skill），但意图是 knowledge/query/write
（需要业务处理），先经本节点加载对应 skill，把结果写回 active_skill。

skill 选择：竖切片阶段用确定性规则从文本识别领域（order/product/policy/vision）；
真实模型路径下由模型调用 load_skill 工具决定。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph.state import AgentState
from app.skills import SKILL_CATALOG, get_skill


def _detect_skill_code(text: str) -> str | None:
    """从文本识别应加载的 skill code（确定性兜底）。"""
    product_keywords = ["商品", "库存", "规格", "颜色", "尺码", "参数", "型号", "多少钱", "价格"]
    policy_keywords = ["政策", "规则", "时效", "运费", "无理由", "售后政策", "退款政策", "退货政策", "换货政策"]
    order_keywords = [
        "订单", "退款", "退货", "取消", "地址", "优惠券", "发货", "售后", "物流",
        "退", "换货", "快递", "到哪", "配送",
    ]
    if any(k in text for k in policy_keywords):
        return "policy"
    if any(k in text for k in product_keywords):
        return "product"
    if any(k in text for k in order_keywords):
        return "order"
    return None


def load_skill_node(state: AgentState) -> dict:
    """加载 skill：识别领域 → 写 active_skill → 返回 skill 说明。"""
    last = state["messages"][-1].content if state.get("messages") else ""
    code = _detect_skill_code(last)

    if code is None or get_skill(code) is None:
        # 无法识别领域 → 转人工
        return {
            "active_skill": None,
            "needs_human": True,
            "final_answer": "抱歉，我无法判断您的问题属于哪个领域，已为您转接人工客服。",
        }

    skill = get_skill(code)
    # 返回说明（供后续模型参考），并写入 active_skill
    notice = f"已加载「{skill.name}」技能，可为您处理相关问题。"
    return {
        "active_skill": code,
        "needs_human": False,
        "messages": [AIMessage(content=notice)],
    }
