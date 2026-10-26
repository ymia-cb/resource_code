"""Skill 目录 + 业务类型定义（definite） + 渐进披露提示词。

三层结构：
1. ``BusinessType``：业务类型枚举（query / write / knowledge / multimodal）。
2. ``TypeDefinition``（definite）：定义某业务类型的语义 + 该类型具体用到的工具 + 行为约束。
3. ``SKILL_CATALOG``：领域 skill（order/product/policy/vision），每个 skill 标注其业务类型。

「业务类型 → 工具」的映射由 definite 定义（TypeDefinition.tools），
用于：
- catalog 按业务类型归类展示（渐进披露索引）。
- 中间件差异化处理（如 write 类型走写操作控制流、tool 超时中间件按类型设超时）。
- skill 隔离校验（active_skill 的领域下，只开放该领域涉及的业务类型对应的工具）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BusinessType(str, Enum):
    """业务类型（分类目录 + 中间件差异化处理的维度）。"""

    QUERY = "query"  # 只读查询类（商品/订单查询）
    WRITE = "write"  # 写操作类（取消/退款/退货/改址，有副作用）
    KNOWLEDGE = "knowledge"  # 知识/政策类（退款政策、发货时效等 FAQ）
    MULTIMODAL = "multimodal"  # 多模态类（图像理解）


@dataclass
class TypeDefinition:
    """业务类型定义（definite）：定义某业务类型的语义 + 用到的工具 + 行为约束。"""

    type: BusinessType
    description: str  # 该业务类型的语义定义
    tools: list[str]  # 该类型具体用到的工具（白名单）
    # 行为约束（供中间件差异化处理）
    needs_confirm: bool = False  # 是否需二次确认
    read_only: bool = True  # 是否只读（无副作用）
    timeout_seconds: float = 10.0  # 工具调用超时


# ---------------------------------------------------------------------------
# 领域 skill 的补充工具（跨业务类型：写操作领域需前置只读查询）
# 键：skill code；值：该领域额外开放的工具（超出其业务类型 tools 的部分）
# ---------------------------------------------------------------------------

SKILL_EXTRA_TOOLS: dict[str, list[str]] = {
    # order 领域核心是写操作（WRITE），但写前需只读查询订单/物流/优惠券做前置
    "order": ["query_order", "query_logistics", "query_coupon"],
}


# ---------------------------------------------------------------------------
# 业务类型定义（definite）
# ---------------------------------------------------------------------------

TYPE_DEFINITIONS: dict[BusinessType, TypeDefinition] = {
    BusinessType.QUERY: TypeDefinition(
        type=BusinessType.QUERY,
        description="只读查询：查订单、查物流、查优惠券、查商品，无副作用",
        tools=["query_order", "query_logistics", "query_coupon", "query_product"],
        needs_confirm=False,
        read_only=True,
        timeout_seconds=10.0,
    ),
    BusinessType.WRITE: TypeDefinition(
        type=BusinessType.WRITE,
        description="写操作：取消订单、退款、退货、修改地址，有副作用",
        tools=["cancel_order", "apply_refund", "apply_return", "update_address"],
        needs_confirm=True,
        read_only=False,
        timeout_seconds=15.0,  # 写操作超时略长（事务 + 行锁）
    ),
    BusinessType.KNOWLEDGE: TypeDefinition(
        type=BusinessType.KNOWLEDGE,
        description="知识/政策：退款政策、退货政策、发货时效等平台规则，无需工具",
        tools=[],
        needs_confirm=False,
        read_only=True,
        timeout_seconds=10.0,
    ),
    BusinessType.MULTIMODAL: TypeDefinition(
        type=BusinessType.MULTIMODAL,
        description="多模态：商品图识别、截图诊断、票据解析",
        tools=["image_understand"],
        needs_confirm=False,
        read_only=True,
        timeout_seconds=20.0,  # VLM 识别较慢
    ),
}


@dataclass
class Skill:
    """一个领域技能（标注业务类型）。"""

    code: str  # 稳定标识（如 order / product / policy / vision）
    name: str  # 显示名
    type: BusinessType  # 业务类型（关联 TypeDefinition）
    description: str  # 简短描述（索引用）
    guidance: list[str]  # 完整处理规则（加载后注入提示词）


# ---------------------------------------------------------------------------
# Skill 目录（领域 skill，标注业务类型）
# ---------------------------------------------------------------------------

SKILL_CATALOG: dict[str, Skill] = {
    "order": Skill(
        code="order",
        name="订单售后",
        type=BusinessType.WRITE,  # order 领域核心是写操作（查询作为前置）
        description="查询订单、取消订单、退款、退货、修改地址等订单与售后处理",
        guidance=[
            "查询订单状态、金额、优惠券等，用 query_order / query_coupon",
            "取消订单仅待付款/待发货可取消，已发货走退货",
            "退款/退货需二次确认，金额超 ¥1000 转人工审批",
            "修改地址仅待发货可改",
            "写操作必须用户明确确认后才能执行",
        ],
    ),
    "product": Skill(
        code="product",
        name="商品咨询",
        type=BusinessType.QUERY,
        description="查询商品信息、库存、规格等商品相关问题",
        guidance=[
            "查询商品信息、库存、规格用 query_product",
            "商品相关问题只读，无写操作",
        ],
    ),
    "policy": Skill(
        code="policy",
        name="政策咨询",
        type=BusinessType.KNOWLEDGE,
        description="退款政策、退货政策、发货时效、运费等平台规则",
        guidance=[
            "回答平台规则类问题，可知识库直答",
            "不涉及订单操作，无需调用工具",
        ],
    ),
    "vision": Skill(
        code="vision",
        name="图像理解",
        type=BusinessType.MULTIMODAL,
        description="识别商品图片、诊断截图、解析票据",
        guidance=[
            "商品图片识别、截图诊断、票据解析用 image_understand",
            "低置信度（<0.7）结果转人工",
        ],
    ),
}


# ---------------------------------------------------------------------------
# 工具 → 业务类型 的归属（由 TypeDefinition.tools 反查）
# ---------------------------------------------------------------------------

_TOOL_TO_TYPE: dict[str, BusinessType] = {
    tool: td.type for td in TYPE_DEFINITIONS.values() for tool in td.tools
}


def tool_type(tool_name: str) -> BusinessType | None:
    """查工具所属业务类型。"""
    return _TOOL_TO_TYPE.get(tool_name)


def tools_of_type(business_type: BusinessType) -> list[str]:
    """查某业务类型具体用到的工具（definite 定义）。"""
    td = TYPE_DEFINITIONS.get(business_type)
    return list(td.tools) if td else []


# ---------------------------------------------------------------------------
# 目录 / 渐进披露
# ---------------------------------------------------------------------------

_TYPE_LABEL = {
    BusinessType.QUERY: "查询类",
    BusinessType.WRITE: "写操作类",
    BusinessType.KNOWLEDGE: "知识/政策类",
    BusinessType.MULTIMODAL: "多模态类",
}


def type_label(business_type: BusinessType) -> str:
    return _TYPE_LABEL.get(business_type, business_type.value)


def render_index() -> str:
    """渲染 skill 索引（按业务类型分组，未加载时注入提示词）。"""
    lines = []
    for bt in BusinessType:
        members = [s for s in SKILL_CATALOG.values() if s.type == bt]
        if not members:
            continue
        lines.append(f"【{type_label(bt)}】")
        for s in members:
            lines.append(f"- {s.code}：{s.description}")
    return "\n".join(lines)


def get_skill(code: str | None) -> Skill | None:
    """按 code 查 skill。"""
    if code is None:
        return None
    return SKILL_CATALOG.get(code)


def tools_for_skill(code: str | None) -> list[str]:
    """返回某 skill 涉及的全部工具（业务类型 tools + 领域补充工具）。

    领域 skill 可能涉及多个业务类型的工具（如 order 既查询又写）。
    业务类型定义的 tools 由 definite 给出；跨类型补充工具由 SKILL_EXTRA_TOOLS 声明
    （如 order 写前需 query_order/query_coupon 前置查询，但不含 query_product）。
    """
    skill = get_skill(code)
    if skill is None:
        return []
    tools = list(tools_of_type(skill.type))
    for t in SKILL_EXTRA_TOOLS.get(skill.code, []):
        if t not in tools:
            tools.append(t)
    return tools


def render_skill_prompt(active_skill: str | None) -> str:
    """根据当前 skill 生成 Skill 提示词段（渐进披露）。"""
    if active_skill is None:
        return (
            "【动态 Skill 路由】\n"
            "用户询问商品、订单、物流、售后、平台规则等业务问题时，先调用 load_skill 加载对应领域；"
            "普通寒暄可以直接回答。\n"
            "用户同时询问多个领域时，一次只处理一个 Skill；完成当前领域所需查询后，再切换到下一个 Skill。\n"
            f"可用 Skill（按业务类型）：\n{render_index()}"
        )
    skill = get_skill(active_skill)
    if skill is None:
        return render_skill_prompt(None)
    guidance = "\n".join(f"- {item}" for item in skill.guidance)
    return (
        f"【当前 Skill：{skill.name}（{skill.code}，类型：{type_label(skill.type)}）】\n"
        f"{guidance}\n"
        "用户同时询问多个领域时，一次只处理一个 Skill；完成当前领域所需查询后，再切换到下一个 Skill。"
    )


def render_support_prompt(active_skill: str | None) -> str:
    """生成完整的客服系统提示词（base + skill + 约束）。"""
    skill_prompt = render_skill_prompt(active_skill)
    base = (
        "你是电商智能客服。请准确、简洁地服务用户。\n"
        "涉及写操作（取消/退款/退货/改址）必须用户明确确认后执行。\n"
        "如用户请求包含相互矛盾的操作，请要求澄清，不要自行选择其中一种结果。"
    )
    return f"{base}\n\n{skill_prompt}"
