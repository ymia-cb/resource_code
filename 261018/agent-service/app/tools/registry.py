"""工具注册表 + 统一执行框架 execute_tool（五道安全闸门）。

Spec 4.2：所有工具调用强制经过 execute_tool，禁止模型/节点直接调工具函数。
五道闸门（Spec 4.3）：
  ① 白名单校验  ② 参数校验(pydantic)  ③ 权限校验(thread→user)
  ④ 幂等校验    ⑤ 审计落库（与执行同事务）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from pydantic import BaseModel, ValidationError

from app.db.models.audit import AuditLog
from app.db.models.order import Order
from app.db.session import get_session
from app.tools import order as order_tools
from app.tools import vision as vision_tools
from app.tools import skill as skill_tools
from app.tools.errors import (
    ToolNotAllowed,
    ParamInvalid,
    PermissionDenied,
    DuplicateRequest,
    NeedApproval,
)
from app.tools.idempotency import check_idempotency, record_idempotency
from app.tools.permission import resolve_user
from app.tools.schemas import (
    QueryOrderInput,
    QueryLogisticsInput,
    CancelOrderInput,
    QueryCouponInput,
    QueryProductInput,
    ApplyRefundInput,
    ApplyReturnInput,
    UpdateAddressInput,
    ImageUnderstandInput,
    LoadSkillInput,
)


@dataclass
class ToolSpec:
    """工具定义（Spec 4.4）。"""

    name: str
    description: str
    fn: Callable
    schema: type[BaseModel]
    risk: str  # low / mid / high
    idempotent: bool  # 是否天然幂等
    needs_confirm: bool  # 是否需二次确认
    needs_approval: bool = False  # 是否超阈值转人工
    threshold: float | None = None  # 金额阈值（needs_approval 时生效）
    skill: str | None = None  # 归属 skill（未加载该 skill 时不可调）
    idempotent_keys: tuple[str, ...] = ("order_id",)  # 参与幂等键的业务字段


# ---------------------------------------------------------------------------
# 工具白名单
# ---------------------------------------------------------------------------

TOOL_WHITELIST: dict[str, ToolSpec] = {
    "load_skill": ToolSpec(
        name="load_skill",
        description="加载领域 Skill（order=订单售后 / product=商品咨询），返回该 Skill 的处理规则",
        fn=skill_tools.load_skill,
        schema=LoadSkillInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill=None,  # 无归属：任何 skill 状态下都可用（渐进披露入口）
    ),
    "query_order": ToolSpec(
        name="query_order",
        description="查订单状态/详情",
        fn=order_tools.query_order,
        schema=QueryOrderInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill="order",
    ),
    "query_logistics": ToolSpec(
        name="query_logistics",
        description="查订单物流状态/轨迹",
        fn=order_tools.query_logistics,
        schema=QueryLogisticsInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill="order",
    ),
    "query_coupon": ToolSpec(
        name="query_coupon",
        description="查优惠券/活动规则",
        fn=lambda **kw: {"coupon_code": kw.get("coupon_code"), "status": "valid"},
        schema=QueryCouponInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill="order",
        idempotent_keys=("coupon_code",),
    ),
    "query_product": ToolSpec(
        name="query_product",
        description="查商品信息/库存",
        fn=lambda **kw: {"product_id": kw.get("product_id"), "stock": 100},
        schema=QueryProductInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill="product",
        idempotent_keys=("product_id",),
    ),
    "cancel_order": ToolSpec(
        name="cancel_order",
        description="取消订单（需二次确认）",
        fn=order_tools.cancel_order,
        schema=CancelOrderInput,
        risk="mid",
        idempotent=False,
        needs_confirm=True,
        skill="order",
        idempotent_keys=("order_id",),
    ),
    "update_address": ToolSpec(
        name="update_address",
        description="修改收货地址（需二次确认，仅待发货可改）",
        fn=order_tools.update_address,
        schema=UpdateAddressInput,
        risk="mid",
        idempotent=False,
        needs_confirm=True,
        skill="order",
        idempotent_keys=("order_id", "address"),
    ),
    "apply_refund": ToolSpec(
        name="apply_refund",
        description="发起退款（需二次确认，金额超阈值转人工）",
        fn=order_tools.apply_refund,
        schema=ApplyRefundInput,
        risk="high",
        idempotent=False,
        needs_confirm=True,
        needs_approval=True,
        threshold=float(order_tools.REFUND_APPROVAL_THRESHOLD),
        skill="order",
        idempotent_keys=("order_id",),
    ),
    "apply_return": ToolSpec(
        name="apply_return",
        description="申请退货/换货（需二次确认，金额超阈值转人工）",
        fn=order_tools.apply_return,
        schema=ApplyReturnInput,
        risk="high",
        idempotent=False,
        needs_confirm=True,
        needs_approval=True,
        threshold=float(order_tools.REFUND_APPROVAL_THRESHOLD),
        skill="order",
        idempotent_keys=("order_id",),
    ),
    "image_understand": ToolSpec(
        name="image_understand",
        description="VLM 图像理解：识别商品/诊断截图/解析票据（无副作用）",
        fn=vision_tools.image_understand,
        schema=ImageUnderstandInput,
        risk="low",
        idempotent=True,
        needs_confirm=False,
        skill=None,  # 视觉工具不限 skill（跨领域通用）
    ),
}


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


def _resolve_order_amount(validated: BaseModel, user_id: str) -> float | None:
    """从订单表解析金额（用于高风险工具阈值审批）。

    入参含 order_id 时查订单金额；无则返回 None（不做阈值判断）。
    """
    from sqlalchemy import select as _select

    order_id = getattr(validated, "order_id", None)
    if order_id is None:
        return None
    with get_session() as session:
        order = session.execute(
            _select(Order.amount).where(
                Order.order_id == order_id, Order.user_id == user_id
            )
        ).scalar_one_or_none()
    return float(order) if order is not None else None


def _audit_log(
    thread_id: str,
    tool_name: str,
    params: dict,
    result: dict,
    before: dict | None,
    after: dict | None,
) -> None:
    """闸门⑤：审计落库（与执行同 session）。"""
    with get_session() as session:
        session.add(
            AuditLog(
                thread_id=thread_id,
                tool_name=tool_name,
                action=tool_name,
                params=params,
                result=result,
                before_value=before,
                after_value=after,
            )
        )


# ---------------------------------------------------------------------------
# 统一执行入口
# ---------------------------------------------------------------------------


def execute_tool(name: str, params: dict, thread_id: str, active_skill: str | None = None) -> dict:
    """统一工具入口，内置安全闸门（含 skill 级动态白名单）。

    Returns:
        {"status": "ok" | "need_confirm" | "pending_human", "data": ...,
         "idempotent_replay": bool}
        - idempotent_replay=True 表示命中幂等，返回的是已有结果（未重新执行）。
    """
    # 闸门①：白名单
    spec = TOOL_WHITELIST.get(name)
    if spec is None:
        raise ToolNotAllowed(f"工具不在白名单: {name}")

    # 闸门①-2：skill 级动态白名单（工具隔离）
    # - 工具无 skill 归属（skill=None）→ 跨领域通用，始终放行。
    # - 工具声明 skill=X：
    #   * active_skill 为 None（未加载 skill / 兼容旧路径）→ 放行；
    #   * active_skill == X → 放行；
    #   * active_skill != X → 拒绝（跨 skill 调用）。
    if spec.skill is not None and active_skill is not None and spec.skill != active_skill:
        raise ToolNotAllowed(
            f"工具 {name} 不属于当前 skill（active={active_skill}, required={spec.skill}）"
        )

    # 闸门②：参数校验（pydantic）
    try:
        validated = spec.schema(**params)
    except ValidationError as e:
        raise ParamInvalid(f"参数校验失败: {e.errors()}")

    # 闸门③：权限校验（thread → user）
    user_id = resolve_user(thread_id)

    # 闸门④：幂等校验（写操作）——命中则正常返回已有结果，不再抛异常
    if not spec.idempotent:
        existing = check_idempotency(name, thread_id, params, spec.idempotent_keys)
        if existing is not None:
            # 幂等命中：返回已有结果（复用，不重复执行）
            return {"status": "ok", "data": existing, "idempotent_replay": True}

    # 二次确认（写操作）
    if spec.needs_confirm and not getattr(validated, "confirm", False):
        # 未确认 → 不执行，返回提示（不抛异常，交由上层再次请求）
        return {"status": "need_confirm", "data": {"message": "写操作需二次确认"}}

    # 高风险阈值审批：从订单查询金额（入参无 amount 时）
    if spec.needs_approval and spec.threshold is not None:
        order_amount = _resolve_order_amount(validated, user_id)
        if order_amount is not None and order_amount > spec.threshold:
            raise NeedApproval(f"金额 {order_amount} 超阈值 {spec.threshold}，转人工审批")

    # 执行（注入 user_id 强制租户隔离；写操作传入 thread_id 供业务内审计）
    validated_dict = validated.model_dump()
    validated_dict["user_id"] = user_id
    validated_dict.pop("confirm", None)  # confirm 仅作闸门标记，不传给业务函数
    if not spec.idempotent:
        validated_dict["thread_id"] = thread_id

    before: dict | None = None
    if not spec.idempotent:
        # 写操作：记录改前值
        before = {"params": params}

    result = spec.fn(**validated_dict)

    # 统一为 dict（cancel_order 返回 CancelOrderOutput dataclass）
    if hasattr(result, "__dataclass_fields__"):
        result_dict = {
            "success": result.success,
            "message": result.message,
            "refund_amount": float(result.refund_amount)
            if result.refund_amount is not None
            else None,
            "need_refund": result.need_refund,
            "restock": result.restock,
        }
    elif isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {"result": result}

    # 闸门⑤：审计落库（与执行结果关联）
    _audit_log(thread_id, name, params, result_dict, before, None)

    # 写操作：记录幂等结果
    if not spec.idempotent:
        # 注入 order_id 到结果（供事实校验 ground truth，写工具返回值未必带）
        if "order_id" not in result_dict and params.get("order_id"):
            result_dict["order_id"] = params["order_id"]
        record_idempotency(name, thread_id, params, result_dict, spec.idempotent_keys)

    return {"status": "ok", "data": result_dict, "idempotent_replay": False}
