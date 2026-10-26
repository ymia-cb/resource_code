"""订单工具：查询 + 可取消性规则引擎 + 取消/退款/退货/改址执行。

核心设计（Spec 11.3）：可取消性等判断用**确定性规则**，不交给模型。
写操作（cancel/refund/return/update_address）经 execute_tool 五道闸门统一执行，
业务函数只专注状态变更，审计/幂等由闸门层负责。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select

from app.db.models.order import Order, OrderStatus, RefundStatus, ReturnStatus
from app.db.session import get_session


# ---------------------------------------------------------------------------
# 可取消性规则引擎
# ---------------------------------------------------------------------------

# 规则表：订单状态 -> (是否可取消, 处理建议)
_CANCEL_RULES: dict[str, tuple[bool, str]] = {
    OrderStatus.PENDING_PAYMENT.value: (True, "可取消"),
    OrderStatus.PENDING_SHIP.value: (True, "可取消，将退款"),
    OrderStatus.SHIPPED.value: (False, "已发货，请走退货流程"),
    OrderStatus.COMPLETED.value: (False, "已完成，请走售后流程"),
    OrderStatus.CANCELLED.value: (False, "订单已取消"),
}


def check_cancelable(status: str) -> tuple[bool, str]:
    """确定性规则：可取消性判断，不交给模型。

    Args:
        status: 订单状态字符串。

    Returns:
        (是否可取消, 处理建议)。未知状态返回不可取消并提示转人工。
    """
    return _CANCEL_RULES.get(status, (False, "未知状态，转人工"))


# ---------------------------------------------------------------------------
# 输出模型
# ---------------------------------------------------------------------------


@dataclass
class CancelOrderOutput:
    success: bool
    refund_amount: Decimal | None  # 退款金额（待付款为 None）
    need_refund: bool  # 是否涉及退款
    restock: bool  # 是否需库存回补
    message: str


# ---------------------------------------------------------------------------
# 订单查询 / 取消执行
# ---------------------------------------------------------------------------


def query_order(order_id: str, user_id: str | None = None) -> dict | None:
    """查询订单状态与详情（只读）。"""
    with get_session() as session:
        stmt = select(Order).where(Order.order_id == order_id)
        if user_id:
            stmt = stmt.where(Order.user_id == user_id)
        order = session.execute(stmt).scalar_one_or_none()
        if order is None:
            return None
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "status": order.status,
            "amount": float(order.amount),
            "paid": order.paid,
            "stock_deducted": order.stock_deducted,
            "coupon_code": order.coupon_code,
        }


# 订单状态 → 物流状态摘要（M1 简化：无独立物流轨迹表，按订单状态推导）
_LOGISTICS_STATUS_MAP = {
    "pending_payment": {"state": "not_shipped", "desc": "订单待付款，尚未发货"},
    "pending_ship": {"state": "not_shipped", "desc": "订单已付款，仓库备货中，尚未发货"},
    "shipped": {"state": "in_transit", "desc": "商品已发货，运输中"},
    "completed": {"state": "delivered", "desc": "商品已送达/签收"},
    "cancelled": {"state": "cancelled", "desc": "订单已取消，无物流信息"},
    "refunding": {"state": "refunding", "desc": "订单退款中，物流已暂停"},
    "returning": {"state": "returning", "desc": "退货中，商品已寄回"},
}


def query_logistics(order_id: str, user_id: str | None = None) -> dict | None:
    """查询订单物流状态（只读，M1 简化：按订单状态推导物流摘要）。"""
    with get_session() as session:
        stmt = select(Order).where(Order.order_id == order_id)
        if user_id:
            stmt = stmt.where(Order.user_id == user_id)
        order = session.execute(stmt).scalar_one_or_none()
        if order is None:
            return None
        info = _LOGISTICS_STATUS_MAP.get(order.status, {"state": "unknown", "desc": "未知状态"})
        return {
            "order_id": order.order_id,
            "status": order.status,
            "logistics_state": info["state"],
            "logistics_desc": info["desc"],
            "address": order.address,
        }


def cancel_order(
    order_id: str,
    user_id: str,
    reason: str | None = None,
    thread_id: str = "system",
) -> CancelOrderOutput:
    """取消订单：事务内完成状态更新 + 退款 + 库存回补 + 优惠券回退。

    任一步失败整体回滚（依赖 ``get_session`` 的事务边界）。

    注意：
    - 本函数假设已通过二次确认（由 execute_tool 在调用前完成确认）。
    - 审计/幂等由 execute_tool 闸门④⑤负责，业务函数只专注状态变更。
    - thread_id 参数保留供兼容，审计已上移。
    """
    with get_session() as session:
        order = session.execute(
            select(Order)
            .where(Order.order_id == order_id, Order.user_id == user_id)
            .with_for_update()
        ).scalar_one_or_none()

        if order is None:
            return CancelOrderOutput(False, None, False, False, "订单不存在或不属于该用户")

        ok, msg = check_cancelable(order.status)
        if not ok:
            # 已取消视为幂等成功（不重复退款），其余不可取消原因直接拒绝
            if order.status == OrderStatus.CANCELLED.value:
                return CancelOrderOutput(
                    True, None, False, False, "订单已取消（幂等返回）"
                )
            return CancelOrderOutput(False, None, False, False, msg)

        # 执行取消：联动处理
        need_refund = order.paid
        restock = order.stock_deducted
        refund_amount: Decimal | None = None

        if need_refund:
            # 待发货已付款 → 退款；待付款未付款 → 无退款
            refund_amount = order.amount
            order.refund_amount = refund_amount
        if restock:
            # 库存回补：本系统以 stock_deducted 标记位表示；真实场景调用库存服务
            order.stock_deducted = False
        if order.coupon_code and not order.coupon_reverted:
            # 优惠券回退
            order.coupon_reverted = True

        order.status = OrderStatus.CANCELLED.value
        order.cancel_reason = reason

        # 审计职责统一由 execute_tool 闸门⑤负责（见 app/tools/registry.py），
        # 业务函数只专注状态变更，避免重复审计。
        session.flush()

        return CancelOrderOutput(
            success=True,
            refund_amount=refund_amount,
            need_refund=need_refund,
            restock=restock,
            message="已取消" + ("，退款将原路退回" if need_refund else ""),
        )


# ---------------------------------------------------------------------------
# 发货后售后：退款 / 退货 / 改址
# ---------------------------------------------------------------------------


# 退款金额阈值：超过转人工审批（Spec 4.3 高风险工具）
REFUND_APPROVAL_THRESHOLD = Decimal("1000.00")


def _check_refundable(status: str, refund_status: str, paid: bool) -> tuple[bool, str]:
    """退款可申请性规则（确定性，不交模型）。"""
    if not paid:
        return False, "订单未付款，无需退款"
    if refund_status != RefundStatus.NONE.value:
        return False, "退款已发起或已完成"
    if status in (OrderStatus.SHIPPED.value, OrderStatus.COMPLETED.value):
        return True, "可申请退款"
    if status == OrderStatus.PENDING_SHIP.value:
        return True, "可申请退款（未发货直接退）"
    if status == OrderStatus.CANCELLED.value:
        return False, "订单已取消，退款已处理"
    return False, "当前状态不支持退款"


def apply_refund(
    order_id: str,
    user_id: str,
    reason: str | None = None,
    thread_id: str = "system",
) -> dict:
    """发起退款（高风险写操作，金额超阈值转人工由 execute_tool 判断）。

    事务内：状态 → refunding + refund_status → processing + 记录退款金额。
    """
    with get_session() as session:
        order = session.execute(
            select(Order)
            .where(Order.order_id == order_id, Order.user_id == user_id)
            .with_for_update()
        ).scalar_one_or_none()
        if order is None:
            return {"success": False, "message": "订单不存在或不属于该用户"}

        ok, msg = _check_refundable(order.status, order.refund_status, order.paid)
        if not ok:
            return {"success": False, "message": msg}

        order.refund_status = RefundStatus.PROCESSING.value
        order.status = OrderStatus.REFUNDING.value
        order.refund_amount = order.amount
        session.flush()

        return {
            "success": True,
            "message": "退款申请已受理，金额将原路退回",
            "refund_amount": float(order.amount),
        }


def _check_returnable(status: str, return_status: str) -> tuple[bool, str]:
    """退货可申请性规则（确定性）。"""
    if return_status != ReturnStatus.NONE.value:
        return False, "退货已发起或已完成"
    if status in (OrderStatus.SHIPPED.value, OrderStatus.COMPLETED.value):
        return True, "可申请退货"
    if status == OrderStatus.PENDING_SHIP.value:
        return True, "未发货，可申请退货"
    if status == OrderStatus.CANCELLED.value:
        return False, "订单已取消"
    return False, "当前状态不支持退货"


def apply_return(
    order_id: str,
    user_id: str,
    reason: str | None = None,
    thread_id: str = "system",
) -> dict:
    """申请退货/换货（高风险写操作）。"""
    with get_session() as session:
        order = session.execute(
            select(Order)
            .where(Order.order_id == order_id, Order.user_id == user_id)
            .with_for_update()
        ).scalar_one_or_none()
        if order is None:
            return {"success": False, "message": "订单不存在或不属于该用户"}

        ok, msg = _check_returnable(order.status, order.return_status)
        if not ok:
            return {"success": False, "message": msg}

        order.return_status = ReturnStatus.PROCESSING.value
        order.status = OrderStatus.RETURNING.value
        session.flush()

        return {
            "success": True,
            "message": "退货申请已受理，请按指引寄回商品",
        }


def update_address(
    order_id: str,
    user_id: str,
    address: str,
    thread_id: str = "system",
) -> dict:
    """修改收货地址（中风险写操作，仅待发货状态可改）。"""
    with get_session() as session:
        order = session.execute(
            select(Order)
            .where(Order.order_id == order_id, Order.user_id == user_id)
            .with_for_update()
        ).scalar_one_or_none()
        if order is None:
            return {"success": False, "message": "订单不存在或不属于该用户"}

        # 仅待发货可改地址（已发货/已完成不可改）
        if order.status != OrderStatus.PENDING_SHIP.value:
            return {"success": False, "message": "当前状态不支持修改地址"}

        old_address = order.address
        order.address = address
        session.flush()

        return {
            "success": True,
            "message": "收货地址已更新",
            "old_address": old_address,
            "new_address": address,
        }
