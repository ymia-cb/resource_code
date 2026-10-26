"""订单表（Order）与订单项表（OrderItem）。

支撑 ``query_order`` / ``cancel_order`` 工具。订单状态机与 Spec 11.3 可取消性规则对齐：

    pending_payment（待付款）→ pending_ship（待发货）→ shipped（已发货）
        → completed（已完成） / cancelled（已取消）

M1 阶段：整单取消，不支持部分取消（见 Spec 11.6）。
"""
from __future__ import annotations

from enum import Enum

from sqlalchemy import (
    Column,
    String,
    BigInteger,
    Integer,
    Numeric,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    func,
    Index,
)
from sqlalchemy.orm import relationship

from app.db.base import Base


class OrderStatus(str, Enum):
    """订单状态枚举。"""

    PENDING_PAYMENT = "pending_payment"  # 待付款
    PENDING_SHIP = "pending_ship"  # 待发货
    SHIPPED = "shipped"  # 已发货
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消
    REFUNDING = "refunding"  # 退款中
    RETURNING = "returning"  # 退货中


class RefundStatus(str, Enum):
    """退款子状态（独立于订单主状态）。"""

    NONE = "none"  # 未发起
    PROCESSING = "processing"  # 退款处理中
    DONE = "done"  # 已退款


class ReturnStatus(str, Enum):
    """退货子状态（独立于订单主状态）。"""

    NONE = "none"  # 未发起
    PROCESSING = "processing"  # 退货处理中
    DONE = "done"  # 已退货


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=False, unique=True, index=True)  # 业务订单号
    user_id = Column(String(64), nullable=False, index=True)

    status = Column(String(32), nullable=False, default=OrderStatus.PENDING_PAYMENT.value)

    # 金额（Decimal，避免浮点误差）
    amount = Column(Numeric(10, 2), nullable=False, default=0)
    paid = Column(Boolean, nullable=False, default=False)  # 是否已付款

    # 联动字段
    stock_deducted = Column(Boolean, nullable=False, default=False)  # 库存是否已扣减
    coupon_code = Column(String(64), nullable=True)  # 使用的优惠券码（可回退）
    coupon_reverted = Column(Boolean, nullable=False, default=False)  # 优惠券是否已回退
    refund_amount = Column(Numeric(10, 2), nullable=True)  # 已退款金额

    # 取消相关
    cancel_reason = Column(String(255), nullable=True)  # 取消原因（审计用）

    # 收货地址（update_address 用）
    address = Column(String(500), nullable=True)

    # 退款/退货子状态
    refund_status = Column(String(32), nullable=False, default=RefundStatus.NONE.value)
    return_status = Column(String(32), nullable=False, default=ReturnStatus.NONE.value)

    extra = Column(JSON, nullable=True)  # 扩展字段

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    items = relationship(
        "OrderItem",
        back_populates="order",
        cascade="all, delete-orphan",
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    order_id = Column(
        String(64),
        ForeignKey("orders.order_id"),
        nullable=False,
        index=True,
    )
    sku = Column(String(64), nullable=False)
    product_name = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Numeric(10, 2), nullable=False, default=0)

    order = relationship("Order", back_populates="items")
