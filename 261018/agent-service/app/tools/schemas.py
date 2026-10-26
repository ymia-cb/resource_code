"""工具入参模型（pydantic，闸门②参数校验用）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryOrderInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    user_id: str | None = None


class QueryLogisticsInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    user_id: str | None = None


class CancelOrderInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=255)
    # 二次确认标记：执行写操作前必须为 True
    confirm: bool = False


class QueryCouponInput(BaseModel):
    coupon_code: str = Field(..., min_length=1, max_length=64)
    user_id: str | None = None


class QueryProductInput(BaseModel):
    product_id: str = Field(..., min_length=1, max_length=64)


class ApplyRefundInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=255)
    confirm: bool = False


class ApplyReturnInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=255)
    confirm: bool = False


class UpdateAddressInput(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    address: str = Field(..., min_length=1, max_length=500)
    confirm: bool = False


class ImageUnderstandInput(BaseModel):
    image_url: str = Field(..., min_length=1)
    image_type: Literal["product", "screenshot", "receipt"]
    task: str | None = None


class LoadSkillInput(BaseModel):
    skill_code: str = Field(..., min_length=1, max_length=64)
