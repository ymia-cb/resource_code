"""订单路由：查询 + 取消（含二次确认）。

对接 app/tools/order.py 的 query_order / cancel_order / check_cancelable。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import CancelOrderRequest, CancelOrderResponse, OrderInfo
from app.tools.order import query_order, cancel_order, check_cancelable

router = APIRouter(tags=["orders"])


@router.get("/orders/{order_id}", response_model=OrderInfo)
def get_order(order_id: str, user_id: str | None = None) -> OrderInfo:
    info = query_order(order_id, user_id)
    if info is None:
        raise HTTPException(status_code=404, detail="订单不存在")
    return OrderInfo(**info)


@router.post("/orders/cancel", response_model=CancelOrderResponse)
def cancel(req: CancelOrderRequest) -> CancelOrderResponse:
    """取消订单。

    安全点：
    - 先查订单 + 规则引擎判可取消性（check_cancelable）
    - 必须显式 confirm=True（二次确认由调用方完成）
    """
    info = query_order(req.order_id, req.user_id)
    if info is None:
        raise HTTPException(status_code=404, detail="订单不存在或不属于该用户")

    ok, msg = check_cancelable(info["status"])
    if not ok and msg != "订单已取消":
        raise HTTPException(status_code=409, detail=msg)

    if not req.confirm:
        # 未二次确认 → 返回可取消性提示，不执行
        return CancelOrderResponse(
            success=False,
            message=f"需二次确认：{msg}。确认取消请再次请求并 confirm=true",
        )

    out = cancel_order(
        req.order_id,
        req.user_id,
        reason=req.reason,
        thread_id=f"http-{req.user_id}",
    )
    return CancelOrderResponse(
        success=out.success,
        message=out.message,
        refund_amount=float(out.refund_amount) if out.refund_amount is not None else None,
        need_refund=out.need_refund,
        restock=out.restock,
    )


@router.get("/orders/{order_id}/cancelable")
def order_cancelable(order_id: str, user_id: str | None = None) -> dict:
    """查询订单是否可取消（只读，供前端/agent 前置判断）。"""
    info = query_order(order_id, user_id)
    if info is None:
        raise HTTPException(status_code=404, detail="订单不存在")
    ok, msg = check_cancelable(info["status"])
    return {"order_id": order_id, "cancelable": ok, "message": msg, "status": info["status"]}
