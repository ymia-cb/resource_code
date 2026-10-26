# -*- coding: utf-8 -*-
"""意图标签体系定义（电商通用 8 类）。

全项目统一从这里 import，训练/推理/数据生成共用一份标签，避免不一致。
"""

# 意图标签：id -> 中文名
INTENT_LABELS = {
    "presale": "售前咨询",        # 商品参数、用法、对比、推荐
    "aftersale": "售后服务",      # 维修、保修、使用问题、退换政策咨询
    "logistics": "物流查询",      # 发货、到哪了、多久到、改地址
    "refund": "退款退货",         # 申请退款、退货流程、退款进度
    "complaint": "投诉建议",      # 对商品/服务不满、投诉、差评
    "price_stock": "价格库存",    # 价格、优惠、库存、补货
    "chitchat": "闲聊",           # 打招呼、无关话题
    "other": "其他",              # 兜底，无法归类
}

# 中文名 -> id（用于反查）
LABEL_TO_ID = {v: k for k, v in INTENT_LABELS.items()}

# 有序 id 列表（训练时索引对齐用）
LABEL_IDS = list(INTENT_LABELS.keys())

# 类别数量
NUM_LABELS = len(INTENT_LABELS)


def label_id(name: str) -> str:
    """中文名转 id，找不到返回 other。"""
    return LABEL_TO_ID.get(name, "other")
