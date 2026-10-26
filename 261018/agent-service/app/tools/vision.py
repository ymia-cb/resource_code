"""VLM 图像理解工具（Spec 4.7 / 7.8）。

唯一调用 VLM 的工具，把图片转成结构化结果（JSON + confidence）供推理循环消费。

关键约束（Spec 4.7）：
- 结构化输出：强制 VLM 返回 JSON，禁止自由文本
- 置信度必带：confidence < 阈值（0.7）→ low_confidence 转人工
- 无副作用：只读图片，风险 low
"""
from __future__ import annotations

import json
import re
from typing import Literal

from app.llm.router import llm

# 置信度阈值：低于此值转人工（Spec 4.7）
CONFIDENCE_THRESHOLD = 0.7

ImageType = Literal["product", "screenshot", "receipt"]


# ---------------------------------------------------------------------------
# 提示词构造（按 image_type）
# ---------------------------------------------------------------------------


def build_vision_prompt(image_type: str, task: str | None = None) -> str:
    """按图片类型构造结构化提示词，强制 JSON 输出。"""
    task_hint = f"\n额外要求：{task}" if task else ""

    prompts = {
        "product": (
            "你是电商商品图识别助手。请识别图片中的商品，只输出 JSON：\n"
            '{"type":"product","category":"<品类>","attributes":{"color":"<颜色>","size":"<尺码>"},"confidence":<0-1>}'
        ),
        "screenshot": (
            "你是电商客服截图诊断助手。请分析截图中的报错/异常，只输出 JSON：\n"
            '{"type":"screenshot","error_code":"<错误码>","issue":"<问题描述>","suggestion":"<建议>","confidence":<0-1>}'
        ),
        "receipt": (
            "你是电商票据解析助手。请提取票据中的订单信息，只输出 JSON：\n"
            '{"type":"receipt","order_id":"<订单号>","amount":<金额数字>,"date":"<日期YYYY-MM-DD>","confidence":<0-1>}'
        ),
    }
    base = prompts.get(image_type, prompts["product"])
    return base + task_hint


def parse_structured(raw: str) -> dict:
    """从 VLM 输出中解析 JSON（容忍 markdown 代码块包裹）。"""
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 提取首个 {...} 片段
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 解析失败 → 返回低置信占位
    return {"type": "unknown", "confidence": 0.0}


# ---------------------------------------------------------------------------
# 图像理解主函数
# ---------------------------------------------------------------------------


def image_understand(
    image_url: str,
    image_type: str,
    task: str | None = None,
) -> dict:
    """调 VLM 识别图片，返回结构化结果（含 confidence）。

    返回：
    - {"status": "ok", "data": {...}}  —— confidence >= 阈值
    - {"status": "low_confidence", "data": {...}} —— confidence < 阈值（转人工）
    """
    prompt = build_vision_prompt(image_type, task)

    # 构造多模态消息（OpenAI 兼容的 content 数组格式）
    from langchain_core.messages import HumanMessage

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    # 用自定义消息承载多模态 content（llm 走 vlm 通道）
    message = HumanMessage(content=content)

    raw = llm("vlm", [message])
    result = parse_structured(raw)

    # 补全 type 字段
    if "type" not in result:
        result["type"] = image_type

    confidence = float(result.get("confidence", 0.0))
    if confidence < CONFIDENCE_THRESHOLD:
        return {"status": "low_confidence", "data": result}

    return {"status": "ok", "data": result}
