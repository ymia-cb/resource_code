"""VLM 图像理解工具测试。

覆盖：三类图片类型、低置信转人工、mock 降级、提示词构造、JSON 解析。
运行：python -m pytest tests/test_vision.py -v
"""
from __future__ import annotations

import json

import pytest

from app.tools.vision import (
    image_understand,
    build_vision_prompt,
    parse_structured,
    CONFIDENCE_THRESHOLD,
)


def test_build_vision_prompt_product():
    prompt = build_vision_prompt("product")
    assert "product" in prompt
    assert "confidence" in prompt


def test_build_vision_prompt_screenshot_with_task():
    prompt = build_vision_prompt("screenshot", task="提取错误码")
    assert "screenshot" in prompt
    assert "提取错误码" in prompt


def test_parse_structured_direct_json():
    raw = '{"type":"product","confidence":0.9}'
    result = parse_structured(raw)
    assert result["type"] == "product"
    assert result["confidence"] == 0.9


def test_parse_structured_markdown_wrapped():
    raw = '```json\n{"type":"receipt","order_id":"123","confidence":0.95}\n```'
    result = parse_structured(raw)
    assert result["order_id"] == "123"


def test_parse_structured_fallback():
    raw = "无法识别"
    result = parse_structured(raw)
    assert result["confidence"] == 0.0


def test_image_understand_product_mock():
    """无 VLM key 时走 mock_vlm，返回 product 结构化结果 + 高置信。"""
    result = image_understand("http://img/product.jpg", "product")
    assert result["status"] == "ok"
    assert result["data"]["type"] == "product"
    assert result["data"]["confidence"] >= CONFIDENCE_THRESHOLD


def test_image_understand_screenshot_mock():
    result = image_understand("http://img/err.png", "screenshot")
    assert result["status"] == "ok"
    assert result["data"]["type"] == "screenshot"
    assert "error_code" in result["data"]


def test_image_understand_receipt_mock():
    result = image_understand("http://img/receipt.png", "receipt")
    assert result["status"] == "ok"
    assert result["data"]["type"] == "receipt"
    assert result["data"]["order_id"] == "20261021001"


def test_image_understand_low_confidence(monkeypatch):
    """模拟 VLM 返回低置信 → low_confidence 转人工。"""
    # 直接测 parse + confidence 判断逻辑：构造低置信结果
    result = {"type": "product", "confidence": 0.3}
    # 通过 monkeypatch llm 返回低置信 JSON
    import app.tools.vision as vision

    monkeypatch.setattr(
        vision, "llm",
        lambda role, messages: json.dumps({"type": "product", "confidence": 0.3}),
    )
    out = image_understand("http://img/x.jpg", "product")
    assert out["status"] == "low_confidence"
    assert out["data"]["confidence"] < CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
