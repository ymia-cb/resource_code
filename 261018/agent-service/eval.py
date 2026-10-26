"""竖切片评测：L2 分类 + L4 端到端，golden set 基础版。

覆盖正常/边界/对抗三类，作为 harness 的起点。真实项目需扩到 ≥50 条
并接入 LLM-as-Judge（见 spec 第 5 章）。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.graph.build import build_graph
from app.memory.checkpointer import get_checkpointer

# golden set：每条含 query、期望 intent、期望命中 chunk（可多选）
GOLDEN = [
    # 正常
    {"query": "退款政策是什么", "intent": "simple", "expect_chunk": "faq-refund-0"},
    {"query": "秒杀商品能退吗", "intent": "simple", "expect_chunk": "faq-refund-1"},
    {"query": "多久发货", "intent": "simple", "expect_chunk": "faq-ship-0"},
    {"query": "优惠券会退吗", "intent": "simple", "expect_chunk": "faq-coupon-0"},
    # 复杂（竖切片阶段转 human）
    {"query": "我要退这个订单", "intent": "complex", "expect_chunk": None},
    {"query": "帮我取消订单123", "intent": "complex", "expect_chunk": None},
    # 边界 / 对抗：无法归类 → human
    {"query": "今天天气怎么样", "intent": "human", "expect_chunk": None},
    {"query": "", "intent": "human", "expect_chunk": None},
]


def run_eval() -> dict:
    app = build_graph(get_checkpointer("eval_checkpoints.db"))

    correct_intent = 0
    correct_chunk = 0
    total = len(GOLDEN)

    for i, case in enumerate(GOLDEN):
        r = app.invoke(
            {"messages": [HumanMessage(content=case["query"])]},
            {"configurable": {"thread_id": f"eval-{i}"}},
        )
        intent_ok = r.get("intent") == case["intent"]
        if intent_ok:
            correct_intent += 1

        chunks = [s["chunk_id"] for s in r.get("sources", [])]
        if case["expect_chunk"] is None:
            chunk_ok = chunks == []
        else:
            chunk_ok = case["expect_chunk"] in chunks
        if chunk_ok:
            correct_chunk += 1

        print(f"[{i+1}/{total}] intent={'OK' if intent_ok else 'FAIL'} "
              f"chunk={'OK' if chunk_ok else 'FAIL'}  query={case['query']!r}")

    intent_acc = correct_intent / total
    chunk_acc = correct_chunk / total
    print(f"\n意图分类准确率: {intent_acc:.0%} ({correct_intent}/{total})")
    print(f"检索命中准确率: {chunk_acc:.0%} ({correct_chunk}/{total})")
    return {"intent_acc": intent_acc, "chunk_acc": chunk_acc}


if __name__ == "__main__":
    run_eval()
