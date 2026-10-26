"""竖切片入口：跑通「文本 → 分类 → RAG 直答 → checkpoint 记忆」。

用法：
    python run.py "退款政策是什么"
    python run.py --demo    # 演示同一 thread_id 多轮记忆不丢
"""
from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from app.graph.build import build_graph
from app.memory.checkpointer import get_checkpointer


def _invoke(app, thread_id: str, text: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=text)]},
        config,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="电商客服 Agent 竖切片")
    parser.add_argument("text", nargs="?", help="单条消息")
    parser.add_argument("--demo", action="store_true", help="演示多轮记忆")
    args = parser.parse_args()

    app = build_graph(get_checkpointer())

    if args.demo:
        tid = "user-demo-1"
        print("=== 演示：同一 thread_id 多轮记忆 ===")
        for q in ["退款政策是什么", "那秒杀商品能退吗"]:
            r = _invoke(app, tid, q)
            print(f"\nQ: {q}")
            print(f"A: {r.get('final_answer')}")
            print(f"intent={r.get('intent')} sources={[s['chunk_id'] for s in r.get('sources', [])]}")
        return

    if not args.text:
        parser.print_help()
        sys.exit(1)

    r = _invoke(app, "user-1", args.text)
    print(f"intent   : {r.get('intent')}")
    print(f"answer   : {r.get('final_answer')}")
    print(f"sources  : {[s['chunk_id'] for s in r.get('sources', [])]}")
    print(f"needs_human: {r.get('needs_human')}")


if __name__ == "__main__":
    main()
