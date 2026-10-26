# -*- coding: utf-8 -*-
"""加载 LoRA adapter 做意图预测。

用法：
    python infer.py --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --adapter output/intent-lora \
        --text "我的快递到哪了"

也支持交互模式（不传 --text 时进入循环输入）。
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import build_prompt
from labels import INTENT_LABELS, LABEL_TO_ID


def load(model_name, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text):
    prompt = build_prompt("", text)
    inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    pred = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for name in INTENT_LABELS.values():
        if name in pred:
            return name, LABEL_TO_ID.get(name, "other")
    return pred, "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="output/intent-lora")
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model_name, args.adapter)
    print("[已加载] LoRA 意图分类模型\n")

    if args.text:
        label, label_id = predict(model, tokenizer, args.text)
        print(f"输入：{args.text}")
        print(f"意图：{label}（{label_id}）")
    else:
        print("输入用户消息，回车判断意图，输入 exit 退出：")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.lower() in ("exit", "quit", "退出"):
                break
            label, label_id = predict(model, tokenizer, text)
            print(f"意图：{label}（{label_id}）\n")


if __name__ == "__main__":
    main()
