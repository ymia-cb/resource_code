# -*- coding: utf-8 -*-
"""Qwen 意图分类 LoRA 微调训练脚本（PEFT + Transformers 手写）。

流程：
    加载 Qwen 基座 + tokenizer -> 组装 instruction 模板 -> LoRA 配置
    -> Trainer 训练 -> 评估（准确率） -> 保存 adapter 到 output_dir

用法：
    python train_lora.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --data_dir data \
        --output_dir output/intent-lora \
        --num_epochs 3

说明：
    - 默认用 Qwen2.5-0.5B-Instruct，显存友好（约 4-6GB 可训）；
      换 1.5B/7B 只需改 --model_name。
    - 用 CausalLM 做分类：instruction+input 作为 prompt，output 作为回答，
      只对回答部分算 loss（response-only），比纯 generate 更稳。
    - 需要 bitsandbytes 可选开启 4bit 量化（--use_4bit）。
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from labels import INTENT_LABELS

# Qwen 指令模板：把 instruction+input 拼成 user 消息
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}{input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SYSTEM = "你是一个电商客服意图分类助手，请判断用户这句话属于哪个意图类别，只输出类别名称。"


def build_prompt(instruction, input_):
    return PROMPT_TEMPLATE.format(
        system=SYSTEM, instruction=instruction, input=input_ or ""
    )


def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def preprocess(samples, tokenizer, max_length):
    """把样本转成模型输入。只对 assistant 回答部分算 loss。"""
    prompts = [build_prompt(s["instruction"], s["input"]) for s in samples]
    answers = [s["output"] + "<|im_end|>" for s in samples]

    tokenized = tokenizer(
        prompts, truncation=True, max_length=max_length, add_special_tokens=False
    )
    answer_tok = tokenizer(
        answers, truncation=True, max_length=64, add_special_tokens=False
    )

    input_ids, attention_mask, labels = [], [], []
    for i in range(len(prompts)):
        prompt_len = len(tokenized["input_ids"][i])
        ans_ids = answer_tok["input_ids"][i]
        full_ids = tokenized["input_ids"][i] + ans_ids
        full_mask = [1] * len(full_ids)
        # 回答部分为标签，prompt 部分设为 -100 不参与 loss
        lbl = [-100] * prompt_len + ans_ids

        # padding 到 max_length
        pad_len = max_length - len(full_ids)
        if pad_len > 0:
            full_ids += [tokenizer.pad_token_id] * pad_len
            full_mask += [0] * pad_len
            lbl += [-100] * pad_len
        else:
            full_ids = full_ids[:max_length]
            full_mask = full_mask[:max_length]
            lbl = lbl[:max_length]

        input_ids.append(full_ids)
        attention_mask.append(full_mask)
        labels.append(lbl)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class EvalMetrics:
    """推理时贪心解码，计算准确率。"""

    def compute(self, model, tokenizer, valid_path, max_length=256):
        samples = load_jsonl(valid_path)
        correct = 0
        model.eval()
        with torch.no_grad():
            for s in samples:
                prompt = build_prompt(s["instruction"], s["input"])
                inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inp = {k: v.to(model.device) for k, v in inp.items()}
                out = model.generate(**inp, max_new_tokens=32, do_sample=False)
                pred = tokenizer.decode(
                    out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                # 取首个命中的标签
                pred_label = first_label(pred)
                if pred_label == s["output"]:
                    correct += 1
        model.train()
        return {"accuracy": correct / len(samples)}


def first_label(text):
    for name in INTENT_LABELS.values():
        if name in text:
            return name
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="output/intent-lora")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_4bit", action="store_true", help="启用 4bit 量化（需 bitsandbytes）")
    args = ap.parse_args()

    # ---------- 加载 tokenizer & 模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "trust_remote_code": True,
    }
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if args.use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # ---------- LoRA 配置 ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- 数据 ----------
    train_raw = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    valid_raw = load_jsonl(os.path.join(args.data_dir, "valid.jsonl"))
    train_ds = Dataset.from_list(train_raw)
    valid_ds = Dataset.from_list(valid_raw)
    train_ds = train_ds.map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        batched=True, remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        batched=True, remove_columns=valid_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # ---------- 训练 ----------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # ---------- 评估 ----------
    metrics = EvalMetrics().compute(model, tokenizer, os.path.join(args.data_dir, "valid.jsonl"))
    print(f"\n[验证集准确率] {metrics['accuracy']:.4f}")

    # ---------- 保存 adapter ----------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[完成] LoRA adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
