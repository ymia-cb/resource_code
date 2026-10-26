# -*- coding: utf-8 -*-
"""真实客服日志预处理脚本：脱敏 -> 清洗 -> 打标 -> 输出训练数据。

输入支持两种原始日志格式（jsonl 或 csv）：
  1. 带标签的日志（推荐）：
     每行包含「用户消息」和「意图标签」字段，标签会映射到 8 类标签体系。
  2. 无标签的日志：
     仅含用户消息，脚本用关键词规则打弱标签（可选的 LLM 打标见下方说明）。

输出：与 generate_dataset.py 同 schema 的 jsonl，可直接喂给 train_lora.py，
     也支持「真实日志 + 模板数据」混合（--mix-template 传入模板数据比例）。

用法示例：
  # 带标签（字段名：utterance=用户消息, label=原始标签）
  python prepare_real_logs.py \
      --input raw_logs.csv --format csv \
      --utterance-col utterance --label-col label \
      --out data/real_train.jsonl

  # 无标签，纯关键词规则打标
  python prepare_real_logs.py \
      --input raw_logs.jsonl --format jsonl \
      --utterance-col text \
      --out data/real_train.jsonl

  # 与模板数据混合（80% 真实 + 20% 模板）
  python prepare_real_logs.py \
      --input raw_logs.csv --format csv \
      --utterance-col utterance --label-col label \
      --mix-template data/train.jsonl --mix-ratio 0.2 \
      --out data/mixed_train.jsonl
"""

import argparse
import csv
import json
import os
import re
import random

from labels import INTENT_LABELS, LABEL_TO_ID, label_id

# ---------------------------------------------------------------------------
# 1) 脱敏规则
# ---------------------------------------------------------------------------
# 手机号（含中间空格/横线）
RE_PHONE = re.compile(r"1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}")
# 座机
RE_TEL = re.compile(r"0\d{2,3}-\d{7,8}")
# 订单号（常见形态：纯长数字 / 字母数字混合，取较长者，避免误伤）
RE_ORDER = re.compile(r"\b(?:[A-Z]{2,4}-?)?\d{10,20}\b")
# 身份证
RE_IDCARD = re.compile(r"\b\d{17}[\dXx]\b")
# 邮箱
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 地址关键词（粗略，命中即整体替换）
RE_ADDR = re.compile(r"(省|市|区|县|路|街|号|栋|单元|室|小区|花园|大厦){2,}")
# 姓名（常见姓氏+名，粗略）
RE_NAME = re.compile(r"(张|王|李|赵|刘|陈|杨|黄|周|吴|徐|孙|马|朱|胡|郭|何|林|罗|郑|梁|谢|宋|唐|许|韩|冯|邓|曹|彭|曾)[\u4e00-\u9fa5]{1,2}")


def desensitize(text):
    """替换敏感信息为占位符。"""
    t = RE_PHONE.sub("<PHONE>", text)
    t = RE_IDCARD.sub("<IDCARD>", t)
    t = RE_EMAIL.sub("<EMAIL>", t)
    t = RE_ORDER.sub("<ORDER>", t)
    t = RE_TEL.sub("<TEL>", t)
    t = RE_NAME.sub("<NAME>", t)
    t = RE_ADDR.sub("<ADDR>", t)
    return t


# ---------------------------------------------------------------------------
# 2) 清洗规则
# ---------------------------------------------------------------------------
def clean(text):
    """去掉空白/无效字符，过滤无意义短消息。返回清洗后的文本或 None（丢弃）。"""
    t = text.strip()
    if not t:
        return None
    # 去掉纯标点/表情/空白
    if len(re.sub(r"[\s\W_]+", "", t)) < 2:
        return None
    # 超长消息截断（可能是粘贴）
    if len(t) > 200:
        t = t[:200]
    # 压缩连续空白
    t = re.sub(r"\s+", " ", t)
    return t


# ---------------------------------------------------------------------------
# 3) 弱标签规则（无标签日志时用）
# ---------------------------------------------------------------------------
# 每个意图的关键词，按命中优先级匹配
RULE_KEYWORDS = [
    ("refund", ["退款", "退货", "退钱", "退款进度", "取消订单", "七天无理由"]),
    ("logistics", ["发货", "物流", "快递", "到哪", "多久到", "收货地址", "派送", "签收", "催单"]),
    ("complaint", ["投诉", "差评", "态度", "赔偿", "太差", "不满意", "体验差"]),
    ("aftersale", ["保修", "维修", "售后", "坏了", "用不了", "开不了机", "换货", "说明书"]),
    ("price_stock", ["多少钱", "优惠", "补货", "有货", "库存", "打折", "便宜", "价格"]),
    ("presale", ["怎么样", "好用吗", "哪个好", "功能", "介绍", "参数", "推荐", "适合"]),
    ("chitchat", ["你好", "在吗", "谢谢", "再见", "拜拜", "哈哈", "早安", "晚安"]),
]


def rule_label(text):
    """关键词规则打标，命中多个取第一个。无命中返回 other。"""
    for intent, kws in RULE_KEYWORDS:
        for kw in kws:
            if kw in text:
                return INTENT_LABELS[intent]
    return INTENT_LABELS["other"]


# ---------------------------------------------------------------------------
# 标签映射：原始系统标签 -> 8 类（可自行扩充）
# ---------------------------------------------------------------------------
RAW_LABEL_MAP = {
    # 售前
    "售前": "presale", "咨询": "presale", "商品咨询": "presale", "导购": "presale",
    # 售后
    "售后": "aftersale", "维修": "aftersale", "保修": "aftersale", "使用问题": "aftersale",
    # 物流
    "物流": "logistics", "发货": "logistics", "催发货": "logistics", "配送": "logistics",
    # 退款
    "退款": "refund", "退货": "refund", "退换货": "refund", "仅退款": "refund",
    # 投诉
    "投诉": "complaint", "客诉": "complaint", "差评": "complaint",
    # 价格库存
    "价格": "price_stock", "库存": "price_stock", "优惠": "price_stock", "补货": "price_stock",
    # 闲聊
    "闲聊": "chitchat", "寒暄": "chitchat",
    # 其他
    "其他": "other", "未分类": "other",
}


def map_raw_label(raw):
    """把原始标签映射到 8 类 id，未知映射到 other。"""
    raw = (raw or "").strip()
    for key, intent in RAW_LABEL_MAP.items():
        if key in raw:
            return intent
    return "other"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def read_records(path, fmt):
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError(f"不支持的格式: {fmt}")


def build_instruction():
    return ("你是一个电商客服意图分类助手。请判断用户这句话属于哪个意图类别，"
            "只输出类别名称，不要输出任何多余内容。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始日志文件路径")
    ap.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    ap.add_argument("--utterance-col", default="utterance", help="用户消息字段名")
    ap.add_argument("--label-col", default=None, help="标签字段名（无则用规则打标）")
    ap.add_argument("--out", default="data/real_train.jsonl")
    ap.add_argument("--mix-template", default=None, help="模板数据文件，用于混合")
    ap.add_argument("--mix-ratio", type=float, default=0.0, help="模板数据占比 0~1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = read_records(args.input, args.format)

    samples = []
    dropped = 0
    label_hits = 0
    for r in records:
        text = r.get(args.utterance_col, "")
        text = clean(text)
        if text is None:
            dropped += 1
            continue
        text = desensitize(text)

        if args.label_col and r.get(args.label_col):
            intent = map_raw_label(str(r.get(args.label_col)))
            label_hits += 1
        else:
            intent = LABEL_TO_ID[rule_label(text)]

        samples.append({
            "instruction": build_instruction(),
            "input": text,
            "output": INTENT_LABELS[intent],
        })

    print(f"[读取] {len(records)} 条原始记录")
    print(f"[清洗] 丢弃 {dropped} 条无效消息，保留 {len(samples)} 条")
    if args.label_col:
        print(f"[标签] 命中已有标签 {label_hits} 条，其余用规则补齐")

    # 可选：与模板数据混合
    if args.mix_template and args.mix_ratio > 0:
        tpl = read_records(args.mix_template, "jsonl")
        n_tpl = int(len(samples) * args.mix_ratio / (1 - args.mix_ratio))
        tpl = rng.sample(tpl, min(n_tpl, len(tpl)))
        samples = samples + tpl
        rng.shuffle(samples)
        print(f"[混合] 加入模板数据 {len(tpl)} 条，总计 {len(samples)} 条")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 分布统计
    from collections import Counter
    cnt = Counter(s["output"] for s in samples)
    print(f"[输出] {len(samples)} 条 -> {args.out}")
    print("[分布]")
    for k, name in INTENT_LABELS.items():
        print(f"  {name:<6} {cnt.get(name, 0)} 条")
    print("\n[样例]")
    for s in samples[:3]:
        print(f"  input={s['input']!r}  output={s['output']}")


if __name__ == "__main__":
    main()
