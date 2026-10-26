# -*- coding: utf-8 -*-
"""意图数据集生成脚本。

程序化生成电商客服意图分类数据集：
- 每个意图预置一批「种子问句」，再通过同义模板 + 槽位词替换 + 轻微噪声扰动
  扩充到目标数量，最终产出 train.jsonl / valid.jsonl。
- 每条样本为 {"instruction": ..., "input": ..., "output": <中文标签>}，
  与 LLaMA-Factory / 自写训练脚本通用。

用法：
    python generate_dataset.py [--total 1000] [--out data]

不依赖第三方库，纯标准库可跑。
"""

import argparse
import json
import os
import random

from labels import INTENT_LABELS, LABEL_IDS

# ---------------------------------------------------------------------------
# 种子数据：每个意图一组「句式模板 + 槽位词」
# 模板中用 {slot} 占位，槽位词在 slot_pool 里随机抽取。
# ---------------------------------------------------------------------------

SEED = {
    "presale": {
        "templates": [
            "这个{s}怎么样，好用吗",
            "{s}和{t}哪个好",
            "{s}有什么功能",
            "能介绍下{s}吗",
            "{s}适合{t}吗",
            "你们家{s}和别家比有什么优势",
            "{s}是{t}的吗",
            "买{s}前想先了解下参数",
        ],
        "slots": ["手机", "耳机", "空调", "洗衣机", "笔记本电脑", "扫地机器人", "智能手表", "电饭煲", "路由器", "投影仪"],
        "slots2": ["学生党", "老人", "租房用", "办公室", "家用", "送礼", "宿舍", "户外"],
    },
    "aftersale": {
        "templates": [
            "{s}坏了怎么办",
            "{s}用不了，怎么保修",
            "买的{s}出现{s2}问题",
            "{s}保修多久",
            "{s}说明书找不到了",
            "{s}需要维修，怎么联系售后",
            "收到{s}不会用，能指导下吗",
            "{s}用了几天就{s2}了，能换吗",
        ],
        "slots": ["手机", "耳机", "空调", "洗衣机", "电脑", "手表", "电饭煲", "路由器"],
        "slots2": ["开不了机", "噪音大", "发热", "连接不上", "屏幕坏了", "漏电", "卡顿", "异味"],
    },
    "logistics": {
        "templates": [
            "我的{s}发货了吗",
            "{s}什么时候能到",
            "物流到哪了，帮我查一下",
            "{s}能改收货地址吗",
            "订单怎么还没发货",
            "{s}多久能送到",
            "能帮我催下快递吗",
            "显示签收了但我没收到{s}",
        ],
        "slots": ["订单", "手机", "包裹", "快递", "耳机", "衣服", "商品", "货"],
        "slots2": ["北京", "上海", "广州", "深圳", "成都", "武汉", "杭州", "南京"],
    },
    "refund": {
        "templates": [
            "{s}想退货，怎么操作",
            "申请退款后{s}多久到账",
            "{s}不满意，能退吗",
            "退款进度怎么查",
            "{s}和描述不符，我要退",
            "退货流程是什么",
            "{s}质量问题，要求退款",
            "取消订单后钱{s}退回来",
        ],
        "slots": ["这个商品", "东西", "衣服", "鞋子", "手机", "货", "订单", "包裹"],
        "slots2": ["什么时候", "多久", "几天", "怎么", "能", "会"],
    },
    "complaint": {
        "templates": [
            "我要投诉{s}",
            "{s}太差了，客服态度{s2}",
            "你们{s}有问题，必须给个说法",
            "对{s}很不满意",
            "差评，{s}{s2}",
            "投诉你们{s}",
            "{s}体验极差，要赔偿",
            "我要找你们{s}投诉",
        ],
        "slots": ["客服", "商品", "物流", "服务", "产品质量", "包装", "发货速度", "售后"],
        "slots2": ["很差", "敷衍", "恶劣", "太慢", "有问题", "不行", "离谱"],
    },
    "price_stock": {
        "templates": [
            "{s}多少钱",
            "{s}有优惠吗",
            "{s}什么时候补货",
            "{s}有货吗",
            "能便宜点吗，{s}",
            "{s}有活动价吗",
            "{s}库存还有吗",
            "双十一{s}打几折",
        ],
        "slots": ["这个", "手机", "耳机", "空调", "商品", "衣服", "鞋子", "电脑"],
        "slots2": ["多少钱", "什么价", "有货没", "优惠吗", "打折吗"],
    },
    "chitchat": {
        "templates": [
            "你好",
            "在吗",
            "谢谢你",
            "再见",
            "今天天气不错",
            "你吃饭了吗",
            "哈哈",
            "你是谁",
        ],
        "slots": [],
        "slots2": [],
    },
    "other": {
        "templates": [
            "帮我{s}一下",
            "这个{s}是什么意思",
            "我想{s}",
            "请问{s}",
            "能不能{s}",
            "怎么{s}",
        ],
        "slots": ["查", "看", "问", "了解", "咨询", "知道", "弄"],
        "slots2": [],
    },
}

# 轻微噪声扰动（可选，模拟真实输入的口语化/错别字）
NOISE_PAIRS = [
    ("吗", "嘛"), ("吧", "呗"), ("的", "得"), ("呢", "呐"),
    ("怎么", "咋"), ("什么", "啥"), ("这个", "这"),
]


def _fill(template, slot, slot2):
    s = template.replace("{s}", slot)
    if "{s2}" in s:
        s = s.replace("{s2}", slot2)
    return s


def _add_noise(text, rng, p=0.15):
    """以 p 的概率做一次口语化替换。"""
    if rng.random() > p:
        return text
    for a, b in NOISE_PAIRS:
        if a in text:
            return text.replace(a, b, 1)
    return text


def _sample_text(intent, rng):
    """为某个意图生成一条问句文本。"""
    cfg = SEED[intent]
    tpl = rng.choice(cfg["templates"])
    slot = rng.choice(cfg["slots"]) if cfg["slots"] else ""
    slot2 = rng.choice(cfg["slots2"]) if cfg["slots2"] else ""
    text = _fill(tpl, slot, slot2)
    text = _add_noise(text, rng)
    return text.strip()


def generate(total, rng):
    """均匀分配：每个意图约 total/8 条。"""
    per = max(1, total // len(LABEL_IDS))
    samples = []
    for intent in LABEL_IDS:
        for _ in range(per):
            samples.append({
                "instruction": "你是一个电商客服意图分类助手。请判断用户这句话属于哪个意图类别，"
                               "只输出类别名称，不要输出任何多余内容。",
                "input": _sample_text(intent, rng),
                "output": INTENT_LABELS[intent],
            })
    # 补齐余数
    remain = total - len(samples)
    for i in range(remain):
        intent = LABEL_IDS[i % len(LABEL_IDS)]
        samples.append({
            "instruction": "你是一个电商客服意图分类助手。请判断用户这句话属于哪个意图类别，"
                           "只输出类别名称，不要输出任何多余内容。",
            "input": _sample_text(intent, rng),
            "output": INTENT_LABELS[intent],
        })
    rng.shuffle(samples)
    return samples


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1000, help="训练集总条数")
    ap.add_argument("--valid-ratio", type=float, default=0.1, help="验证集占比")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    all_samples = generate(args.total, rng)
    n_valid = int(len(all_samples) * args.valid_ratio)
    valid = all_samples[:n_valid]
    train = all_samples[n_valid:]

    train_path = os.path.join(args.out, "train.jsonl")
    valid_path = os.path.join(args.out, "valid.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    # 统计各意图分布
    from collections import Counter
    cnt = Counter(s["output"] for s in train)
    print(f"[完成] 训练集 {len(train)} 条 -> {train_path}")
    print(f"[完成] 验证集 {len(valid)} 条 -> {valid_path}")
    print("[分布]")
    for k in LABEL_IDS:
        print(f"  {INTENT_LABELS[k]:<6} {cnt.get(INTENT_LABELS[k], 0)} 条")
    print("\n[样例]")
    for s in train[:3]:
        print(f"  input={s['input']!r}  output={s['output']}")


if __name__ == "__main__":
    main()
