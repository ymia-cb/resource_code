# 电商客服意图分类 — LoRA 微调

针对电商智能客服的「问题意图识别」任务，提供数据集生成 + LoRA 微调 + 推理的完整链路。

## 目录结构

```
intent-finetune/
├── labels.py              # 意图标签体系（8 类，全项目唯一来源）
├── generate_dataset.py    # 数据集生成脚本（模板 + 槽位替换 + 噪声扰动）
├── prepare_real_logs.py   # 真实日志预处理（脱敏->清洗->打标->输出同 schema）
├── train_lora.py          # LoRA 微调训练脚本（PEFT + Transformers）
├── infer.py               # 加载 adapter 做意图预测
├── requirements.txt       # 依赖清单
└── data/                  # 生成的数据集
    ├── train.jsonl
    ├── valid.jsonl
    └── sample_raw_logs.csv  # 真实日志格式示例
```

## 意图标签（8 类）

| id | 中文名 | 说明 |
|----|--------|------|
| presale | 售前咨询 | 商品参数、用法、对比、推荐 |
| aftersale | 售后服务 | 维修、保修、使用问题 |
| logistics | 物流查询 | 发货、到哪了、多久到、改地址 |
| refund | 退款退货 | 申请退款、退货流程、进度 |
| complaint | 投诉建议 | 不满、投诉、差评 |
| price_stock | 价格库存 | 价格、优惠、库存、补货 |
| chitchat | 闲聊 | 打招呼、无关话题 |
| other | 其他 | 兜底 |

## 快速开始

### 1. 生成数据集

```bash
python generate_dataset.py --total 1000 --seed 42
# 产出 data/train.jsonl（900 条）+ data/valid.jsonl（100 条）
```

数据格式（LLaMA-Factory / 自写脚本通用）：

```json
{"instruction": "你是一个电商客服意图分类助手...", "input": "我的快递到哪了", "output": "物流查询"}
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. LoRA 微调

```bash
python train_lora.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --data_dir data \
    --output_dir output/intent-lora \
    --num_epochs 3
```

显存不足时加 `--use_4bit`（需 bitsandbytes）。换更大模型只改 `--model_name`，
例如 `Qwen/Qwen2.5-1.5B-Instruct`。

### 4. 推理验证

```bash
python infer.py --model_name Qwen/Qwen2.5-0.5B-Instruct --adapter output/intent-lora --text "我的快递到哪了"
```

不传 `--text` 进入交互模式。

## 用真实客服日志训练（推荐）

真实日志比模板数据泛化上限更高。`prepare_real_logs.py` 完成 **脱敏 → 清洗 → 打标 → 输出**，
输出与 `generate_dataset.py` 同 schema，可无缝接入 `train_lora.py`。

输入格式（jsonl 或 csv），至少需要「用户消息」字段；有「标签」字段时优先用它：

```csv
utterance,label
我的快递什么时候到啊,物流
我要退款 订单号 2026102300123456,退款
```

### 带标签（推荐，标签来自工单类型/技能组）

```bash
python prepare_real_logs.py \
    --input raw_logs.csv --format csv \
    --utterance-col utterance --label-col label \
    --out data/real_train.jsonl
```

原始标签会自动映射到 8 类（映射表在脚本 `RAW_LABEL_MAP` 里，可按你的系统口径扩充）。

### 无标签（关键词规则弱打标）

```bash
python prepare_real_logs.py \
    --input raw_logs.jsonl --format jsonl \
    --utterance-col text \
    --out data/real_train.jsonl
```

### 与模板数据混合（补少数类，推荐 70~80% 真实 + 20~30% 模板）

```bash
python prepare_real_logs.py \
    --input raw_logs.csv --format csv \
    --utterance-col utterance --label-col label \
    --mix-template data/train.jsonl --mix-ratio 0.2 \
    --out data/mixed_train.jsonl
```

### 自动脱敏项

手机号 `<PHONE>`、座机 `<TEL>`、订单号 `<ORDER>`、身份证 `<IDCARD>`、
邮箱 `<EMAIL>`、姓名 `<NAME>`、地址 `<ADDR>`。训练前请务必确认脱敏到位。

### 已知边界

- **规则打标有盲区**：关键词覆盖不到的表达会落到「其他」。生产建议用大模型
  （Qwen/GPT）自动打标 + 人工抽检，或直接用客服系统已有工单类型做弱标签。
- 类别分布失衡时（物流/退款多、投诉少），配合模板混合或训练时的类别权重缓解。
- 日志里客服自己的回复、无效消息（纯表情/单字）会在清洗阶段被丢弃。

## 关键设计

- **Response-only loss**：只对 assistant 回答部分算交叉熵，prompt 部分 label 置 -100，
  分类任务更稳。
- **Qwen ChatML 模板**：`<|im_start|>` 结构，与 Qwen 官方一致。
- **LoRA 目标模块**：覆盖 q/k/v/o + gate/up/down 投影，r=8、alpha=16 是分类任务的轻量配置。
- **标签唯一来源**：`labels.py` 被数据生成、训练、推理共同 import，改标签只动一处。

## 已知边界

- 本脚本用 **CausalLM 生成式**做分类（模型吐出中文标签），非 sequence classification head；
  好处是复用 Qwen 指令微调能力、可零样本先试，坏处是推理需一次 generate。
- 数据集是程序化生成的模板数据，**建议补充真实线上客服日志**做混合训练，泛化更佳。
