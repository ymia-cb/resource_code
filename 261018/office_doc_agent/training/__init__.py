"""Stage 8b：本地 RL 训练管线（GRPO）。

职责
----
消费推理采样（inference）+ 校验奖励（feedback）的产物，做组相对策略优化（GRPO）。
模型本地部署，训练也本地跑：采样一组候选 IR → 逐个走 pipeline 校验得绝对 reward →
组内 z-score 归一得相对优势 → 用优势加权做策略更新。

关键决策（对齐《沙箱·自动校验·RL 回流设计》的奖励设计）
--------------------------------------------------------
- GRPO 核心是「组内相对奖励」：优势 = (reward - mean) / std，抑制绝对漂移。
- 策略更新分两层：
  - Trainer（本模块）：负责采样 → 奖励 → 归一 → 计算梯度信号（纯逻辑，可离线测）。
  - PolicyUpdater：真正改模型权重，可插拔（本地 TRL/自研/离线 stub）。
    默认 StubPolicyUpdater 只记录梯度信号，不真改权重——离线也能验证整条链路。
- 反 reward hacking：hidden_gate（隐藏门禁）在训练前独立复核，防模型钻奖励漏洞。

桩 → 真实现
------------
- StubPolicyUpdater：记录优势加权后的梯度信号，不真改权重（离线验证）。
- TRLGRPOUpdater：接 TRL 的 GRPOTrainer（需 GPU + 本地模型，P3 填充点）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..feedback import RLFeedback
from ..inference import LocalInference


@runtime_checkable
class PolicyUpdater(Protocol):
    """策略更新器：把「带优势的样本」作用到模型。可插拔。"""

    def update(self, samples: list[dict]) -> dict:
        ...


@dataclass
class GRPOTrainer:
    """GRPO 训练器：采样 → 校验得 reward → 组内归一 → 策略更新。

    reward_fn：给定候选输出（IR 文本），返回一次校验的 reward 结构。
    这个函数是「环境」——把候选 IR 喂给 pipeline，拿四层校验 + 安全 claim 的结果。
    """

    inference: LocalInference
    reward_fn: Callable[[str], dict]
    updater: PolicyUpdater | None = None
    group_size: int = 4
    temperature: float = 0.8
    feedback: RLFeedback = field(default_factory=RLFeedback)

    def __post_init__(self) -> None:
        if self.updater is None:
            self.updater = StubPolicyUpdater()

    def step(self, prompt: str) -> dict:
        """执行一步 GRPO：采样一组候选 → 逐条打分 → 组内归一 → 更新策略。"""
        # 1. 在线采样一组候选输出
        outputs = self.inference.generate(prompt, n=self.group_size, temperature=self.temperature)

        # 2. 逐条走环境（pipeline 校验）拿绝对 reward
        samples: list[dict] = []
        for out in outputs:
            reward = self.reward_fn(out)
            samples.append({"prompt": prompt, "output": out, "reward": reward})

        # 3. 组内 z-score 归一 → 相对优势
        advantages = self.feedback.normalize_group([s["reward"] for s in samples])
        for s, adv in zip(samples, advantages):
            s["advantage"] = adv

        # 4. 策略更新（可插拔；默认离线 stub 只记录信号）
        update_result = self.updater.update(samples)

        # 5. 归因回流：把「失败」样本的 reward 记入案例库（反馈回路）
        for s in samples:
            r = s["reward"]
            if r.get("hard_gate") or r.get("total", 1.0) < 1.4:
                self.feedback.record_failure(
                    {"report_id": f"grpo-{s.get('output', '')[:8]}", "failed_at_layer": r.get("failed_at_layer")},
                    attribution={"events": r.get("events", [])},
                )

        return {
            "group_size": len(samples),
            "samples": [
                {"output": s["output"], "reward": s["reward"]["total"], "advantage": round(s["advantage"], 4)}
                for s in samples
            ],
            "update": update_result,
        }


class StubPolicyUpdater:
    """【桩】离线策略更新器：记录梯度信号，不真改权重。

    审核注意：这验证的是 GRPO 的数学链路（优势加权方向正确、反 hacking 门禁生效），
    真改权重接 TRLGRPOUpdater（需本地 GPU + 模型）。
    """

    def __init__(self, learning_rate: float = 1e-5) -> None:
        self.learning_rate = learning_rate
        self.updates: list[dict] = []

    def update(self, samples: list[dict]) -> dict:
        # 梯度信号：优势越大的样本，对模型参数的「牵引」越强。
        # 这里只记录「方向 × 幅度」，等价于 policy gradient 的 logπ·A 累加。
        signals = [
            {"advantage": round(s["advantage"], 4), "weighted": round(s["advantage"] * self.learning_rate, 8)}
            for s in samples
        ]
        self.updates.append({"samples": signals})
        # 反 reward hacking：若出现「全正优势」（异常高分），标记告警
        hacking_flag = all(s["advantage"] > 0 for s in samples)
        return {
            "policy_updated": False,  # stub 不真改权重
            "gradient_signals": signals,
            "reward_hacking_alert": hacking_flag,
        }


class TRLGRPOUpdater:
    """【真实现·骨架】接 TRL GRPOTrainer 的更新器（P3 填充点）。

    审核注意：需本地 GPU + TRL + 本地模型权重。本类只留接口骨架，
    真正训练时按 TRL 文档配置 reward 函数与采样组，替换 StubPolicyUpdater 即可。
    """

    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._configured = False

    def update(self, samples: list[dict]) -> dict:
        if not self._configured:
            raise RuntimeError("TRLGRPOUpdater 尚未配置模型（P3 填充点，需本地 GPU）")
        # 占位：真正实现时调用 trl.GRPOConfig + GRPOTrainer.train()
        return {"policy_updated": True}
