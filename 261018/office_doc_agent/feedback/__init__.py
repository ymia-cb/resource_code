"""Stage 8：失败案例库 + RL 信号。

职责
----
把失败案例（带归因）归档，转成 RL 训练信号，驱动模板与契约迭代，防止「越做越差」。

关键决策（RL 信号设计，详见《沙箱·自动校验·RL 回流设计》）：
- 硬约束门：结构损坏/占位符残留/金额公式错误 → 直接置 0，一票否决。
- 分阶段奖励：R_terminal（交付1/人改0.5/拒收0）+ R_gate（每层+0.1封顶0.4）
  + R_repair（修+0.05/引入-0.05）+ R_regress（回退-0.2）+ R_cost（超时-0.01）。
- 归因分层：P0 金额/条款/越权 → 硬约束；P1 分页/图表/字体 → L3 视觉；P2 措辞/格式 → 终局分。
- 反 reward hacking：完整性/变更范围/隐藏门禁/独立通道四道防线。

桩 → 真实现
------------
- StubFeedback：空实现。
- RLFeedback：真实现，把 FailureCase（带 attribution/reward）写入持久化案例库（JSONL），
  并支持 GRPO 组内奖励归一（相对奖励）与 chosen/rejected 样本构建。

本地部署 + RL 训练（新增）
--------------------------
- 模型本地部署（vLLM/OpenAI 兼容）→ 推理采样一组候选 → 每个候选走 pipeline 校验得 reward。
- GRPO 需要「组内相对奖励」：reward 在组内做 z-score 归一，抑制绝对分数漂移。
- 案例库落 JSONL，训练管线（training 模块）离线消费。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class StubFeedback:
    """【桩】空实现。真实现见下方 RLFeedback。"""

    def record_failure(self, report: dict) -> dict:
        return {"report_id": report.get("report_id"), "status": "stub"}


class RLFeedback:
    """【真实现】失败案例库 + RL 信号分解 + GRPO 组内归一。

    审核注意：本模块负责「信号产出」——把一次校验结果转成结构化的
    (sample, reward) 对并落库；「训练」（策略梯度更新）由 training 模块消费
    案例库完成。二者解耦：信号可离线累积，训练可独立触发。
    """

    # 硬约束门：这些失败直接置 0，一票否决，不参与加权
    HARD_GATE_EVENTS = {"value_not_authorized", "ref_forbidden", "ref_unresolved"}

    def __init__(self, case_db_path: str | None = None) -> None:
        self._cases: list[dict] = []
        self._case_db_path = Path(case_db_path) if case_db_path else None

    # ---- 记录 ----
    def record_failure(self, report: dict, attribution: dict | None = None) -> dict:
        reward = self._compute_reward(report, attribution)
        case = {
            "case_id": report.get("report_id", ""),
            "failed_layer": report.get("failed_at_layer"),
            "attribution": attribution or {},
            "reward": reward,
        }
        self._cases.append(case)
        if self._case_db_path:
            self._append_to_db(case)
        return case

    def _append_to_db(self, case: dict) -> None:
        """把案例追加到 JSONL 案例库（每行一条，可增量累积）。"""
        self._case_db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._case_db_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # ---- 奖励分解 ----
    def _compute_reward(self, report: dict, attribution: dict | None) -> dict:
        """按 RL 信号设计分解奖励。

        硬约束门命中 → 总分 0；否则 R_terminal + R_gate + ... 加权。
        """
        events = (attribution or {}).get("events", [])
        if any(e in self.HARD_GATE_EVENTS for e in events):
            return {"hard_gate": True, "total": 0.0}

        # 终局奖励：交付 1 / 需人改 0.5 / 拒收 0
        r_terminal = 1.0 if report.get("overall") == "pass" else 0.5
        # 门禁奖励：每过一层 +0.1，封顶 0.4
        layers = report.get("layers", [])
        passed = sum(1 for l in layers if l.get("passed"))
        r_gate = min(passed * 0.1, 0.4)

        return {
            "hard_gate": False,
            "r_terminal": r_terminal,
            "r_gate": r_gate,
            "total": r_terminal + r_gate,
        }

    # ---- GRPO 组内归一 ----
    @staticmethod
    def normalize_group(rewards: list[dict]) -> list[float]:
        """GRPO 组内 z-score 归一：把一组候选的绝对奖励转成相对优势。

        相对奖励是 GRPO 的核心——组内均值为 0，标准差为 1，抑制「全组都高分」
        或「全组都低分」导致的绝对漂移。单样本（std=0）时全置 0（无相对信息）。
        """
        totals = [r.get("total", 0.0) for r in rewards]
        mean = sum(totals) / len(totals) if totals else 0.0
        var = sum((t - mean) ** 2 for t in totals) / len(totals) if totals else 0.0
        std = math.sqrt(var)
        if std == 0.0:
            return [0.0 for _ in totals]
        return [(t - mean) / std for t in totals]

    # ---- chosen / rejected 样本（供 DPO 备选算法） ----
    @staticmethod
    def build_pairs(samples: list[dict]) -> list[dict]:
        """从一组候选里抽 chosen（最高分）与 rejected（最低分）构成偏好对。

        审核注意：这是 DPO 的备用数据路径；GRPO 直接用 normalize_group 的连续优势。
        """
        if len(samples) < 2:
            return []
        ranked = sorted(samples, key=lambda s: s.get("reward", {}).get("total", 0.0))
        chosen = ranked[-1]
        rejected = ranked[0]
        return [{
            "prompt": chosen.get("prompt", ""),
            "chosen": chosen.get("output", ""),
            "rejected": rejected.get("output", ""),
        }]
