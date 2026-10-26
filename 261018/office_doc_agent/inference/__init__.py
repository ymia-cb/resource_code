"""本地模型推理：OpenAI 兼容协议（vLLM）。

职责
----
把「生成 IR 的 LLM 调用」收敛到本地推理服务，不依赖云端 API。
协议用 OpenAI 兼容的 /v1/chat/completions（vLLM / LM Studio / Ollama 的 OpenAI 端点都兼容）。

接口契约
--------
    class LocalInference(Protocol):
        def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]: ...

关键决策
--------
- base_url 默认 http://localhost:8000/v1（vLLM 默认端口），可用环境变量 LLM_BASE_URL 覆盖。
- 不引第三方 SDK：用 stdlib urllib 直接打 HTTP，避免 httpx/openai 依赖（轻量、可审计）。
- 采样 n>1 时用 temperature 控制多样性（GRPO 在线采样需要一组候选）。
- StubInference：离线桩，返回确定性的伪输出，供无 GPU 环境验证整条 RL 链路逻辑。

桩 → 真实现
------------
- StubInference：返回写死的候选（带序号），离线可测。
- VLLMInference：真实打 /v1/chat/completions，解析 choices[].message.content。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class LocalInference(Protocol):
    """本地推理抽象：给定 prompt，采样 n 条候选输出。"""

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        ...


class StubInference:
    """【桩】离线返回确定性候选，供无本地模型时验证 RL 链路。

    审核注意：候选输出故意做成「部分含违规值、部分合规」，这样 reward 归一
    才有区分度，能验证 GRPO「相对奖励」的数学逻辑，而不是恒等输出。
    """

    def __init__(self, base_url: str = "", model: str = "stub") -> None:
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        # 返回 n 条伪 IR 片段（JSON 字符串），模拟 LLM 采样的多样性
        return [f'{{"candidate": {i}, "text": "保单号：{{policy_no}} 保费：{{premium}}"}}'
                for i in range(n)]


class VLLMInference:
    """【真实现】OpenAI 兼容的本地推理（vLLM / LM Studio / Ollama OpenAI 端点）。

    依赖：仅 stdlib（urllib），无需 openai SDK。
    配置：
      LLM_BASE_URL  （默认 http://localhost:8000/v1）
      LLM_MODEL     （默认空，取服务默认模型）
      LLM_API_KEY   （可选，本地服务通常不需要）
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        payload = {
            "model": self.model or None,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
        }
        # 去掉 model=None，让服务用默认模型
        if payload["model"] is None:
            payload.pop("model")

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM 推理失败 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接本地推理服务 {self.base_url}: {exc.reason}") from exc

        return [choice["message"]["content"] for choice in body.get("choices", [])]
