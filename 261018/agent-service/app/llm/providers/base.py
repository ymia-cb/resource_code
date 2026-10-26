"""Provider 抽象接口。

所有模型 provider 实现同一接口，业务代码只依赖 ``Provider.chat``，
切换真实/ mock / 本地模型不改变调用方。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage


class Provider(ABC):
    """模型 Provider 抽象接口。"""

    @abstractmethod
    def chat(self, messages: list[BaseMessage]) -> str:
        """给定消息列表，返回模型文本回复。"""
        raise NotImplementedError
