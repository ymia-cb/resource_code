"""C 层白名单匹配：数字 token 提取 + 格式归一 + 授权值集合匹配。

归一化只处理「同一个值的不同写法」（1,234.56 / ¥1234.56 / 1234.560 → 1234.56），
不处理「不同的说法」。修饰词（约/超过）不归一，转约束校验。
抄错和幻觉统一处理：都是「未授权的值」。
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# 数字 token：捕获整数/小数，含千分位
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

# 金额修饰符：¥ ￥ $ 元 万元 万 亿 等
_UNIT_SUFFIX = r"(元|万元|万|亿|元整|%)?"
_CURRENCY_PREFIX = r"[¥￥$]?"


def normalize(value: str) -> Decimal | None:
    """把「同一个值的不同写法」归一为 Decimal。

    1,234.56 / ¥1234.56 / 1234.560 → Decimal("1234.56")
    无法归一（如空串）返回 None。
    """
    s = value.strip()
    s = s.replace(",", "")          # 去千分位
    s = s.lstrip("¥￥$")             # 去货币前缀
    s = s.rstrip("元万元亿%")          # 去单位后缀（单位走约束校验，不进归一）
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def extract_numbers(text: str) -> list[str]:
    """扫描出所有数字 token（原始字符串形式）。"""
    return _NUMBER_RE.findall(text)


def check_authorized(text: str, authorized_values: set[Decimal]) -> list[str]:
    """扫描 text 里所有数字，归一后必须在授权集合里。返回未授权值列表。"""
    unauthorized: list[str] = []
    for token in extract_numbers(text):
        normalized = normalize(token)
        if normalized is None:
            continue
        if normalized not in authorized_values:
            unauthorized.append(token)
    return unauthorized
