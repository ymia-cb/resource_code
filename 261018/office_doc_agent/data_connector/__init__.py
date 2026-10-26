"""Stage 3：取数（数据接入）+ 口径断言。

职责
----
从 SQL/API 取数，在接入层做口径断言（口径/精度/时区/空值），返回口径校验后的数据集。
结构化数据是「唯一事实来源」——数字只能来自这里。

接口契约
--------
符合 interfaces.DataConnector Protocol：
    fetch(contract: DataContract) -> Dataset

关键决策
--------
- 口径四要素显式声明：caliber（口径）/ precision（精度）/ timezone（时区）/ null_policy（空值）。
- 空值策略区分 null / empty_string / zero 的语义，禁止混用（这是「数字对但含义错」的根因）。
- ACL 过滤在检索时做（索引层 pre-filter），RLS 在取数时做——顺序：ACL 先于采样，采样 n≥5 防小样本。

桩 → 真实现
------------
- StubDataConnector：返回写死数据集，供 MVP 空跑。
- SqlDataConnector / ApiDataConnector：真实现，取数后走口径断言。
  断言失败抛 DATA-001（口径/精度/时区/空值不符合契约）/ DATA-002（取数结果为空）。
"""
from __future__ import annotations

from typing import Any


class StubDataConnector:
    """【桩】返回写死数据集，供 MVP 空跑。真实现见下方 DataConnector 骨架。"""

    def fetch(self, contract: Any) -> Any:
        return {
            "policy_no": "P202610230001",
            "premium": "¥1,234.56",
        }


class DataConnector:
    """【真实现·半成品】取数 + 口径断言骨架。

    审核注意：真正的 SQL/API 取数尚未接入（需内部数据源），这里先落地
    「口径断言」这个纯逻辑骨架——它是数据契约的核心价值，不依赖 I/O 就能测。
    """

    def fetch(self, contract: Any, raw_data: dict | None = None) -> Any:
        data = raw_data if raw_data is not None else StubDataConnector().fetch(contract)

        if not data:
            # 错误码 DATA-002：取数结果为空
            raise ValueError("DATA-002: 取数结果为空")

        self._assert_caliber(contract, data)
        return data

    @staticmethod
    def _assert_caliber(contract: Any, data: dict) -> None:
        """口径断言：逐字段校验精度/时区/空值策略。"""
        if contract is None:
            return
        for field in contract.get("fields", []):
            name = field["name"]
            if name not in data:
                continue
            value = data[name]

            # 空值策略：forbidden 时不允许 None
            if value is None and field.get("null_policy") == "forbidden":
                # 错误码 DATA-001：空值策略违反
                raise ValueError(f"DATA-001: 字段 {name} 空值策略 forbidden 但值为空")

            # 精度校验：decimal 类型保留位数（骨架示例）
            if field.get("precision") is not None and isinstance(value, float):
                import decimal

                d = decimal.Decimal(str(value))
                if -d.as_tuple().exponent > field["precision"]:
                    raise ValueError(f"DATA-001: 字段 {name} 精度超限")
