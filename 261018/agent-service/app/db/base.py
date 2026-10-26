"""SQLAlchemy 声明基类。

所有 ORM 模型继承自 ``Base``，Alembic 通过 ``Base.metadata`` 自动生成迁移脚本。
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """ORM 声明基类（SQLAlchemy 2.0 风格）。"""
