"""Alembic 迁移环境。

关键约定（Spec 2.4）：
- 从 ``app.config`` 读取 MySQL 连接串（``.env``），不在 ini 里硬编码。
- ``target_metadata`` 指向 ORM 的 ``Base.metadata``，供 autogenerate 使用。
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# 使 app 包可导入（迁移脚本运行时 cwd 可能是项目根）
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_db_config
from app.db.base import Base
# 导入全部模型，确保 metadata 完整
import app.db.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 用 .env / 环境变量里的连接串覆盖 ini 中的 sqlalchemy.url
config.set_main_option("sqlalchemy.url", get_db_config().url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
