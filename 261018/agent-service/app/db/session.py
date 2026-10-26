"""数据库引擎与 session 工厂。

- ``engine``：全局同步引擎（pymysql）
- ``SessionLocal``：业务代码使用的 sessionmaker
- ``get_session``：FastAPI 依赖注入用上下文管理器
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_db_config

# pool_pre_ping：避免 MySQL 长连接被服务端断开后复用报错
_engine = create_engine(
    get_db_config().url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def get_engine():
    return _engine


@contextmanager
def get_session() -> Iterator[Session]:
    """上下文管理器：保证异常回滚、正常提交、最终关闭。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
