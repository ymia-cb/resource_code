"""配置加载：从环境变量 / .env 读取 MySQL 连接串。

约定（Spec 2.4）：连接串不在代码里硬编码，统一走 env.py 和本模块读取。
默认值便于本地 Docker MySQL（root/root@127.0.0.1:3306/ecommerce_agent）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 尝试加载项目根目录下的 .env（不引入额外依赖，手写极简解析）
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv() -> None:
    if not _ENV_FILE.exists():
        return
    for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        # 已存在的环境变量优先（不覆盖 shell 传入）
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    """MySQL 连接配置。"""

    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"

    @property
    def url(self) -> str:
        """返回 SQLAlchemy 连接串（sync，使用 pymysql 驱动）。"""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "root"),
            database=os.getenv("DB_NAME", "ecommerce_agent"),
            charset=os.getenv("DB_CHARSET", "utf8mb4"),
        )


def get_db_config() -> DatabaseConfig:
    return DatabaseConfig.from_env()


# 供 Alembic env.py 与业务代码共用的连接串入口
DATABASE_URL = get_db_config().url


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    """单个模型的 OpenAI 兼容配置。"""

    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0  # 读超时（整个请求）
    connect_timeout: float = 5.0  # 连接超时（单独，比读超时短，及时失败）
    max_retries: int = 2
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        """api_key 与 model 都配置才算启用。"""
        return bool(self.api_key and self.model)


@dataclass(frozen=True)
class LLMSettings:
    """双模型配置：小模型（分类/RAG/reflector）+ 大模型（planner/summarize）+ VLM。"""

    small: LLMConfig  # classify / rag / reflector
    large: LLMConfig  # planner / summarize
    vlm: LLMConfig | None = None  # image_understand（视觉模型）

    def provider_for(self, role: str) -> LLMConfig | None:
        """按 role 返回模型配置；未启用返回 None（降级 mock）。"""
        if role in ("classify", "rag", "reflector"):
            return self.small if self.small.enabled else None
        if role in ("planner", "summarize"):
            return self.large if self.large.enabled else None
        if role == "vlm":
            return self.vlm if (self.vlm and self.vlm.enabled) else None
        return None


def _llm_from_env(prefix: str) -> LLMConfig:
    return LLMConfig(
        base_url=os.getenv(f"{prefix}_BASE_URL", ""),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        model=os.getenv(f"{prefix}_MODEL", ""),
        timeout=float(os.getenv(f"{prefix}_TIMEOUT", "30")),
        connect_timeout=float(os.getenv(f"{prefix}_CONNECT_TIMEOUT", "5")),
        max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "2")),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
    )


def get_llm_settings() -> LLMSettings:
    """读取 LLM 配置。环境变量：

    - SMALL_LLM_BASE_URL / SMALL_LLM_API_KEY / SMALL_LLM_MODEL
    - LARGE_LLM_BASE_URL / LARGE_LLM_API_KEY / LARGE_LLM_MODEL
    - VLM_BASE_URL / VLM_API_KEY / VLM_MODEL（视觉模型，如 Qwen-VL / Gemini）

    未配置 api_key/model 时对应 provider 降级为 mock。
    """
    vlm = _llm_from_env("VLM")
    return LLMSettings(
        small=_llm_from_env("SMALL_LLM"),
        large=_llm_from_env("LARGE_LLM"),
        vlm=vlm if vlm.enabled else None,
    )


# ---------------------------------------------------------------------------
# MCP 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPConfig:
    """MCP 端点配置。"""

    api_token: str | None  # Bearer token（None = 不鉴权，仅本地开发）

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_token)

    @classmethod
    def from_env(cls) -> "MCPConfig":
        token = os.getenv("MCP_API_TOKEN")
        return cls(api_token=token or None)


def get_mcp_config() -> MCPConfig:
    return MCPConfig.from_env()


# ---------------------------------------------------------------------------
# Embedding / Milvus 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 模型配置（OpenAI 兼容 /embeddings 端点）。"""

    base_url: str
    api_key: str
    model: str
    dim: int  # 向量维度

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            model=os.getenv("EMBEDDING_MODEL", ""),
            dim=int(os.getenv("EMBEDDING_DIM", "384")),
        )


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接配置。"""

    host: str
    port: str
    collection: str

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        return cls(
            host=os.getenv("MILVUS_HOST", "127.0.0.1"),
            port=os.getenv("MILVUS_PORT", "19530"),
            collection=os.getenv("MILVUS_COLLECTION", "knowledge_chunks"),
        )


@dataclass(frozen=True)
class MinerUConfig:
    """MinerU 在线 API 配置（Spec 10.2：纯网络请求，抽象 base_url + token）。

    enabled = 配置了 api_key（无 key 时降级本地解析，保证离线可跑）。
    """

    base_url: str
    api_key: str
    timeout: float = 60.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "MinerUConfig":
        return cls(
            base_url=os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4"),
            api_key=os.getenv("MINERU_API_KEY", ""),
            timeout=float(os.getenv("MINERU_TIMEOUT", "60")),
        )


def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig.from_env()


def get_milvus_config() -> MilvusConfig:
    return MilvusConfig.from_env()


def get_mineru_config() -> MinerUConfig:
    return MinerUConfig.from_env()


# ---------------------------------------------------------------------------
# MinIO 配置（对象存储：图片/附件）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinioConfig:
    """MinIO 对象存储配置（业务附件，独立于 Milvus 自带的 minio）。

    enabled = 配置了 endpoint + access_key + secret_key。
    未配置时降级本地磁盘目录（data/uploads/），保证离线可跑。
    """

    endpoint: str  # 如 127.0.0.1:9002
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint and self.access_key and self.secret_key)

    @classmethod
    def from_env(cls) -> "MinioConfig":
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9002"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "agentadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "agentadmin123"),
            bucket=os.getenv("MINIO_BUCKET", "knowledge-assets"),
            secure=os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes"),
        )


def get_minio_config() -> MinioConfig:
    return MinioConfig.from_env()
