"""FastAPI 应用入口。

启动：uvicorn app.main:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, orders, health, knowledge, stream
from app.mcp.server import router as mcp_router

app = FastAPI(
    title="电商智能客服 Agent",
    version="0.1.0",
    description="多模态智能客服 Agent（竖切片 + 数据层 + 接口层）",
)

# CORS：前端知识库管理页跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # M1 放开，生产收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(orders.router)
app.include_router(knowledge.router)
app.include_router(stream.router)
app.include_router(mcp_router)

# 前端知识库管理页（单 HTML 静态托管，无构建依赖）
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/knowledge", include_in_schema=False)
    def knowledge_page():
        """知识库管理页入口。"""
        return FileResponse(_FRONTEND_DIR / "knowledge.html")

    @app.get("/", include_in_schema=False)
    @app.get("/chat", include_in_schema=False)
    def portal_page():
        """单页门户（智能客服 + 知识库管理）入口。"""
        return FileResponse(_FRONTEND_DIR / "index.html")
