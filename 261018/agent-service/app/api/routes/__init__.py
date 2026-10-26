"""路由包：汇总各模块路由。"""
from app.api.routes import chat, orders, health, knowledge

__all__ = ["chat", "orders", "health", "knowledge"]
