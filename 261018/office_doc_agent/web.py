"""FastAPI 看板后端：把 Pipeline 全链路暴露成 HTTP API。

职责
----
把「生成 → 中间表示校验（安全主拦截）→ 渲染 → 四层校验」暴露成 REST，
供前端单页看板调用。同时提供 claim 四通道队列 + 审计日志查询。

接口清单
--------
- POST /api/generate       触发全流程，返回 docx 路径 + 四层校验报告 + claims
- GET  /api/claims         claim 四通道队列（auto_block/auto_pass/human/sample）
- GET  /api/audit          审计日志（按时间倒序）
- GET  /api/download/{id}  下载产物 docx
- GET  /                   看板页面（静态）
- GET  /api/health         健康检查

关键决策
--------
- 服务状态存内存（MVP），正式版换 SQLite/Redis。
- 每次 generate 复用真实模块（DefaultGenerator/DocxRenderer/DocxValidator），
  并把 security 白名单 + ClaimQueue 接进链路，产出真实 claim。
- 审计日志记录「哪次生成、哪层失败、多少 claim、什么通道」，供审核员回查。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .data_connector import StubDataConnector
from .feedback import RLFeedback
from .generator import DefaultGenerator
from .inference import StubInference
from .pipeline import Pipeline
from .renderer import DocxRenderer
from .security.channel import Channel, ClaimQueue
from .security.claim import Claim
from .security.whitelist import normalize
from .spec_loader import StubSpecLoader
from .template_manager import StubTemplateManager
from .training import GRPOTrainer, StubPolicyUpdater
from .validator import DocxValidator

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
_WEB_DIR = Path(__file__).resolve().parent / "web"


@dataclass
class AuditRecord:
    """一条审计记录：某次生成的全链路摘要。"""

    run_id: str
    doc_id: str
    ts: float
    success: bool
    failed_stage: str | None
    claim_count: int
    layer_results: list[dict]
    claims: list[dict]


@dataclass
class AppState:
    """看板服务的内存状态。"""

    runs: list[AuditRecord] = field(default_factory=list)
    queue: ClaimQueue = field(default_factory=lambda: ClaimQueue())


def create_app(state: AppState | None = None) -> FastAPI:
    """构建 FastAPI 应用（依赖注入 state，便于测试）。"""
    app = FastAPI(title="办公文档 Agent 看板", version="1.0")
    state = state or AppState()

    # 组装真实链路
    generator = DefaultGenerator()
    renderer = DocxRenderer()
    validator = DocxValidator()
    spec_loader = StubSpecLoader()
    template_manager = StubTemplateManager()
    data_connector = StubDataConnector()
    feedback = RLFeedback()

    def _build_pipeline(spec: dict, data: dict) -> Pipeline:
        """按数据契约推导授权值集合 + 数值型字段，装配安全白名单。"""
        authorized: set = set()
        numeric_fields: set[str] = set()
        for field_def in spec.get("field_list", []):
            name = field_def.get("name")
            ftype = field_def.get("type", "string")
            if ftype in ("decimal", "money", "number", "float", "int"):
                numeric_fields.add(name)
                raw = data.get(name)
                if raw is not None:
                    n = normalize(str(raw))
                    if n is not None:
                        authorized.add(n)
        return Pipeline(
            generator=generator,
            renderer=renderer,
            validator=validator,
            authorized_values=authorized if authorized else None,
            numeric_fields=numeric_fields or None,
        )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "runs": len(state.runs)}

    @app.post("/api/generate")
    def generate(payload: dict[str, Any] | None = None) -> dict:
        """触发一次全链路生成，返回四层校验 + claims + 审计摘要。"""
        payload = payload or {}
        doc_id = payload.get("doc_id") or f"doc-{uuid.uuid4().hex[:8]}"

        # 组装输入（MVP 用桩数据；payload 可覆盖 data 字段）
        spec = spec_loader.load(None)
        # 若 payload 带 field_list，注入到 spec（供白名单推导 + 业务校验）
        if payload.get("field_list"):
            spec = {**spec, "field_list": payload["field_list"]}
        template = template_manager.load(payload.get("template_id") or "contract-v1.0")
        data = {**data_connector.fetch(None), **(payload.get("data") or {})}

        pipeline = _build_pipeline(spec, data)
        result = pipeline.run(spec, template, data, _ARTIFACTS_DIR, doc_id=doc_id)

        # claims 进队列（四通道）
        claims_payload: list[dict] = []
        for c in result.claims:
            channel = state.queue.route(c)
            claims_payload.append(_claim_to_dict(c, channel))

        # 审计记录
        layers = (result.report or {}).get("layers", [])
        record = AuditRecord(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            doc_id=doc_id,
            ts=time.time(),
            success=result.success,
            failed_stage=result.failed_stage,
            claim_count=len(result.claims),
            layer_results=[
                {"layer": l["layer"], "passed": l["passed"], "issues": l.get("issues", [])}
                for l in layers
            ],
            claims=claims_payload,
        )
        state.runs.append(record)

        # RL 信号：失败案例回流（归因：claim 事件类型集合）
        attribution = {
            "events": [c.event_type.value for c in result.claims],
        }
        rl = feedback.record_failure(result.report or {}, attribution)

        return {
            "run_id": record.run_id,
            "doc_id": doc_id,
            "success": result.success,
            "failed_stage": result.failed_stage,
            "docx_path": str(result.docx_path) if result.docx_path else None,
            "report": result.report,
            "claims": claims_payload,
            "claim_summary": {
                "auto_block": len(state.queue.auto_blocked()),
                "auto_pass": len(state.queue.auto_passed()),
                "human": len(state.queue.human_queue()),
            },
            "rl": rl,
        }

    @app.get("/api/claims")
    def claims() -> dict:
        """按四通道返回当前队列（供看板人工裁决）。"""
        return {
            "auto_block": [_claim_to_dict(c, Channel.AUTO_BLOCK) for c in state.queue.auto_blocked()],
            "auto_pass": [_claim_to_dict(c, Channel.AUTO_PASS) for c in state.queue.auto_passed()],
            "human": [_claim_to_dict(c, Channel.HUMAN) for c in state.queue.human_queue()],
        }

    @app.get("/api/audit")
    def audit() -> dict:
        """审计日志（时间倒序）。"""
        runs = sorted(state.runs, key=lambda r: r.ts, reverse=True)
        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "doc_id": r.doc_id,
                    "ts": r.ts,
                    "success": r.success,
                    "failed_stage": r.failed_stage,
                    "claim_count": r.claim_count,
                    "layers": r.layer_results,
                }
                for r in runs
            ]
        }

    @app.get("/api/download/{doc_id}")
    def download(doc_id: str) -> FileResponse:
        """下载产物 docx。"""
        path = _ARTIFACTS_DIR / f"{doc_id}.docx"
        if not path.exists():
            raise HTTPException(status_code=404, detail="产物不存在")
        return FileResponse(path, filename=f"{doc_id}.docx")

    @app.post("/api/train")
    def train(payload: dict[str, Any] | None = None) -> dict:
        """触发一步 GRPO 训练（本地部署 + 组内相对奖励）。

        审核注意：MVP 用 StubInference（离线）跑通整条 GRPO 数学链路；
        接真实本地模型时把 StubInference 换成 VLLMInference（LLM_BASE_URL 指向 vLLM）。
        """
        payload = payload or {}
        prompt = payload.get("prompt") or "生成一份合同中间表示（IR），保费字段用 {{premium}} 占位符"

        # 环境（reward_fn）：把候选 IR 走 pipeline 校验，产出真实 reward。
        # MVP 用桩推理的候选不直接映射到 pipeline（候选是伪文本），这里用「序号映射」
        # 模拟校验打分，验证归一/更新链路；真接模型时 reward_fn 应真正调用 pipeline。
        def reward_fn(output: str) -> dict:
            import re
            m = re.search(r'"candidate": (\d+)', output)
            idx = int(m.group(1)) if m else 0
            if idx >= 3:
                return {"hard_gate": False, "total": 1.4, "events": []}
            if idx == 2:
                return {"hard_gate": False, "total": 0.9, "events": []}
            return {"hard_gate": False, "total": 0.4, "events": ["value_near_miss"]}

        trainer = GRPOTrainer(
            inference=StubInference(),
            reward_fn=reward_fn,
            updater=StubPolicyUpdater(),
            group_size=payload.get("group_size", 4),
            feedback=RLFeedback(),
        )
        result = trainer.step(prompt)
        return result

    # 静态页面
    _WEB_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = _WEB_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<h1>办公文档 Agent 看板</h1><p>前端页面未生成，请先构建 index.html</p>"

    return app


def _claim_to_dict(c: Claim, channel: Channel) -> dict:
    """Claim → 可序列化 dict（枚举转字符串）。"""
    return {
        "event_type": c.event_type.value,
        "layer": c.layer.value,
        "span": list(c.span),
        "target_field": c.target_field,
        "rule": c.rule,
        "detail": c.detail,
        "expected": c.expected,
        "actual": c.actual,
        "doc_id": c.doc_id,
        "channel": channel.value,
    }


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("office_doc_agent.web:app", host="127.0.0.1", port=8128, reload=True)
