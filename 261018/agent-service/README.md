# 电商智能客服 Agent —— 竖切片

> 阶段：Phase 2 竖切片（可运行最小链路）+ 数据层 + 接口层
> 前置设计：`../docs/多模态智能客服Agent设计概要.md`、`../docs/电商智能客服Agent开发Spec.md`

## 当前已跑通

- 文本 → 意图分类 → RAG 直答 → checkpoint 会话记忆
- 分支：`simple`（RAG 直答）、`complex`（留桩转人工）、`human`（兜底）
- 评测：golden set 8 条，意图分类 + 检索命中 100%

## 当前已跑通（数据层）

- MySQL ORM 模型 + Alembic 迁移（真实 MySQL 建表 6 张）
- 取消订单确定性规则引擎 `check_cancelable`（不交给模型）
- `cancel_order` 联动：状态更新 + 退款 + 库存回补 + 优惠券回退 + 审计（同事务）
- 数据层单元测试 9 条全部通过

## 当前已跑通（接口层）

- FastAPI 端点：`/health`、`/chat`、`/session/*`、`/orders/*`（7 个端点）
- `/chat` 对接 graph（多轮记忆按 thread_id 隔离）
- `/orders/cancel` 二次确认 + 规则引擎前置校验
- 接口层测试 8 条全部通过

## 当前已跑通（工具安全层）

- 五道安全闸门统一执行框架 `execute_tool`（白名单 → 参数校验 → 权限 → 幂等 → 审计）
- `ToolSpec` 工具注册表 + 4 个工具白名单（query_order/query_coupon/query_product/cancel_order）
- 幂等键防重复执行（`idempotency_records` 表 + 唯一索引并发安全）
- 审计职责上移：业务函数专注状态变更，审计统一由闸门⑤负责
- 五道闸门测试 9 条全部通过

## 当前已跑通（自研 MySQL CheckpointSaver）

- 实现 `BaseCheckpointSaver` 同步接口（get_tuple/list/put/put_writes/delete_thread）
- 表结构对齐官方 SqliteSaver（checkpoints + checkpoint_writes，含 checkpoint_ns/type/idx）
- 通过**对拍测试**：同一图分别跑 SqliteSaver 与 MySQLSaver，消息记忆序列完全一致
- 修复消息通道缺陷：节点回复现回写 `messages`（AIMessage），checkpoint 历史完整
- 对拍测试 5 条全部通过

## 当前已跑通（高风险写工具）

- `apply_refund`（退款）/ `apply_return`（退货）/ `update_address`（改地址）
- 工具白名单从 4 个扩展到 7 个
- 金额超阈值（¥1000）自动转人工审批（`NeedApproval`）
- 状态校验：退款需已付款、退货需已发货、改址仅待发货
- 写工具测试 11 条全部通过

## 当前已跑通（复杂推理 Plan-Execute-Reflect）

- `planner → executor → reflector → summarize` 多步推理循环
- 复杂意图（取消/退款/退货/改址）自动拆解为「查订单 → 执行写操作」多步
- Reflector 审视结果：任一步出错 / 大额 / 步数超限 → 转人工
- 分类修正：政策咨询走 RAG 直答，操作意图走复杂推理
- 复杂推理测试 6 条全部通过

## 当前已跑通（真实 LLM 接入）

- `Provider` 抽象接口 + `OpenAICompatProvider`（纯 httpx 调 `/chat/completions`）
- 模型路由按 role 分流：小模型（classify/rag/reflector）+ 大模型（planner/summarize）
- 有 key 走真实模型（OpenAI/DeepSeek/Qwen 等兼容接口），无 key 自动降级 mock
- 配置项见 `.env.example`（`SMALL_LLM_*` / `LARGE_LLM_*`）
- LLM 接入测试 5 条全部通过

## 当前已跑通（降级策略）

- `app/llm/degradation.py`：`DegradeReason`（not_configured/provider_failed/breaker_open）+ `should_degrade_safely`（classify/rag/reflector 可安全降级，planner/summarize/vlm 不可安全降级）+ `record_degradation`（降级统一记录：结构化日志 + 指标累加，**绝不静默降级**）
- `router.llm_with_status()`：返回 `(text, degraded)` 降级标记，上层可感知「真实模型答案 vs 降级答案」；`llm()` 保持旧签名兼容
- 消除静默降级：未配置 key（`NOT_CONFIGURED`）路径现在也记录降级日志 + 指标，之前是完全静默的
- `rag_node` 降级安全修复：真实模型调用失败时降级到**基于检索的确定性 mock_generate**（而非把用户问题原文回显当答案），并打 `degraded` 标记
- `AgentState.degraded` + `ChatResponse.degraded` 字段，降级态对外可感知
- 降级测试 7 条全部通过（全量 158 条）

## 当前已跑通（命令编排 + 工具隔离 + 幂等 + 矛盾检测）

- `app/commands/`：**命令模型**（三级原子性 single/atomic/partial）+ 命令目录（`cancel_order` 是强关联聚合命令）+ 矛盾检测（`detect_conflicts`）+ **编排器**（依赖拓扑分层 + 只读并发 + 写串行）
- **核心原则**：模型永不负责拼事务，事务边界由服务端命令定义；模型只「选命令」，Agent 按依赖/风险确定性编排
- **工具按 Skill 隔离**：`ToolSpec` 新增 `skill` 归属 + `execute_tool` 加 skill 级动态白名单闸门（跨 skill 调用拒绝；未加载 skill 兼容放行）
- **幂等重构**：幂等命中改**正常返回 + `idempotent_replay` 标记**（不再抛异常）；`ToolSpec.idempotent_keys` 声明幂等键字段（`update_address` 含 `address`，实现「同地址去重、异地址重写」）
- **并发去重**：写操作 `FOR UPDATE` 行锁锁订单行 + 单次写事务（非真长事务），`IntegrityError` 兜底回读已有结果
- **命令/步骤状态持久化**：`CommandExecution`（命令级幂等键）+ `CommandStep`（步骤级 pending/success/failed），供审计/恢复/排查
- **矛盾请求约束**：提示词 + planner 双重拦截「取消订单并继续发货」类互斥请求 → 拒绝并要求澄清，不自行选结果
- **部分完成逐项告知**：`summarize` 弱关联批量命令部分失败时逐项列出「✅成功/❌失败」
- 命令编排测试 14 条全部通过（全量 172 条）

## 当前已跑通（渐进披露 Skill + 多领域多子问题编排）

- `app/skills/`：`SKILL_CATALOG`（order/product 两 skill，含 code/description/guidance/tools）+ `render_index`（skill 索引）+ `render_skill_prompt`（未加载→路由提示+索引；已加载→该 skill guidance）+ `render_support_prompt`（含矛盾请求约束）
- `load_skill` 工具：渐进披露入口，无 skill 归属（任何状态可用），返回完整说明；`load_skill_node` + 图路由（未加载 skill 的 simple/complex → 先 load_skill 再分流）
- `app/commands/multi_query.py`：**多领域多子问题编排**——拆解（领域关键词聚类）→ 依赖图（topo_sort）→ 执行（无依赖只读并发 + 并发上限 `max_concurrency` + 单查询超时 + 写命令串行）→ 汇总（按领域汇总 + 保留工具证据 + 局部失败如实告知「哪部分无法确认」）
- **写操作单独控制流** `execute_writes`：先查依赖/冲突/确认，业务独立且安全才并行，全成全败走服务端聚合命令+事务；部分失败逐项告知
- 多问题编排测试 12 条 + Skill 测试 12 条全部通过（全量 196 条）

## 当前已跑通（Skill 业务类型 catalog + definite + Agent 中间件）

- `app/skills/`：三层结构——`BusinessType` 枚举（query/write/knowledge/multimodal）+ `TypeDefinition`（**definite**，定义某业务类型的语义 + 该类型具体用到的工具白名单 + 行为约束 needs_confirm/read_only/timeout_seconds）+ `SKILL_CATALOG`（order/product/policy/vision 四领域，各标注业务类型）
- **「业务类型 → 工具」由 definite 定义**（`TypeDefinition.tools`），`tool_type`/`tools_of_type` 反查归属；`tools_for_skill` 返回「领域业务类型 tools + 跨类型补充工具」（`SKILL_EXTRA_TOOLS`，如 order 写前开放 query_order/query_coupon，但不含 query_product）
- `app/middleware/`：**Agent 中间件层**——`Middleware` 基类 + `MiddlewareChain`（before/after_llm + before/after_tool 四钩子）+ `get_default_chain` 惰性组装
  - `SkillDisclosureMiddleware`：渐进披露抽成中间件（before_llm 按 active_skill 注入/去重提示词；before_tool 做 skill 隔离第一道拦截）
  - `ToolTimeoutMiddleware`：工具调用超时按业务类型差异化（query 10s / write 15s / multimodal 20s），`run_tool_with_timeout` 用 ThreadPoolExecutor 超时保护，超时返回稳定 `E_TOOL_TIMEOUT`
- executor 经中间件 before_tool 校验 + `run_tool_with_timeout` 包裹执行；router 的 `llm_with_status` 注入 before_llm
- 中间件测试 13 条 + Skill 测试 12 条全部通过（全量 215 条，含 test_stream 补装 pytest-asyncio 后 6 条）

## 当前已跑通（细粒度五类意图路由 + 多步规划）

- **意图枚举从粗粒度三分类细化**：`intent` 从 `simple/complex/human` 扩展为 `knowledge/query/multi_step/clarify/human`，并新增 `sub_intent`（写操作细分 cancel/refund/return/address）
  - `knowledge`：政策/平台规则等知识库问答 → **RAG + LLM 直答**（不强制 load_skill，policy 属 KNOWLEDGE 类型无需工具）
  - `query`：物流/订单状态一步查询 → **query 节点轻量路径**（单次工具调用直接返回，不经 planner/executor/reflector 重循环）
  - `multi_step`：多步读写混合 + 顺序/条件逻辑 → **planner 重推理循环**（Plan-Execute-Reflect，支持条件分支）
  - `clarify`：信息不足需澄清 → **clarify 节点反问补充**（如缺订单号，不终止会话 needs_clarify=True）
  - `human`：无法处理/矛盾请求 → 转人工（终止会话）
- `app/graph/nodes/query.py`：一步查询节点，确定性选工具（物流→query_logistics，其余→query_order），缺订单号转澄清，结果直接格式化返回
- `app/graph/nodes/clarify.py`：澄清节点，按 sub_intent 生成针对性反问（如「请提供订单号」），与 human 区分（clarify 不终止、human 终止）
- **新增 `query_logistics` 工具**：查订单物流状态（M1 简化按订单状态推导物流摘要），加入 QUERY 类型 + registry（工具 9→10）
- `classify` 返回四元组 `(intent, skill, sub_intent, confidence)`，mock_classify 六类关键词规则 + 真实模型提示词同步扩展
- 路由测试 16 条（test_routing.py：六类分类 + query 单步 + clarify 反问 + 图级冒烟）全部通过（全量 231 条）

## 当前已跑通（多步规划 + 条件分支）

- **Plan-Execute-Reflect 触发条件从「写操作」改为「多步操作 + 顺序逻辑」**：`intent=write` 重命名为 `multi_step`，触发场景是用户描述包含多步读写操作且可能带顺序/条件逻辑，而非单纯的写操作
- **planner 从硬编码二步模板改为多步动态规划**（`app/graph/nodes/planner.py`）：
  - 识别用户描述中的**多个读写操作意图**（写命令 + 查询命令）
  - 识别**条件分支**（「如果…就…」「没发货就…，发货了就…」）→ 产出带 `condition` 字段的 plan
  - 命令目录补 `depends_on`（写命令依赖先查订单状态），构建真实依赖图
- **条件分支支持**（`app/graph/nodes/executor.py`）：
  - plan step 带 `condition`（如 `{"field":"status","op":"==","value":"pending_ship"}`）
  - executor 执行前检查前面查询结果，条件不满足 → 跳过该步（标记 `skipped`）
  - `_condition_met` 保守策略：找不到查询结果时跳过（避免错误执行写操作）
- **矛盾检测与条件分支区分**（关键设计）：条件分支场景（「没发货就取消，发货了就改地址」）中互斥命令是互斥条件下的分支，**不是矛盾**，planner 跳过矛盾检测；非条件场景的互斥（「取消订单并继续发货」）仍拒绝并要求澄清
- **summarize 识别条件分支跳过**：全部条件分支跳过时，说明「当前订单状态不满足操作条件」而非笼统「处理完成」
- 多步规划 + 条件分支测试 9 条（test_routing.py：条件识别 + 分支拆解 + executor 条件判断 + 图级条件分支端到端）全部通过（全量 240 条）

## 当前已跑通（MCP Streamable HTTP）

- 端点 `/mcp`，手写轻量 JSON-RPC 实现（不依赖 mcp SDK）
- 支持 `initialize` / `tools/list` / `tools/call` / `ping`
- 暴露 9 个工具（经 `execute_tool` 五道闸门 + skill 级动态白名单执行）
- Bearer token 鉴权（`MCP_API_TOKEN` 配置）
- MCP 端点测试 8 条全部通过

## 当前已跑通（VLM 图像理解）

- `image_understand` 工具：识别商品 / 诊断截图 / 解析票据
- 强制结构化 JSON 输出 + confidence，低于 0.7 转人工
- 走 `vlm` 通道（`VLM_*` 配置，如 Qwen-VL / Gemini），无 key 降级 mock
- 多模态消息（text + image_url）经 OpenAI 兼容接口透传
- VLM 测试 9 条全部通过

## 当前已跑通（RAG 接 Milvus）

- `embedding` 向量化：有 key 走 OpenAI 兼容 `/embeddings`，无 key 哈希向量兜底（确定性 + L2 归一化）
- `milvus_client`：MilvusClient 新 API，显式 schema（varchar 主键 + FLOAT_VECTOR）+ AUTOINDEX 向量索引 + load
- `retriever` 检索：优先 Milvus 向量检索，降级内存关键词；category 标量过滤
- 关键修复：**建向量索引 + load_collection**（此前缺索引导致 load 失败 → count 恒 0 → 检索被跳过）
- 阈值策略：真实 embedding 用 COSINE 相似度阈值过滤，哈希向量兜底只做 category 过滤 + top_k
- RAG 测试 12 条全部通过（全量 82 条）

## 当前已跑通（前端知识库管理页 + MinerU 文档导入）

- `KnowledgeDoc` 表（Alembic 迁移）+ 状态机 `uploaded → queued → parsing → done → indexed / failed`
- `mineru_client`：MinerU 在线 API 客户端（抽象 base_url + token，可平滑切私有化）；无 key 时文本类（md/txt）本地降级解析，二进制抛 `MinerUError`
- `chunker`：Markdown 按标题语义切块（保留标题上下文 + overlap）
- `importer`：导入编排（解析 → 切块 → 向量化 → 入库 Milvus → 更新状态），删除联动删向量（`doc_{id}_` 前缀）
- 6 个 API 端点：`/knowledge/import`、`/knowledge/docs`、`/knowledge/docs/{id}`、confirm、reparse、delete
- 前端单 HTML 页面 `/knowledge`（拖拽上传 + 状态徽标 + Markdown 预览 + 确认入库/删除），无构建依赖
- 导入进度条 + 百分比（`progress` 字段 0-100，随状态推进：queued 10 → parsing 30-95 → done 95 → indexed 100，失败红色展示）
- 知识库导入测试 23 条全部通过（全量 105 条）

## 当前已跑通（健壮性：退避重试 / 熔断 / 稳定错误码 / Run 追踪）

- `app/observability/`：错误分类（timeout/rate_limited/server_error/**auth_error**/**config_error**/client_error/network/tool_error）+ 稳定错误码（E_MODEL_UNAVAILABLE/E_RATE_LIMITED/E_TIMEOUT/E_TOOL_FAILED/E_AUTH_FAILED/E_CONFIG_ERROR/E_NEED_RETRY/E_INTERNAL）+ 结构化日志（**内部保留底层原因**：原始异常类型/HTTP 状态码/供应商/模型/耗时/重试次数，只进日志；用户只见稳定错误码+安全提示，绝不泄露堆栈/api_key/供应商内部信息）
- `classify_exception`：底层异常细分类——401/403→**鉴权失败**、400/404/422→**配置错误**、429→限流、5xx→服务端错误、超时→超时、连接失败→网络；并提取状态码供诊断
- `openai_compat`：连接超时/读超时分开；**带随机抖动的指数退避重试**，仅对可重试错误（网络/超时/5xx/429），**鉴权/配置/4xx 立即失败不重试**；`ProviderError` 携带 category/code/retries/exception_type/status_code
- `app/observability/breaker.py`：自研进程内熔断器（连续失败达阈值打开、冷却后半开探测）+ 指标计数器（超时/错误/熔断/降级率，结构化日志输出）
- `app/db/models/agent_run.py`：`AgentRun`（run_id/thread_id/status/error_code）+ `ToolExecution`（工具执行状态关联 run_id），写操作靠幂等键天然「重试跳过已执行、不重放副作用」
- `/chat`：每次请求一个 run_id，异常时返回稳定错误码 + `needs_human` + `retryable`，明确告知无法可靠完成并提供稍后重试/转人工选项
- 健壮性测试 22 条全部通过（全量 151 条）

## 当前已跑通（事实校验 Grounding：用工具结果校验回答）

- `app/graph/grounding.py`：JSON Schema 定义业务实体（`GroundedEntity`/`GroundedAnswer`，类型限定 order_id/amount/status/coupon_code/refund_amount）+ `extract_entities`（从回答文本/JSON 提取实体）+ `build_fact_source`（从工具成功结果构建事实源）+ `verify_entities`（核对实体是否在事实源，识别幻觉）+ `build_grounding_prompt`（提示词约束大模型输出结构化格式）
- `summarize_node`：生成回答后提取实体，用「工具成功结果」做事实校验，回答中出现不在事实源里的业务实体（疑似幻觉）→ 转人工
- `rag_node`：有真实大模型时走 `router.llm` + grounding 提示词约束输出格式 + 实体校验；无 key 降级 mock 保持原行为
- `execute_tool`：写操作结果注入 `order_id`（供事实源 ground truth）
- 分层校验 `verify_grounding`：**主通道**模型自报 entities 优先核对；**兜底**仅当 entities 为空时从正文文本提取核对（抓「自报干净但正文夹带幻觉」），兜底抓到幻觉 → 转人工
- 事实校验测试 15 条全部通过（全量 136 条）

## 当前已跑通（RAG 切分优化 + PDF 图片表格 + MinIO）

- `app/knowledge/chunker.py`：**表格原子性**（Markdown 表格整块不硬切）+ **父级标题累积上下文**（子块携带父级标题）+ 超长段落软切 + `extract_images`（提取图片引用）
- `app/storage/minio_client.py`：MinIO 对象存储上传/下载/删除（SDK 7.2.20），无配置自动降级本地磁盘 `data/uploads/`
- `app/knowledge/describe.py`：`describe_image`（VLM 把图片转文字描述，无 key 降级 alt 占位）+ `generate_filename`（大模型生成规范文件名，防用户随意命名，降级标题+时间戳）
- `app/knowledge/importer.py`：`process_images`（提取图片 → 下载 → 上传 MinIO → VLM 描述 → 图片 URL 填回 md），`create_document` 文本类解析后覆盖用户文件名为大模型生成名
- `config.py`：新增 `MinioConfig`（endpoint/access_key/secret_key/bucket/secure）
- 独立 MinIO 容器 `agent-minio`（端口 9002 API / 9003 控制台，凭证 agentadmin/agentadmin123）
- 存储测试 10 条全部通过（全量 146 条）

## 当前已跑通（异步 SSE 流式 + Redis 推流 + 前端门户）

- **异步流式架构**：`POST /chat/stream/submit` 提交即返回 `message_id`（不阻塞）；
  后台 `asyncio` 协程跑 `graph.stream`（节点级流式，同步 saver 经线程桥接不阻塞事件循环），
  逐节点 `PUBLISH` chunk 到 Redis，最终完整结果落库 `chat_messages` 表。
- **Redis 仅推流**：`app/stream/pubsub.py` 封装 Redis Pub/Sub（channel=`thread_id:message_id`），
  配 `REDIS_URL` 走真 Redis，未配自动降级进程内内存 pub/sub（离线可跑）。
- **断线/刷新恢复**：前端刷新后调 `GET /threads/{id}/messages` 拉历史，`GET /messages/{id}`
  拉完整结果兜底（结果落库为真源，Redis chunk 可丢）。
- **前端单页门户** `frontend/index.html`（`/` 与 `/chat` 入口）：左侧导航（💬 智能客服 /
  📚 知识库管理），会话列表 + 气泡 + 打字机动画 + 溯源折叠 + 转人工/降级/错误徽标 +
  SSE 订阅 + 断线自动重连。知识库管理页 `knowledge.html` 经 iframe 嵌入。
- 流式链路测试 6 条全部通过（全量 202 条）。

## 依赖版本

| 包 | 版本 |
|---|---|
| langgraph | 1.2.16 |
| langgraph-checkpoint-sqlite | 3.1.2 |
| langchain-core | 1.6.12 |
| pydantic | 2.13.5 |
| sqlalchemy | 2.1.5 |
| pymysql | 1.2.3 |
| alembic | 1.20.0 |
| fastapi | 0.141.1 |
| pymilvus | 3.0.2 |
| python-multipart | 0.0.32 |
| uvicorn | 0.54.0 |

## 运行

```bash
# 单条消息
python run.py "退款政策是什么"

# 多轮记忆演示
python run.py --demo

# 评测
python eval.py

# 数据层测试（需本机 MySQL）
python -m pytest tests/test_data_layer.py -v

# 接口层测试
python -m pytest tests/test_api.py -v

# 五道闸门测试
python -m pytest tests/test_tool_gates.py -v

# MySQL CheckpointSaver 对拍测试
python -m pytest tests/test_mysql_saver.py -v

# 高风险写工具测试
python -m pytest tests/test_write_tools.py -v

# 复杂推理测试
python -m pytest tests/test_complex_reasoning.py -v

# LLM 接入测试
python -m pytest tests/test_llm.py -v

# MCP 端点测试
python -m pytest tests/test_mcp.py -v

# VLM 图像理解测试
python -m pytest tests/test_vision.py -v

# RAG / Milvus 向量检索测试（Milvus 未运行时部分 skip）
python -m pytest tests/test_rag.py -v

# 知识库文档导入测试（MinerU 集成 + 前端 API）
python -m pytest tests/test_knowledge_import.py -v

# 健壮性测试（退避重试/熔断/错误码/Run 追踪）
python -m pytest tests/test_resilience.py -v

# 事实校验测试（grounding：工具结果校验回答）
python -m pytest tests/test_grounding.py -v

# 存储测试（MinIO 上传/下载/删除 + 图片处理）
python -m pytest tests/test_storage.py -v

# 全量测试
python -m pytest tests/ -v

# 启动 HTTP 服务
uvicorn app.main:app --reload   # 文档 /docs

# 知识库管理页（前端）
# 启动后浏览器访问 http://127.0.0.1:8000/knowledge

# 单页门户（智能客服 + 知识库管理）
# 启动后浏览器访问 http://127.0.0.1:8000/  或  /chat
```

## 真实模型接入

复制 `.env.example` 为 `.env`，填 `SMALL_LLM_API_KEY` / `SMALL_LLM_MODEL`（小模型）和
`LARGE_LLM_API_KEY` / `LARGE_LLM_MODEL`（大模型）。base_url 支持任意 OpenAI 兼容端点
（OpenAI / DeepSeek / Qwen / Moonshot / 本地 vLLM / Ollama）。未配置时自动降级 mock，离线可跑。

## 数据库（MySQL）

- 本机 Docker：`mysql-server`（root/root@127.0.0.1:3306，库 `ecommerce_agent`）
- 连接串从 `.env` / 环境变量读取（`app/config.py`），不硬编码
- 迁移：`alembic upgrade head`（表结构一律走 Alembic，禁手写 SQL 建表）
- checkpoint 默认存 MySQL（`CHECKPOINT_BACKEND=sqlite` 可回退官方 SqliteSaver 对拍）

## 已知局限（竖切片阶段，后续 Phase 补齐）

1. **模型默认 mock，可切真实**：分类/生成默认确定性规则；配 `.env` 后走 OpenAI 兼容接口（`app/llm/router.py` 按 role 路由 + 降级）。
2. **检索已接 Milvus**：向量检索 + category 标量过滤 + 内存关键词降级。哈希向量（无 embedding key）近似正交、语义排序退化，生产必须配真实 embedding 模型（`EMBEDDING_*` 环境变量）。
3. **complex 分支已实现 Plan-Execute-Reflect**：planner/executor/reflector/summarize 均为确定性 mock（后续换真实大模型/小模型）。
4. **checkpoint 已用 MySQLSaver**：自研实现已通过对拍测试，`apply_refund`/`apply_return` 等高风险写工具后续补齐。
5. **知识库前端已接，MinerU 需配 key**：知识库管理页 `/knowledge` 已实现；文档解析无 `MINERU_API_KEY` 时仅文本类（md/txt）本地降级，pdf/docx/xlsx/pptx 等二进制需配 MinerU token（`MINERU_API_KEY` / `MINERU_BASE_URL`）。
6. **五道安全闸门已接 execute_tool**：7 个工具经统一入口执行（query_order/query_coupon/query_product/cancel_order/update_address/apply_refund/apply_return）。
7. **库存/支付/优惠券为标记位模拟**：`stock_deducted`/`refund_amount`/`coupon_reverted` 均为本地标记位，真实库存/支付/优惠券服务联动后续对接。
8. **MySQLSaver 未实现 delta channel 快路径**：`get_delta_channel_history`/`prune`/`copy_thread`/`delete_for_runs` 等高级方法未实现（M1 竖切片用不到，需 DeltaChannel 的图后续补）。
9. **图片处理需配真实 VLM**：`describe_image` 无 `VLM_*` 配置时降级 alt 占位文本，图片转文字描述需配真实多模态模型；文件名生成同样无 key 降级标题+时间戳。
10. **MinIO 未配则降级本地磁盘**：无 `MINIO_*` 配置时图片落到 `data/uploads/`，生产多副本需配 MinIO 共享存储。

## 目录结构

```
agent-service/
├── run.py              # CLI 入口
├── eval.py             # golden set 评测
├── alembic/            # 数据库迁移（env.py + versions/）
├── alembic.ini
├── tests/              # 单元测试（数据层 + 接口层）
├── app/
│   ├── config.py       # 配置加载（.env → MySQL 连接串）
│   ├── main.py         # FastAPI 入口
│   ├── api/            # 接口层（schemas / routes / deps）
│   ├── db/             # ORM 数据模型 + engine/session
│   │   ├── base.py     # Base 声明基类
│   │   ├── session.py  # engine + sessionmaker + get_session
│   │   └── models/     # session / audit / checkpoint / order
│   ├── tools/          # 工具层（order + vision + registry 五道闸门 + errors + idempotency + permission）
│   ├── graph/          # 图编排（state + build + nodes：classify/rag/planner/executor/reflector/summarize/human）
│   ├── llm/            # 模型路由（role 分流 + OpenAI 兼容 + mock 降级）
│   ├── mcp/            # MCP streamable HTTP 端点（/mcp）
│   ├── memory/         # checkpoint（MySQLSaver + sqlite 回退）
│   ├── knowledge/      # 知识库（embedding/milvus/retriever/indexer/chunker/mineru/importer/describe）
│   ├── storage/        # 对象存储（minio_client：上传/下载/删除，降级本地磁盘）
│   └── observability/  # 可观测（错误分类/熔断/指标/Run 追踪）
```
