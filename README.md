# TonyHar

TonyHar 是一个基于 Python、DeepSeek、语义路由和 Chroma 的本地知识库 Agent。

全部生产代码位于统一的 `tonyhar` Python 包中；Agent、会话、工具运行时、
韧性策略、RAG 和具体工具以平级模块组成模块化单体。

## 项目规则

本项目不维护旧代码兼容层。接口变更时直接迁移全部调用方、测试和文档；旧模块迁移完成后立即删除。详细规则见 [AGENTS.md](AGENTS.md)。

## 当前架构

```text
CLI / FastAPI + SSE
    ↓
ChatService
    ├── SQLiteSessionStore + session lock
    ├── ContextBuilder + MemoryCompactor
    └── Agent.stream(context)
            ├── RunDispatcher → RunHandler
            ├── AgentLoop → AsyncOpenAI / DeepSeekLLM
            └── ToolRunner → ToolManager
                              └── ToolRequest → ToolResult
```

### Agent 与会话

- `AgentRunContext` 保存单次运行状态。
- `AgentState` 管理请求分派、模型调用、工具执行和终态转换。
- `RunDispatcher` 只把意图分类结果映射为直接工具、检索后生成或普通 Agent 三种策略。
- 知识库检索采用 dense 向量召回与 Chroma 上的 BM25 词法召回混合，再以加权 RRF 去重排序；知识库工具统一调用 `Retriever.retrieve_hybrid()`。
- 三种 `RunHandler` 承担策略执行，`AgentLoop` 只处理 LLM 与模型发起的 Tool 循环。
- `AgentEvent` 描述运行过程，可直接转发到 SSE 或 WebSocket。
- `AgentResult` 表示一次运行的最终结果。
- `ChatService` 负责完整会话日志、短期记忆压缩、上下文组装、会话级串行化和持久化。
- `ContextBuilder` 按模型 token 预算选择最近的完整轮次，不再按固定轮次数永久删除历史。
- 较早的已完成轮次由 `LLMMemoryCompactor` 压缩为摘要和结构化任务状态；压缩失败时使用抽取式压缩。
- 会话默认写入 `data/sessions.sqlite3`，进程重启后可以恢复。
- 失败或取消的轮次会保留在日志中，但不会进入后续模型上下文。
- 同一会话串行执行，不同会话通过 `asyncio` 并发执行。

### 模型与工具

- `DeepSeekLLM` 使用 `AsyncOpenAI`，并通过唯一的异步 `BaseLLM.stream()`
  协议输出文本增量和拼装后的完整响应。
- 每轮运行共享一个总 deadline；模型和工具的单次超时都不能突破总预算。
- 模型调用包含有界重试、full jitter 指数退避和并发安全熔断器。
- 模型流在首个文本增量发出前可以重试；开始向用户输出后不再自动重试，
  避免重复文本。
- 工具的唯一运行入口是异步 `BaseTool.execute()`，不保留同步执行协议。
- `@tool` 将异步函数或 `BaseTool` 子类转换为完整的 `ToolDefinition / BaseTool`；
  函数参数类型提示、docstring、JSON Schema 或 Pydantic 模型可用于生成参数 Schema。
- `ToolManager` 在应用启动时构建全部 `BaseTool` 实例，并统一管理 Schema、超时、
  重试、并发和批量执行。
- 阻塞型依赖由具体工具显式通过 `asyncio.to_thread()` 隔离。
- 幂等工具可配置重试；副作用工具默认不重试、串行执行。
- 标记为 `parallel_safe` 的同轮工具调用受并发上限保护并发执行。
- 具体工具可以直接实现 `BaseTool`，也可以使用 `@tool` 将异步函数转换为 `BaseTool`；
  装饰器不修改全局 Manager。
- URL 正文工具在 Trafilatura 提取后统一规范 Unicode、空白和软换行，同时保留
  Markdown 标题、列表、表格及代码块结构，便于后续文档切块。
- `RunDispatcher` 为普通 Agent 请求声明本轮模型可见的只读 Tool；`AgentLoop`
  通过 `ToolManager.schemas(allowed_names)` 只发送允许的 Schema，并拒绝越权调用。
- `spider_url` 由 `RunDispatcher` 确定性预执行：路由层从用户指令提取并校验
  HTTP(S) URL，再生成 `ToolRequest`；抓取结果写入上下文后进入 `AgentLoop`，
  并向模型提供 `file_ingestion` Tool Schema。
- 所有 Tool 都只通过 `ToolManager.execute()` 执行模型或 Handler 产生的 `ToolRequest`。
- Tool 的完整结构化结果继续用于业务和展示；写入模型上下文的内容受单工具和单轮总量限制，避免大结果撑爆上下文。

正式错误码区分 `run_timeout`、`model_timeout`、
`model_rate_limited`、`model_unavailable`、`model_circuit_open`、
`tool_timeout` 和 `tool_execution_error`，调用方不需要解析异常文本。

### RAG

```text
本地文件 → FileTextReader ─┐
网页正文或其他字符串 ──────┤
                           ↓
                  TextIngestionService
  → HybridSplitter
  → DocumentService
  → VectorStore
  → ChromaDB
```

文件读取/解析与文本切块/入库是两个独立步骤。`TextIngestionService` 只接收正文字符串和
文档 metadata，因此本地文件与网页抓取结果可以复用同一条入库流水线。RAG 服务通过
`VectorStore` 抽象隔离 Chroma，工具只接收由组合根注入的服务，不自行创建基础设施。

## 本地运行

```bash
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY="你的密钥"
python main.py
```

## Web 后端

Web 层是 `ChatService.stream()` 的 HTTP/SSE 适配器，不直接访问 Agent 或
SessionStore。组合根位于 `tonyhar/bootstrap.py`，CLI 和 FastAPI 共用同一套
依赖装配。

启动本地开发服务：

```bash
export DEEPSEEK_API_KEY="你的密钥"
env3.13/bin/uvicorn tonyhar.web.app:app --reload --host 127.0.0.1 --port 8000
```

主要接口：

```text
POST /api/v1/sessions
GET  /api/v1/sessions/{session_id}
POST /api/v1/sessions/{session_id}/messages/stream
GET  /health/live
GET  /health/ready
```

流式消息接口接收 `{"message": "..."}`，并以 SSE 输出 `AgentEvent`。
模型生成的可见文本通过连续的 `text_delta` 事件增量返回；`final_answer`
仍携带完整答案，用于终态确认、会话持久化和客户端最终校准。浏览器应使用
支持 POST 响应流的 `fetch()` 消费它。

开发环境默认允许 `http://localhost:5173` 和
`http://127.0.0.1:5173` 跨域访问。可以使用逗号分隔的
`TONYHAR_CORS_ORIGINS` 显式修改允许来源。

生产组合根使用 SQLite 持久化会话；`InMemorySessionStore` 只用于单元测试。
会话锁仍是进程内锁，因此启动 Uvicorn 时不要配置多个 worker。

## Web 前端

前端位于 `frontend/`，采用 React、Vite、TypeScript 和 assistant-ui。
它直接使用浏览器 `fetch(POST)` 消费 FastAPI SSE，不增加 BFF 或兼容协议层。

```bash
cd frontend
npm install
npm run dev
```

开发服务器默认把 `/api` 和 `/health` 代理到 `127.0.0.1:8000`。部署到独立域名时，
可在 `frontend/.env` 设置 `VITE_API_BASE_URL`，例如 `https://api.example.com`。
当前 SSE 同时返回 Agent 执行事件和模型 `text_delta`；界面会增量追加回答，
显示模型、工具和终态状态，并支持停止当前请求、Markdown/GFM 渲染、错误重试和移动端会话抽屉。
直接执行“列出文档”时，工具仍返回结构化目录数据，Agent 会通过工具声明的展示格式化器
将最终答案转换为包含文档总数、文件名和文档 ID 的 Markdown 列表。

前端检查命令：

```bash
npm run lint
npm run test
npm run build
```

## 测试

当前测试使用项目虚拟环境中的 `unittest`：

```bash
env3.13/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## 下一阶段

下一步可增加短期记忆质量回归集和上下文 token 指标，并将文件入库从线程池隔离的工具调用改造成独立 Workflow，使用
`task_id`、任务状态和进度管理，并为 Web 增加受控文件上传和任务状态接口。
