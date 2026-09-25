# TonyHar

TonyHar 是一个基于 Python、DeepSeek、语义路由和 Chroma 的本地知识库 Agent。

全部生产代码位于统一的 `tonyhar` Python 包中；Agent、会话、工具运行时、
韧性策略、RAG 和具体工具以平级模块组成模块化单体。

## 项目规则

本项目不维护旧代码兼容层。接口变更时直接迁移全部调用方、测试和文档；旧模块迁移完成后立即删除。详细规则见 [AGENTS.md](AGENTS.md)。

## 当前架构

```text
CLI / FastAPI
    ↓
ChatService
    ├── SessionStore + session lock
    └── Agent.stream(context)
            ├── IntentPlanner
            ├── Deadline → AsyncOpenAI / DeepSeekLLM
            └── ToolExecutor
                    └── ToolRequest → ToolResult
```

### Agent 与会话

- `AgentRunContext` 保存单次运行状态。
- `AgentState` 管理规划、模型调用、工具执行和终态转换。
- `AgentEvent` 描述运行过程，可直接转发到 SSE 或 WebSocket。
- `AgentResult` 表示一次运行的最终结果。
- `ChatService` 负责会话加载、会话级串行化和消息持久化。
- 同一会话串行执行，不同会话通过 `asyncio` 并发执行。

### 模型与工具

- `DeepSeekLLM` 使用 `AsyncOpenAI`。
- 每轮运行共享一个总 deadline；模型和工具的单次超时都不能突破总预算。
- 模型调用包含有界重试、full jitter 指数退避和并发安全熔断器。
- 工具的唯一运行入口是异步 `BaseTool.execute()`，不保留同步执行协议。
- 阻塞型依赖由具体工具显式通过 `asyncio.to_thread()` 隔离。
- 幂等工具可配置重试；副作用工具默认不重试、串行执行。
- 标记为 `parallel_safe` 的同轮工具调用受并发上限保护并发执行。
- 工具使用 `@tool` 自动生成 JSON Schema。

正式错误码区分 `run_timeout`、`model_timeout`、
`model_rate_limited`、`model_unavailable`、`model_circuit_open`、
`tool_timeout` 和 `tool_execution_error`，调用方不需要解析异常文本。

### RAG

```text
文件
  → Parser
  → HybridSplitter
  → DocumentService
  → VectorStore
  → ChromaDB
```

RAG 服务通过 `VectorStore` 抽象隔离 Chroma，工具只接收由组合根注入的服务，不自行创建基础设施。

## 本地运行

```bash
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY="你的密钥"
python main.py
```

## 测试

当前测试使用项目虚拟环境中的 `unittest`：

```bash
env3.13/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## 下一阶段

下一步是将文件入库从线程池隔离的工具调用改造成独立 Workflow，使用 `task_id`、任务状态和进度管理；之后再增加 FastAPI 会话、任务和 SSE 接口。
