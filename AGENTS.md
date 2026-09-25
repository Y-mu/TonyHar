# TonyHar 项目开发规则

## 接口迁移策略

本项目不维护旧代码兼容层。

- 接口、协议或数据模型发生变化时，直接修改全部生产代码、测试和文档调用方。
- 不添加旧接口别名、fallback、adapter、版本分支或“同时支持新旧两套协议”的逻辑。
- 不为了兼容旧调用方式保留 `getattr`、`hasattr`、动态分派或隐式参数转换。
- 旧模块完成迁移后立即删除；无引用的旧文件不能继续作为备用实现保留。
- 测试只验证当前正式协议，不验证已经废弃的调用方式。

## 当前正式运行协议

- `AgentStateMachine.stream(context)` 是 Agent 的唯一运行入口。
- `Agent.invoke(context)` 只是收集事件后的非流式结果适配，不是另一套运行协议。
- `ChatService.stream()` 只调用 `Agent.stream()`，不兼容旧版 Agent。
- 会话状态由 `ChatService` 和 `SessionStore` 管理，Agent 不持有跨请求会话状态。
- 工具调用统一使用 `ToolRequest -> ToolExecutor -> ToolResult`。
- `BaseTool.execute(**arguments)` 是工具的唯一异步运行入口。

## 修改要求

- 优先修改正式接口及其所有调用方，不在边界处隐藏迁移问题。
- 完成重构后同步更新测试、README 和 `项目框架.md`。
- 保持模块化单体结构；除非明确要求，不引入微服务或重量级工作流框架。
