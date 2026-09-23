"""semantic_router 第三方库的 core 层适配器。"""

from dataclasses import dataclass

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder


@dataclass(frozen=True)
class RouteResult:
    name: str
    score: float | None = None


class SemanticIntentRouter:
    """使用 semantic_router 对用户消息做意图路由。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.encoder = HuggingFaceEncoder(name=model_name)
        self.layer = SemanticRouter(
            encoder=self.encoder,
            # 把上面的 route utterances 编码并写入默认 LocalIndex。
            # 不设置时 LocalIndex 为空，首次调用会报 Index is not ready。
            auto_sync="local",
            routes=[
                Route(name="ingest", utterances=[
                    "请解析这个文件并存入知识库", "把这个文档导入向量数据库",
                    "读取文件并建立索引", "上传文件到知识库", "帮我把这个文件入库",
                ]),
                Route(name="knowledge_query", utterances=[
                    "根据已经入库的文档回答我的问题", "从知识库检索相关内容",
                    "查询我之前上传的资料", "文档里有没有关于这个问题的答案",
                    "知识库里有什么文档", "查看知识库中的文档列表",
                    "我存入了哪些文件", "搜索知识库里的资料",
                ]),
                Route(name="chat", utterances=[
                    "你好", "帮我计算一个数学问题", "回答一个普通问题", "介绍一下你自己",
                ]),
            ],
        )

    def classify(self, text: str) -> RouteResult:
        choice = self.layer(text)
        if choice is None:
            return RouteResult(name="chat")
        score = getattr(choice, "similarity_score", None)
        return RouteResult(
            name=getattr(choice, "name", "chat") or "chat",
            score=float(score) if score is not None else None,
        )
