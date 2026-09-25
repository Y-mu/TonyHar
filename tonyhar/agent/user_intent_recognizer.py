"""用户意图识别层：Dense embedding + BM25 混合路由。"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import Mapping

import numpy as np
from semantic_router import HybridRouter, Route
from semantic_router.encoders import BM25Encoder, HuggingFaceEncoder
from semantic_router.index import HybridLocalIndex
from semantic_router.tokenizers import BaseTokenizer, PretrainedTokenizer
from tokenizers import Tokenizer

from tonyhar.model_config import (
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    EmbeddingModelConfig,
)


logger = logging.getLogger(__name__)


# 路由训练语料：每个 key 都是最终返回给 Agent 的意图名称。
ROUTE_UTTERANCES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    # 文件入库：解析用户提供的文件，并写入知识库。
    "ingest": (
        "请解析这个文件并存入知识库",
        "把这个文档导入向量数据库",
        "读取文件并建立索引",
        "上传文件到知识库",
        "帮我把这个文件入库",
        "请把 manual.txt 导入知识库",
        "导入本地文件 manual.txt",
        "解析并入库 /path/to/document.md",
        "把这个路径下的文档加入知识库",
    ),
    # 知识库清单：查询已经入库的文件、文档数量或入库状态。
    "knowledge_list": (
        "知识库里有什么文档",
        "查看知识库中的文档列表",
        "我存入了哪些文件",
        "列出全部入库文件",
        "知识库中一共有多少篇文档",
        "manual.txt 是否已经入库",
    ),
    # 知识库问答：从已入库内容中检索资料并回答问题。
    "knowledge_query": (
        "根据已经入库的文档回答我的问题",
        "从知识库检索相关内容",
        "文档里有没有关于这个问题的答案",
        "搜索知识库里的资料内容",
        "从资料中检索相关段落",
        "根据文档总结问题的答案",
        "查询文档中关于发动机漏水的内容",
        "文档里的维修建议是什么",
    ),
    # 普通对话：不要求访问知识库的问候、计算或通用问题。
    "chat": (
        "你好",
        "帮我计算一个数学问题",
        "回答一个普通问题",
        "介绍一下你自己",
    ),
})


# 各路由的最低接受分数；低于对应值时统一返回 unknown。
# 初始值由项目正负样本标定，增加真实语料后应通过回归测试重新调整。
DEFAULT_ROUTE_THRESHOLDS: Mapping[str, float] = MappingProxyType({
    "ingest": 0.65,  # 防止删除、修改等操作被误判为文件入库
    "knowledge_list": 0.65,  # 防止无意义短文本被误判为查询文档列表
    "knowledge_query": 0.55,  # 知识问答表达较多样，因此使用稍低阈值
    "chat": 0.55,  # 保留正常通用问题，同时拒绝明显无关输入
})


@dataclass(frozen=True)
class IntentRouterConfig:
    """意图路由运行配置。"""

    # Dense encoder 与 Chroma 共用的模型及缓存配置。
    model: EmbeddingModelConfig = DEFAULT_EMBEDDING_MODEL_CONFIG
    # Dense 相似度在 Dense + BM25 混合分数中的权重，取值范围为 0～1。
    alpha: float = 0.6
    # 输入保护上限；空输入、非字符串和超长输入都会直接返回 unknown。
    max_input_length: int = 4_000
    # 路由名称及其训练语料，可在测试或不同部署环境中整体替换。
    utterances: Mapping[str, tuple[str, ...]] = ROUTE_UTTERANCES
    # 每个路由独立的拒识阈值，未达到阈值的结果返回 unknown。
    thresholds: Mapping[str, float] = DEFAULT_ROUTE_THRESHOLDS


DEFAULT_INTENT_ROUTER_CONFIG = IntentRouterConfig()


class LocalModelTokenizer(BaseTokenizer):
    """从本地 snapshot/tokenizer.json 加载，避免访问 HF Hub。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = Tokenizer.from_file(
            str(Path(model_path) / "tokenizer.json")
        )
        self.tokenizer.enable_padding(direction="right", pad_id=0)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def config(self) -> dict[str, Any]:
        return {"model_path": self.model_path}

    def tokenize(self, texts: str | list[str], pad: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        encodings = self.tokenizer.encode_batch_fast(
            texts,
            add_special_tokens=False,
        )
        return np.array([encoding.ids for encoding in encodings])


@dataclass(frozen=True)
class IntentResult:
    """稳定的分类结果，不向上层泄漏 semantic-router 类型。"""

    name: str
    score: float | None = None
    matched_by: str = "hybrid"


class UserIntentRecognizer:
    """负责输入校验、混合分类与低置信度拒识。"""

    def __init__(self, config: IntentRouterConfig = DEFAULT_INTENT_ROUTER_CONFIG):
        if not 0.0 <= config.alpha <= 1.0:
            raise ValueError("alpha 必须在 0 到 1 之间")
        self.config = config
        self.model_name = config.model.model_name
        self.local_model = config.model.find_cached_model()
        self.model_source = (
            str(self.local_model) if self.local_model else self.model_name
        )

        logger.info(
            "初始化语义路由模型 model=%s source=%s",
            self.model_name,
            "local-cache" if self.local_model else "hugging-face",
        )
        # （稠密特征）
        self.dense_encoder = self._create_dense_encoder()
        # （稀疏特征）
        self.sparse_encoder = self._create_sparse_encoder()
        self.routes = self._create_routes()
        self.index = HybridLocalIndex()
        self.layer = self._create_router()

    def _create_dense_encoder(self) -> HuggingFaceEncoder:
        return HuggingFaceEncoder(name=self.model_source)

    def _create_sparse_encoder(self) -> BM25Encoder:
        tokenizer = (
            LocalModelTokenizer(str(self.local_model))
            if self.local_model
            else PretrainedTokenizer(self.model_name)
        )
        return BM25Encoder(tokenizer=tokenizer, use_default_params=False)

    def _create_routes(self) -> list[Route]:
        return [
            Route(
                name=name,
                utterances=list(utterances),
            )
            for name, utterances in self.config.utterances.items()
        ]

    def _create_router(self) -> HybridRouter:
        return HybridRouter(
            encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            routes=self.routes,
            index=self.index,
            auto_sync="local",
            alpha=self.config.alpha,
        )

    def classify(self, text: str) -> IntentResult:
        if not isinstance(text, str):
            return IntentResult(name="unknown", matched_by="validation")

        normalized = text.strip()
        if not normalized or len(normalized) > self.config.max_input_length:
            return IntentResult(name="unknown", matched_by="validation")

        # 无 function schema 时无需similarity_score simulate_static；当前 semantic-router 只有此分支
        # 会把聚合后的真实  写入 RouteChoice。
        choice = self.layer(normalized, simulate_static=False)
        name = getattr(choice, "name", None)
        score = getattr(choice, "similarity_score", None)
        numeric_score = float(score) if score is not None else None
        threshold = self.config.thresholds.get(name) if name else None
        if (
            not name
            or numeric_score is None
            or (threshold is not None and numeric_score < threshold)
        ):
            return IntentResult(
                name="unknown",
                score=numeric_score,
                matched_by="threshold",
            )
        return IntentResult(
            name=name,
            score=numeric_score,
        )


@lru_cache(maxsize=1)
def get_shared_intent_recognizer() -> UserIntentRecognizer:
    """返回进程级共享实例，避免每个 Agent 重复加载模型与构建索引。"""
    return UserIntentRecognizer()
