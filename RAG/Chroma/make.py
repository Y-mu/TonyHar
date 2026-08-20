import os
from pathlib import Path

# 使用项目内缓存；已有缓存时离线加载，避免每次启动访问 Hugging Face。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = PROJECT_ROOT / ".hf-cache"
if MODEL_CACHE.exists():
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
import torch
from chromadb.utils import embedding_functions

# ─── 1. 初始化客户端 ───────────────────────────────────────────
# 持久化到本地（推荐）
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "my_vector_db"))

# 使用本地中文嵌入模型，不需要 OpenAI API Key
# 首次运行会自动从 Hugging Face 下载模型，之后可离线使用。
model_name = "BAAI/bge-small-zh-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name,
    device=device,
    normalize_embeddings=True,
)
print(f"嵌入模型: {model_name} ({device})")

# ─── 2. 创建集合（类似关系库里的"表"）────────────────────────
collection = client.get_or_create_collection(
    name="my_documents",                   # 集合名称
    embedding_function=local_ef,            # 绑定本地嵌入函数
    metadata={"hnsw:space": "cosine"}      # 使用余弦相似度
)

# ─── 3. 插入文档 ──────────────────────────────────────────────
documents = [
    "Python 是一种面向对象的解释型编程语言，广泛用于数据科学和 AI 开发",
    "机器学习是人工智能的子领域，让计算机从数据中学习规律",
    "深度学习使用多层神经网络，在图像识别和 NLP 任务中表现优异",
    "向量数据库专门存储高维向量，支持语义相似度搜索",
    "PostgreSQL 是功能强大的开源关系型数据库",
    "Redis 是基于内存的高性能键值数据库，常用于缓存",
    "Docker 容器化技术让应用可以在任何环境中一致运行",
    "Git 是分布式版本控制系统，是现代软件开发的基础工具",
]

ids = [f"doc_{i}" for i in range(len(documents))]

# 批量插入（Chroma 自动调用嵌入模型转为向量后存储）
collection.add(
    documents=documents,
    ids=ids,
    metadatas=[{"source": "tutorial", "index": i} for i in range(len(documents))]
)

print(f"已插入 {len(documents)} 条文档")
