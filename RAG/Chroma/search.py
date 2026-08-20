import os
from pathlib import Path

# 使用项目内的模型缓存，避免 VS Code/不同启动目录导致重复联网下载。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = PROJECT_ROOT / ".hf-cache"
if MODEL_CACHE.exists():
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import chromadb
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
    embedding_function=local_ef,          # 绑定嵌入函数
    metadata={"hnsw:space": "cosine"}      # 使用余弦相似度
)

# ─── 4. 语义搜索 ──────────────────────────────────────────────
query = "如何用 Python 做人工智能"

results = collection.query(
    query_texts=[query],
    n_results=3,                # 返回最相似的 3 条
    include=["documents", "distances", "metadatas"]
)

print(f"\n查询：{query}")
print("-" * 50)
for i, (doc, dist) in enumerate(zip(
    results["documents"][0],
    results["distances"][0]
)):
    similarity = 1 - dist   # 余弦距离转相似度
    print(f"第 {i+1} 名（相似度 {similarity:.4f}）：")
    print(f"  {doc}")
    print()
