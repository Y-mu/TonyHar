import os
from pathlib import Path
from typing import Any, Union
# 统一管理跨平台设备、模型和 Chroma 客户端配置

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = PROJECT_ROOT / ".hf-cache"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DATABASE_NAME = "my_vector_db"
COLLECTION_NAME = "my_documents"

if MODEL_CACHE.exists():
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
import torch
from chromadb.utils import embedding_functions


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_embedding_function(model_name: str = DEFAULT_MODEL_NAME):
    device = get_device()
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
        )
    )
    print(f"嵌入模型: {model_name} ({device})")
    return embedding_function


def create_client(database_path: Union[str, Path]):
    database_path = Path(database_path)
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    return chromadb.PersistentClient(path=str(database_path))


def get_collection() -> Any:
    """获取 demo 使用的 Chroma Collection。

    数据库路径、Collection 名称、embedding 模型和距离空间都在本模块统一管理。
    """
    client = create_client(DATABASE_NAME)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=create_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
