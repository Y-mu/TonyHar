import os
from pathlib import Path
from typing import Any, Union
# 统一管理跨平台设备、模型和 Chroma 客户端配置

# chroma_config.py 位于 RAG/Chroma/store/，项目根目录需要向上三级。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CACHE = PROJECT_ROOT / ".hf-cache"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DATABASE_NAME = "my_vector_db"
COLLECTION_NAME = "my_documents"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE))

# Chroma 会从已有 Collection 的配置中重建 embedding function；该配置保存的
# 是模型仓库名而非本地 snapshot 路径。确认默认模型已完整缓存后，在导入
# chromadb/transformers 前启用离线模式，避免重建时再次访问 Hugging Face。
_default_model_cache = (
    MODEL_CACHE
    / "hub"
    / ("models--" + DEFAULT_MODEL_NAME.replace("/", "--"))
    / "snapshots"
)
if _default_model_cache.is_dir() and any(
    (snapshot / "config.json").is_file()
    and (snapshot / "modules.json").is_file()
    and (snapshot / "model.safetensors").is_file()
    for snapshot in _default_model_cache.iterdir()
    if snapshot.is_dir()
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
import torch
from chromadb.utils import embedding_functions


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_cached_model(model_name: str) -> Union[str, None]:
    """返回 Hugging Face 缓存中完整模型 snapshot 的本地路径。"""
    model_dir = MODEL_CACHE / "hub" / ("models--" + model_name.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    snapshots_dir = model_dir / "snapshots"

    candidates = []
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots_dir / revision)
    if snapshots_dir.is_dir():
        candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())

    required_files = ("config.json", "modules.json", "model.safetensors")
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required_files):
            return str(candidate.resolve())
    return None


def create_embedding_function(model_name: str = DEFAULT_MODEL_NAME):
    device = get_device()
    cached_model = find_cached_model(model_name)
    resolved_model = cached_model or model_name
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=resolved_model,
            device=device,
            normalize_embeddings=True,
        )
    )
    source = "local cache" if cached_model else "Hugging Face"
    print(f"嵌入模型: {model_name} ({device}, {source})")
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
