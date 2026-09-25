"""Embedding 模型的统一配置与本地缓存解析。"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """RAG 与意图路由共享的 embedding 模型配置。"""

    model_name: str = "BAAI/bge-small-zh-v1.5"
    cache_dir: Path = PROJECT_ROOT / ".hf-cache"
    device_env_var: str = "RAG_DEVICE"

    def find_cached_model(self) -> Path | None:
        """优先返回 refs/main 指向的完整 snapshot，再检查其他 snapshot。"""
        model_dir = self.cache_dir / "hub" / (
            f"models--{self.model_name.replace('/', '--')}"
        )
        snapshots_dir = model_dir / "snapshots"
        candidates: list[Path] = []

        main_ref = model_dir / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots_dir / revision)

        if snapshots_dir.is_dir():
            candidates.extend(
                sorted(
                    (
                        path
                        for path in snapshots_dir.iterdir()
                        if path.is_dir() and path not in candidates
                    ),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )

        required_files = (
            "config.json",
            "modules.json",
            "model.safetensors",
            "tokenizer.json",
        )
        for candidate in candidates:
            if all((candidate / filename).is_file() for filename in required_files):
                return candidate.resolve()
        return None

    @property
    def model_source(self) -> str:
        cached_model = self.find_cached_model()
        return str(cached_model) if cached_model else self.model_name


DEFAULT_EMBEDDING_MODEL_CONFIG = EmbeddingModelConfig()

# 两个模型消费者都会间接导入本模块；在导入 Hugging Face 相关库前统一环境。
os.environ.setdefault("HF_HOME", str(DEFAULT_EMBEDDING_MODEL_CONFIG.cache_dir))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
