# TonyHar

本项目是一个使用 Chroma 和本地中文嵌入模型的向量检索示例。

## 目录

- `RAG/Chroma/chunking.py`：将文本按 token 分块
- `RAG/Chroma/index_documents.py`：提取向量并写入 ChromaDB
- `RAG/Chroma/search.py`：查询已建立的向量索引
- `RAG/Chroma/main.py`：运行完整的建索引与查询示例
- `RAG/Chroma/chroma_config.py`：统一管理跨平台设备、模型和 Chroma 客户端配置
- `requirements.txt`：Python 依赖

## 安装

Windows PowerShell：

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux：

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

Windows 上的 NVIDIA 显卡可安装 CUDA 版 PyTorch：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Python 代码会自动选择 CUDA（Windows/NVIDIA）、MPS（Apple Silicon）或 CPU，不需要维护不同平台的代码。

## 运行

首次运行需要下载 `BAAI/bge-small-zh-v1.5`。网络较慢时可使用 Hugging Face 镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$PWD\.hf-cache"
python "RAG\Chroma\main.py"
```

macOS/Linux：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$PWD/.hf-cache"
python RAG/Chroma/main.py
```

模型缓存以及生成的 `my_vector_db/` 和 `chroma_db/` 已加入 `.gitignore`，不会提交到仓库。
