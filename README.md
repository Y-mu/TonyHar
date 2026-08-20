# TonyHar

本项目是一个使用 Chroma 和本地中文嵌入模型的向量检索示例。

## 目录

- `RAG/Chroma/make.py`：创建向量数据库并插入示例文档
- `RAG/Chroma/search.py`：使用语义相似度搜索文档
- `requirements.txt`：Python 依赖

## 安装

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

RTX 显卡可安装 CUDA 版 PyTorch：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## 运行

首次运行需要下载 `BAAI/bge-small-zh-v1.5`。网络较慢时可使用 Hugging Face 镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$PWD\.hf-cache"
python "RAG\Chroma\make.py"
python "RAG\Chroma\search.py"
```

模型缓存和生成的 `my_vector_db` 已加入 `.gitignore`，不会提交到仓库。
