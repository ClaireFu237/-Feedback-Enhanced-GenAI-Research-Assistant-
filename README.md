# Feedback-Enhanced GenAI Research Assistant

当前可运行的是本地 **PDF → 清洗文本 → chunks → 检索结果**。
PDF 解析、清洗和切块统一位于 `services/ingestion/`。
支持 BM25、dense、hybrid 和 hybrid + cross-encoder reranker。
已提供 FastAPI 原生 Swagger UI，支持全流程和各阶段独立测试。
LLM 回答、AWS 和用户反馈仍是后续工作，目前输出检索证据段。

## FastAPI 原生接口页面

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.5PDFprocess"
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 **http://127.0.0.1:8000/docs**，展开接口，点击 **Try it out → Execute** 测试。
首页也会自动跳转到 `/docs`。使用上传、清洗、切块、检索或 pipeline 接口测试对应流程。
测试记录与原有 OpenAlex 语料隔离，结果和中间文件可通过产物接口下载。
操作说明、接口输入输出和功能测试见 **[WEB_GUIDE.md](WEB_GUIDE.md)**。

## 安装和快速运行

建议 Python 3.12；完整神经检索沿用原 repo 的 Python 3.12 / macOS x86_64 依赖。

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.5PDFprocess"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 现有 PDF 位于 ../data/openalex/pdf。
python scripts/prepare_pdf_chunks.py --overwrite
python -m services.retrieval --dataset openalex --method bm25 --query "How does retrieval augmented generation work?" --top-k 5
```

如果没有 tokenizer 缓存，首次切块命令加 `--download-tokenizer`。
统一的 `requirements.txt` 包含切块、BM25、神经检索、Web API 和测试依赖（含 PyTorch）。
安装依赖不会下载检索模型权重。依赖版本沿用现有配置；神经检索原验证环境为 Python 3.12 / macOS x86_64。
默认 dataset 已设为 openalex，可省略 `--dataset openalex`。

## 神经检索模型准备

```bash
python -m pip install -r requirements.txt
python scripts/download_retrieval_models.py --config config/retrieval.json --method hybrid_reranker
python -m services.retrieval --dataset openalex --method hybrid_reranker --query "How is RAG evaluated?" --top-k 5
```

`dense` 使用向量检索，`hybrid` 融合 BM25 和 dense，`hybrid_reranker` 再重排候选段落。
首次 dense 查询生成并缓存向量。查询不会隐式下载模型，请显式运行下载脚本。
默认使用 Hugging Face 模型缓存，也可通过 `HF_HUB_CACHE` 指定位置。
当前使用本地模型，尚未连接 Bedrock。

## 运行测试和输入输出

完整步骤和字段解释见 **[RUN_TESTS_AND_IO.md](RUN_TESTS_AND_IO.md)**。
该文件包含全量处理、冒烟检查、重跑一致性、文档范围过滤及原 HotPotQA demo 的运行方法。
没有新增复杂 evaluation 程序。

| 路径 | 用途 |
| --- | --- |
| `services/ingestion/` | PDF 解析、清洗、切块和产物管理 |
| `scripts/prepare_pdf_chunks.py` | 切块 CLI，可从任何工作目录通过脚本路径调用 |
| `services/retrieval/` | corpus 读取、BM25、dense、融合和重排 |
| `config/retrieval.json` | 默认 OpenAlex 配置，包含 SciFact/HotPotQA 配置键 |
| `config/demo_retrieval.json` | 原 repo 附带的 HotPotQA demo 配置 |
| `data_local/processed/openalex/` | 生成的文本、chunks、来源映射和报告 |
| `data_local/embeddings/` | 生成的向量缓存 |
| `demo_data/hotpotqa/` | 原 repo 附带的示例语料 |

原始 PDF 留在共享 `../data/` 目录。生成数据、向量缓存和虚拟环境被 git 忽略，
新 clone 需要准备输入后重新运行切块。原 repo 的其他架构目录及 demo 保留。
多栏/章节识别是启发式，表格、公式和扫描页需要核查，当前不做 OCR。
`W3040807780.pdf` 有多页文字稀少，详见 `quality_report.json`。
