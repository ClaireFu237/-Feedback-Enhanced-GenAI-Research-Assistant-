# Feedback-Enhanced GenAI Research Assistant

论文研究助手，当前支持 **PDF → 文本清洗 → 切块 → 发布资料库 → 检索证据 → LLM 生成带引用的回答**。

PDF 处理和检索在本地运行，回答生成通过 Amazon Bedrock Converse 调用。
FastAPI 提供 Swagger UI，可逐步执行流程并查看任务状态、中间产物和来源信息。
用户反馈学习、S3 / DynamoDB 存储及云端部署尚未实现。

## 安装与启动

在项目根目录执行。建议使用 Python 3.12；神经检索依赖原验证环境为 Python 3.12 / macOS x86_64。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

首次处理 PDF 或文本前，准备本地 tokenizer 和 dense 模型：

```bash
python scripts/download_retrieval_models.py --config config/retrieval.json --method hybrid
```

启动 API：

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/docs>，点击 **Try it out → Execute** 调用接口。
`/redoc` 提供接口文档，`/` 自动跳转到 `/docs`。
依赖安装不包含模型权重；模型下载脚本需联网，日常检索不会隐式下载模型。

## 通过 API 运行 workflow

| 步骤 | 接口 | 作用 |
| --- | --- | --- |
| 1 | `POST /api/runs` | 上传 PDF，取得 `run_id` |
| 2 | `POST /api/runs/{run_id}/extract` | 提取并清洗文本 |
| 3 | `POST /api/runs/{run_id}/chunks` | 切块并保存来源位置 |
| 4 | `POST /api/runs/{run_id}/publish` | 将已切块论文发布到共享 library |
| 5 | `GET /api/library/documents` | 查看资料库及可用于过滤的 `document_id` |
| 6 | `POST /api/search` | 检索 library 中的证据 |
| 7 | `POST /api/answers` | 检索并调用 Bedrock 生成带引用的回答 |

处理、检索、发布和生成接口返回异步任务；通过 `GET /api/tasks/{task_id}` 查看状态和结果。
上传接口直接返回 run 信息。

`POST /api/runs/{run_id}/pipeline` 可自动执行该 run 所需的提取、切块和检索，**不包含发布和 LLM 生成**。
也可通过 `POST /api/runs/text` 输入文本，或通过 `POST /api/runs/import-cleaned` 导入清洗后的 JSON。

检索示例，默认方法为 BM25：

```json
{
  "query": "What methods are used in these papers?",
  "dataset": "library",
  "method": "bm25",
  "top_k": 5
}
```

可选 `document_ids` 限定检索范围。详细的阶段参数和产物下载方式见 [WEB_GUIDE.md](WEB_GUIDE.md)。

## 检索方式与本地 CLI

| 方法 | 行为 |
| --- | --- |
| `bm25` | 关键词检索 |
| `dense` | 本地 embedding 向量检索 |
| `hybrid` | BM25 与 dense 的候选结果通过 RRF 融合 |
| `hybrid_reranker` | 对融合后的完整候选集合执行 cross-encoder 重排 |

如需重排，先额外准备 reranker 模型：

```bash
python scripts/download_retrieval_models.py --config config/retrieval.json --method hybrid_reranker
```

使用仓库自带的 HotPotQA 示例语料，可直接运行 BM25，无需准备自己的 PDF 或下载模型：

```bash
python -m services.retrieval \
  --config config/demo_retrieval.json --dataset hotpotqa \
  --method bm25 --query "Who wrote this work?" --top-k 5
```

处理自己的 PDF 并检索：

```bash
python scripts/prepare_pdf_chunks.py \
  --pdf /path/to/paper.pdf \
  --output data_local/processed/openalex \
  --download-tokenizer

python -m services.retrieval \
  --dataset openalex --method bm25 \
  --query "What methods are used in this paper?" --top-k 5
```

批量处理可使用 `--pdf-dir /path/to/pdfs`；替换已有处理产物时使用 `--overwrite`。
默认正文 chunk 上限为 200 tokens，重叠上限为 30 tokens，标题加正文及特殊标记上限为 256 tokens。
CLI 默认语料为 `openalex`，API 默认检索 `library`。
PDF CLI 的输出不会自动发布到 API 的 library。

## LLM 回答生成

生成前，需通过 API 将至少一篇论文切块并发布到 library，并准备好 dense 模型。
配置运行环境中的 AWS SDK 凭证，确保账号可调用配置指定的 Bedrock 模型。

[config/generation.json](config/generation.json) 当前配置为：

| 配置 | 当前值 |
| --- | --- |
| AWS region | `us-east-1` |
| 模型 / inference profile ID | `us.amazon.nova-lite-v1:0` |
| 最大输出 tokens | `1400` |
| temperature | `0.0` |
| 上下文字符预算 | `40000` |

生成必须显式选择 [hybrid_v1](config/retrieval_profiles/hybrid_v1.json) profile：
BM25 和 dense 各取 50 个候选，以 RRF 60 融合后取前 10 个 chunks，不使用 cross-encoder 重排。

CLI 示例：

```bash
python -m services.generation \
  --profile config/retrieval_profiles/hybrid_v1.json \
  --query "What methods are used in these papers?"
```

API：向 `POST /api/answers` 提交：

```json
{
  "query": "What methods are used in these papers?",
  "retrieval_profile": "hybrid_v1"
}
```

结果包含结构化 claims、来源引用和缺失信息；回答状态为 `answered`、`partial` 或 `insufficient_evidence`。
代码校验输出结构和引用 ID，引用可追溯到原论文、chunk 和来源位置。
生成服务复用首次加载的 library 快照；发布新论文后，重启 API 或重新运行生成 CLI 以使用最新资料库。

具体生成协议、配置和产物说明见 [services/generation/README.md](services/generation/README.md)。

## 目录与产物

| 路径 | 用途 |
| --- | --- |
| `main.py` | FastAPI 入口、请求校验与路由 |
| `services/workbench.py` | 处理任务队列、阶段调度、检索和产物管理 |
| `services/ingestion/` | PDF 提取、清洗、切块与来源映射 |
| `services/retrieval/` | 语料读取、BM25、dense、融合与重排 |
| `services/generation/` | 上下文构建、Bedrock 调用、引用校验与记录 |
| `storage/local/library.py` | 本地资料库发布与快照管理 |
| `config/` | 检索、生成和版本化 retrieval profile 配置 |
| `prompts/` | 生成系统提示词和上下文模板 |
| `scripts/` | PDF 处理和模型准备 CLI |
| `tests/` | 离线功能与回归测试 |
| `demo_data/hotpotqa/` | 仓库自带的示例检索语料及来源说明 |
| `data_local/processed/` | CLI 生成的语料、中间产物和处理报告 |
| `data_local/embeddings/` | CLI 检索向量缓存 |
| `data_local/web_runs/` | API 运行记录、任务、library、向量缓存和回答产物 |
| `outputs/generation/` | CLI 生成的回答记录 |

每次回答按 `ans_<id>/` 保存 `record.json`、`retrieval.json`、`context.json`、
`request.json`，以及成功收到模型响应后的 `response.json`。
失败时保存已完成的产物及错误记录。

`frontend/` 记录 Swagger UI 的使用方式；`infrastructure/`、`routes/`、`schemas/`、
`utils/`、`services/embeddings/`、`services/feedback/` 和部分 storage 子目录仍为说明或架构预留。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

测试覆盖 PDF workflow、阶段复用、资料隔离、检索重排、生成协议及失败处理，使用 mock 验证云端调用契约。
更多输入输出与运行检查见 [RUN_TESTS_AND_IO.md](RUN_TESTS_AND_IO.md)。

多栏和章节识别采用启发式，表格、公式及扫描页需要人工核查；当前不支持 OCR。
回答引用校验不保证模型陈述在语义上完全由来源支持。
本地生成数据、模型缓存、虚拟环境和回答产物被 Git 忽略；新 clone 需准备模型和个人语料。
