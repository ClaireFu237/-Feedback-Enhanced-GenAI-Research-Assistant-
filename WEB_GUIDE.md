# FastAPI 原生页面：运行与测试

## 启动

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.5PDFprocess"
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 **http://127.0.0.1:8000/docs**。这是 FastAPI 默认的 Swagger UI，
没有自定义 HTML、CSS 或 JavaScript。首页 `/` 会跳转到 `/docs`。
也可用 `/redoc` 阅读接口说明。

在 Swagger 中展开一个接口，点击 **Try it out**，填入输入，点击 **Execute**。
输入格式和响应字段可直接在页面中的 Schema 查看。
本地版本使用单 worker；按 Ctrl+C 正常停止服务。服务重启会将未完成任务标记为中断。

## 测试一篇 PDF 的全流程

1. 展开 `POST /api/runs`，选择 PDF，可填写 title，执行。PDF 最大 20 MB。
2. 从响应中复制 `run_id`。
3. 展开 `POST /api/runs/{run_id}/pipeline`，填入 run_id，并输入：

```json
{
  "chunk": {
    "chunk_size": 200,
    "overlap": 30,
    "max_input_tokens": 256,
    "include_references": false
  },
  "search": {
    "query": "How is retrieval augmented generation evaluated?",
    "method": "bm25",
    "top_k": 5
  }
}
```

4. 执行后返回 HTTP 202。复制 `task_id`，在 `GET /api/tasks/{task_id}` 中查询。
5. 任务为 `queued` 或 `running` 时可再次执行 GET；变成 `completed` 后，result 是检索结果。
   如果为 `failed`，查看 error 字段。
6. `GET /api/runs/{run_id}` 返回当前阶段、统计和有效产物的下载 URL。

结果是排序后的证据段，目前不生成 LLM 答案。

## 分别测试每个阶段

### PDF → 清洗文本

先执行上传接口，得到 run_id，然后执行：

```text
POST /api/runs/{run_id}/extract
```

使用任务接口查询完成状态，再执行 `GET /api/runs/{run_id}/cleaned`。
响应包含完整 text、paragraphs、标题和质量报告。
此阶段不需要 tokenizer 或检索模型。

### 清洗文本 → chunks

可以沿用已清洗的 run_id，也可以通过 `POST /api/runs/import-cleaned`
上传保存的 cleaned.json，创建独立记录。文件需包含 text 和 paragraphs。

执行 `POST /api/runs/{run_id}/chunks`，输入：

```json
{
  "chunk_size": 200,
  "overlap": 30,
  "max_input_tokens": 256,
  "include_references": false
}
```

完成后用 `GET /api/runs/{run_id}/chunks` 分页查看，offset 默认 0、limit 默认 20。
切块直接读取保存的清洗结果，不重新解析 PDF。

如果只有纯文本，用 `POST /api/runs/text`：

```json
{
  "title": "My text",
  "text": "First paragraph.\n\nSecond paragraph."
}
```

然后使用同一个 chunks 接口。纯文本只提供字符范围，不虚构 PDF 页码。

### chunks → 检索结果

已有 chunks 后，执行 `POST /api/runs/{run_id}/search`：

```json
{
  "query": "How is RAG evaluated?",
  "method": "bm25",
  "top_k": 5
}
```

完成后查看任务 result，或 `GET /api/runs/{run_id}/results`。
只换问题会复用现有 chunks 和索引。

### 直接搜索已有 OpenAlex 语料

无需上传文件，执行 `POST /api/search`：

```json
{
  "dataset": "openalex",
  "query": "How is RAG evaluated?",
  "method": "bm25",
  "top_k": 5
}
```

然后通过任务接口获取结果。不会覆盖已有 corpus。

## 输入、输出和版本

- 每次导入创建独立 run_id；同一记录同一时间只执行一个任务。
- 中间文件保存在 `data_local/web_runs/`，被 git 忽略，与命令行的 OpenAlex corpus 隔离。
- 切块默认正文最多 200 tokens，重叠最多 30；标题、正文和 special tokens 总计最多 256。
- 修改参数后重新执行 chunks，旧索引和搜索结果失效；重新执行 extract 会使下游全部失效。
- pipeline 复用已有清洗文本，缺少 chunks 或切块参数变化时重新切块，然后检索。
- `GET /api/runs/{run_id}/artifacts/{name}` 下载对应文件；可用名称由 run 响应的 artifacts 提供。
- cleaned.json 保留 text、paragraphs、report；正式 corpus 包含 documents/chunks/id_map/manifest 等七个文件。
- PDF 页码从 1 开始；字符范围是 normalized text 的 Python 下标 `[char_start, char_end)`。
- 检索 results 包含 rank、score、score_type、正文、文档标题、来源位置和 IDs。
  不同方法的分数类型不能直接比较。
- 任务失败原因在 error；404 表示产物不存在或已失效，409 表示前置步骤未完成或记录正忙。

## 模型准备

BM25 无需模型权重。切块需要 tokenizer；缓存不存在时可显式准备一次：

```bash
python scripts/prepare_pdf_chunks.py --limit 1 --download-tokenizer --output data_local/test_tokenizer --overwrite
```

神经检索依赖已包含在 requirements.txt 中；使用 dense / hybrid / hybrid_reranker 前准备模型：

```bash
python -m pip install -r requirements.txt
python scripts/download_retrieval_models.py --config config/retrieval.json --method hybrid_reranker
```

重启服务后可调用相应方法。接口不会隐式下载模型。首次向量计算可能较慢，后续复用缓存。

## 功能测试

无需启动 Uvicorn：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_web_api.py' -v
```

测试使用临时目录和生成的小 PDF，检查阶段复用、失效规则、来源映射、隔离、
错误处理和任务状态。
复杂双栏、表格和公式仍需人工核查，扫描页不做 OCR。
