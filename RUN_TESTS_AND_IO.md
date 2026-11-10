# 如何运行测试，以及系统的输入输出

当前系统完成 **PDF → chunks → 排序后的证据段**。
以下检查验证 workflow 运行和输出一致性。
所有命令在 `11.5PDFprocess` 根目录执行。

## 1. 安装环境

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.5PDFprocess"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/prepare_pdf_chunks.py --help
python -m services.retrieval --help
```

统一安装包含 PDF 切块、BM25、神经检索、API 和测试依赖（含 PyTorch），不需要 AWS 账号或 LLM。
切块需要 MiniLM tokenizer；若没有本地缓存，首次切块加 `--download-tokenizer`。
可以用 `--tokenizer-file /path/to/tokenizer.json` 指定离线文件。

## 2. 系统输入

### PDF 输入

| 参数/来源 | 含义 |
| --- | --- |
| 默认 PDF 目录 | `../data/openalex/pdf/`；非递归读取 `.pdf` |
| `--pdf` | 只处理指定的一篇 PDF |
| `--pdf-dir` | 处理其他 PDF 文件夹，与 `--pdf` 互斥 |
| `--manifest` | 可选 OpenAlex 元数据 JSON 数组；默认 PDF 目录自动使用 `../data/openalex/manifest.json` |
| `--output` | 默认 `data_local/processed/openalex/`，与默认检索配置一致 |
| `--limit` | 只处理文件名排序后的前 N 篇，用于试跑 |
| `--chunk-size` | 正文 token 上限，默认 200 |
| `--overlap` | 重叠 token 上限，默认 30，实际可能更小或为零 |
| `--max-input-tokens` | 标题 + 正文 + special tokens 上限，默认 256 |
| `--include-references` | 同时切分检测到的参考文献，默认不包含 |
| `--overwrite` | 允许替换非空输出目录内的七个生成文件 |

元数据的 `doc_id` 与 PDF 文件名去掉 `.pdf` 后匹配。常用字段为 `title`、
`source`、`doi`、`openalex_id`、`pdf_sha256`；提供 PDF hash 时会校验文件内容。
没有元数据时也可处理：文件名作为标题，PDF hash 作为来源 ID。
元数据不会作为程序指令执行。

```bash
# 默认数据：现有 30 篇 PDF；重复执行可重建产物。
python scripts/prepare_pdf_chunks.py --overwrite
```

其他来源示例（把路径改为实际路径）：

```bash
python scripts/prepare_pdf_chunks.py --pdf-dir /path/to/pdfs \
  --manifest /path/to/manifest.json --output data_local/processed/custom
```

如果没有 manifest，省略该参数。自定义输出需要在检索配置的 `datasets` 中添加对应目录。
默认路径按仓库位置计算，手动传入的相对路径按当前终端目录计算。
输入文件不修改；扫描页不做 OCR，低文字量页记录警告。
参考文献、附录和双栏阅读顺序使用启发式识别，仍需按 PDF 核查。

### 检索输入

| 参数 | 含义 |
| --- | --- |
| `--query` | 必填问题/关键词，1–4000 字符；默认模型适合英文 |
| `--dataset` | 配置里的语料键；默认配置选择 `openalex` |
| `--method` | `bm25`（默认）、`dense`、`hybrid`、`hybrid_reranker` |
| `--top-k` | 最多返回几个 chunks，1–20，默认 5 |
| `--document-id` | 可重复指定，只搜索这些文档 |
| `--config` | 自定义配置文件；配置里的语料路径相对于仓库根目录 |

配置见 `config/retrieval.json`。dataset 从所选配置校验，不再写死为两个旧数据集。
保留的 SciFact/HotPotQA 键需要你自己准备完整 processed 数据；原 repo 附带的
HotPotQA demo 使用 `config/demo_retrieval.json`。

## 3. 系统输出

### 七个 PDF 切块产物

默认目录：`data_local/processed/openalex/`。JSONL 每行一个 JSON 对象。

| 文件 | 内容 |
| --- | --- |
| `documents.jsonl` | 文档 ID、标题、源 PDF 路径/hash、切块参数、版本、chunk 数量 |
| `pages.jsonl` | 页面原始文本块、行信息和 bbox |
| `normalized_texts.jsonl` | 清洗后的完整文本、段落 offsets、章节、页码和来源元数据 |
| `chunks.jsonl` | chunk ID、正文、正文 token 数、hash、页码和字符范围 |
| `id_map.jsonl` | 来源文档/chunk ID、章节、block IDs 和完整输入 token 数 |
| `quality_report.json` | 每篇处理结果、低文字量页、警告和失败原因 |
| `manifest.json` | 状态、参数/版本、数量和其他六个文件的 hash |

页码从 **1** 开始，表示 PDF 物理页，不一定等于论文印刷页。
字符范围是对应文档 normalized text 的 Python 下标，左闭右开：
`normalized_text[char_start:char_end] == chunk.text`。
`token_count` 只计正文；`id_map.jsonl` 的 `input_token_count` 计入标题和 special tokens。

相同输入、代码、依赖和参数重跑，chunk ID 和正文不变，时间字段会更新。
源内容、标题、解析/切块代码、tokenizer 或参数变化会产生新的索引版本。
合并到本仓库改变了实现版本，因此不要混用旧 `11.4` 的 chunk ID/缓存。

沿用原设计，document 的 `status` 暂为 `processing`、`embedding_model_id` 为 `null`；
本地 chunks 是否完成看 manifest 的 `complete` 和报告的 `chunked`。
本地检索单独管理向量缓存，不会把文档标为未来 AWS 服务的 `ready`。

### 检索 JSON 输出

```bash
mkdir -p outputs
python -m services.retrieval --dataset openalex --method bm25 \
  --query "How does retrieval augmented generation work?" --top-k 5 \
  > outputs/openalex_bm25.json
```

不加重定向时，JSON 直接输出到终端。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `retrieval_id`, `created_at` | 本次查询的 ID 和时间 |
| `query`, `method`, `mode`, `top_k` | 实际查询参数；当前 mode 为 baseline |
| `corpus_fingerprint` | 所用语料的指纹 |
| `eligible_unit_count` | 文档范围过滤后可检索的单元数 |
| `results` | 排序后的证据段，数量可能少于 top-k |
| `results[].document_id`, `chunk_id`, `document_title` | 来源身份和标题 |
| `results[].text`, `source_location`, `source` | 正文、PDF 页码/字符范围、原始来源 ID |
| `results[].rank`, `score`, `score_type`, `stage_scores` | 名次、分数类型和各阶段分数 |
| `setup_timings`, `query_ms` | 初始化和查询耗时 |
| `candidate_k`, `candidate_document_ids` | cross-encoder 候选信息；其他方法为 0/空数组 |
| `truncated_reranker_pairs` | 重排超长输入数量，其他方法为 0 |

分数类型为 `bm25`、`dense_cosine`、`rrf` 或 `cross_encoder_logit`，不能直接跨类型比较。
相同标题和正文的 chunks 会合并为检索单元，原 IDs 保存在 alias 映射中。
这里输出证据段，目前没有 LLM 生成的最终答案。

## 4. 验证全量切块和 BM25 结果

完成上面的全量切块和 BM25 搜索后，复制运行：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
p = Path('data_local/processed/openalex')
def rows(name):
    return [json.loads(line) for line in (p / name).read_text().splitlines() if line.strip()]
m = json.loads((p / 'manifest.json').read_text())
docs, chunks = rows('documents.jsonl'), rows('chunks.jsonl')
texts = {r['document_id']: r['text'] for r in rows('normalized_texts.jsonl')}
mapping = {r['chunk_id']: r for r in rows('id_map.jsonl')}
assert m['status'] == 'complete'
assert len(docs) == m['counts']['documents'] > 0
assert len(chunks) == m['counts']['chunks'] == len(mapping) > 0
assert len({c['chunk_id'] for c in chunks}) == len(chunks)
for name, expected in m['files'].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected
for c in chunks:
    loc = c['source_location']
    assert texts[c['document_id']][loc['char_start']:loc['char_end']] == c['text']
    assert hashlib.sha256(c['text'].encode()).hexdigest() == c['content_hash']
    assert 1 <= loc['page_start'] <= loc['page_end']
    assert c['token_count'] <= m['recipe']['chunk_size']
    assert mapping[c['chunk_id']]['input_token_count'] <= m['recipe']['max_input_tokens']
by_id = {c['chunk_id']: c for c in chunks}
result = json.loads(Path('outputs/openalex_bm25.json').read_text())
assert len(result['results']) == 5
for rank, hit in enumerate(result['results'], 1):
    original = by_id[hit['chunk_id']]
    assert hit['rank'] == rank and hit['score_type'] == 'bm25'
    assert hit['text'] == original['text']
    assert hit['source_location'] == original['source_location']
print('PASS:', len(docs), 'documents,', len(chunks), 'chunks, 5 valid search results')
q = json.loads((p / 'quality_report.json').read_text())
for d in q['documents']:
    if d.get('warnings'):
        print('REVIEW:', d['filename'], d['low_text_pages'])
PY
```

现有数据预期为 30 篇、537 页、2,441 个 chunks，去重后 2,436 个检索单元。
更换输入或配置后数量可能变化。`W3040807780.pdf` 有 11 个低文字量页，需人工核查。

## 5. 两篇 PDF 重跑一致性检查

```bash
python scripts/prepare_pdf_chunks.py --limit 2 --output data_local/test_first --overwrite
python scripts/prepare_pdf_chunks.py --limit 2 --output data_local/test_repeat --overwrite
python - <<'PY'
import json
from pathlib import Path
def content(folder):
    return [(r['chunk_id'], r['index_version'], r['text'], r['source_location'])
            for r in map(json.loads, (Path(folder) / 'chunks.jsonl').read_text().splitlines())]
assert content('data_local/test_first') == content('data_local/test_repeat')
print('PASS: identical chunk IDs, versions, text and source locations')
PY
```

试跑使用独立目录，不覆盖全量检索数据。

## 6. 文档范围过滤与原 demo

```bash
python - <<'PY'
import json
from pathlib import Path
from services.retrieval import LocalRetriever
d = json.loads(Path('data_local/processed/openalex/documents.jsonl').read_text().splitlines()[0])
r = LocalRetriever(dataset='openalex').search('retrieval', document_ids=[d['document_id']])
assert r['results'] and all(h['document_id'] == d['document_id'] for h in r['results'])
print('PASS: document scope filter')
PY
python -m services.retrieval --config config/demo_retrieval.json \
  --dataset hotpotqa --method bm25 --query "musical instrument" --top-k 3
```

## 7. 可选：神经检索

```bash
python -m pip install -r requirements.txt
python scripts/download_retrieval_models.py --config config/retrieval.json --method hybrid_reranker
python -m services.retrieval --dataset openalex --method dense --query "How is RAG evaluated?" --top-k 5
python -m services.retrieval --dataset openalex --method hybrid --query "How is RAG evaluated?" --top-k 5
python -m services.retrieval --dataset openalex --method hybrid_reranker --query "How is RAG evaluated?" --top-k 5
```

已有权重时，下载脚本加 `--local-only` 只检查本地文件。
首次 dense 查询会生成 `data_local/embeddings/local_retrieval/openalex/` 下的向量缓存，
相同数据/模型/配置再次查询会复用缓存。模型和查询使用同一个 embedding 配置。

## 8. 退出状态和限制

- ingestion `0`：所有文件成功，但可能有低文字量警告；`1`：部分失败，查看质量报告。
- 参数、路径、依赖、未知 dataset 或检索输入错误：退出码 `2`，错误信息在 stderr。
- retrieval 拒绝 PDF ingestion 的 `partial_failure` 或关键产物 hash 不一致的数据。
- 查询时不隐式下载模型；缺少模型需先明确运行下载脚本。
- 自动检查不能证明所有双栏、表格、公式都正确，仍需按来源页抽查。
