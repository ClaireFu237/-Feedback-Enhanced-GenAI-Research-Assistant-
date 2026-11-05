# 如何自己运行测试

这里不做检索或答案质量 evaluation，只检查 PDF → chunks 能否正常运行、输出是否一致。
下面命令均在 `11.4` 目录执行。

## 1. 准备环境

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.4"
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_pdf_chunks.py --help
```

如果尚未创建虚拟环境，先按 README 安装。
如果没有 tokenizer 缓存，在下面的首次运行命令中加 `--download-tokenizer`。

## 2. 两篇 PDF 冒烟测试

```bash
python prepare_pdf_chunks.py --limit 2 --output data_local/test_run --overwrite
```

预期终端最后显示 `complete: 2 documents, ... chunks`，退出码为 `0`。
`--overwrite` 允许重复执行这组命令。

## 3. 检查输出内容

下面这段代码可以直接复制到终端：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('data_local/test_run')
def rows(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]

manifest = json.loads((root / 'manifest.json').read_text())
docs = rows('documents.jsonl')
chunks = rows('chunks.jsonl')
texts = {r['document_id']: r for r in rows('normalized_texts.jsonl')}
mapping = {r['chunk_id']: r for r in rows('id_map.jsonl')}
page_counts = {}
for page in rows('pages.jsonl'):
    key = page['document_id']
    page_counts[key] = max(page_counts.get(key, 0), page['page_number'])

assert manifest['status'] == 'complete'
assert len(docs) == manifest['counts']['documents'] == 2
assert len(chunks) == manifest['counts']['chunks'] > 0
assert len({c['chunk_id'] for c in chunks}) == len(chunks) == len(mapping)
assert set(texts) == {d['document_id'] for d in docs}
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
for c in chunks:
    loc = c['source_location']
    assert c['text'].strip()
    assert c['content_hash'] == hashlib.sha256(c['text'].encode()).hexdigest()
    assert texts[c['document_id']]['text'][loc['char_start']:loc['char_end']] == c['text']
    assert 1 <= loc['page_start'] <= loc['page_end'] <= page_counts[c['document_id']]
    assert c['token_count'] <= manifest['recipe']['chunk_size']
    assert mapping[c['chunk_id']]['input_token_count'] <= manifest['recipe']['max_input_tokens']
for d in docs:
    own = [c for c in chunks if c['document_id'] == d['document_id']]
    assert len(own) == d['chunk_count']
    assert [c['chunk_index'] for c in own] == list(range(len(own)))
print(f"PASS: {len(docs)} documents, {len(chunks)} chunks; hashes, offsets, pages and token limits checked")
PY
```

## 4. 检查重复运行的稳定性

```bash
python prepare_pdf_chunks.py --limit 2 --output data_local/test_repeat --overwrite
python - <<'PY'
import json
from pathlib import Path
def contents(folder):
    path = Path(folder) / 'chunks.jsonl'
    return [(r['chunk_id'], r['index_version'], r['text'], r['source_location'])
            for r in map(json.loads, path.read_text().splitlines())]
assert contents('data_local/test_run') == contents('data_local/test_repeat')
print('PASS: repeated runs produce identical chunk IDs, versions, text and locations')
PY
```

## 5. 运行全部现有 PDF

```bash
python prepare_pdf_chunks.py --overwrite
python - <<'PY'
import json
from pathlib import Path
p = Path('data_local/processed/openalex')
m = json.loads((p / 'manifest.json').read_text())
q = json.loads((p / 'quality_report.json').read_text())
assert m['status'] == 'complete'
assert m['counts']['documents'] == 30
assert q['failed_document_count'] == 0
print('PASS:', m['counts'])
for d in q['documents']:
    if d.get('warnings'):
        print('REVIEW:', d['filename'], d['low_text_pages'], d['warnings'])
PY
```

页码正确、文字来源正确仍需人工抽查：在 `chunks.jsonl` 选一条，按 `page_start/page_end`
打开对应 PDF，检查双栏顺序和文本内容。自动检查通过并不代表表格、公式和所有版式都解析完美。
