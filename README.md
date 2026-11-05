# PDF → chunks

只完成本地 PDF 文本抽取、清洗、切块和来源记录。不调用 AWS、LLM 或 embedding API。
无需依赖 `10.19` 中的代码。需要 Python 3.10 或以上版本。

## 安装与运行

```bash
cd "/Users/fuclaire/Desktop/jhu 26fall/capstone/code/11.4"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 处理 ../data/openalex/pdf 中的全部 PDF，自动读取对应的 manifest。
python prepare_pdf_chunks.py
```

默认使用本地缓存中的 `all-MiniLM-L6-v2` tokenizer，固定模型 revision 为
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`。
如果提示 tokenizer 不存在，首次运行添加 `--download-tokenizer`，只下载 tokenizer，不下载模型权重：

```bash
python prepare_pdf_chunks.py --download-tokenizer
```

也可以用 `--tokenizer-file /path/to/tokenizer.json` 指定已有 tokenizer 文件，完全离线运行。
下载行为使用 Hugging Face 的默认缓存位置；后续处理无需网络。

程序默认路径相对于本项目的位置解析，不依赖当前终端目录；手动传入的相对路径则相对于终端目录。

## 常见用法

```bash
# 少量试跑，独立保存输出。
python prepare_pdf_chunks.py --limit 2 --output data_local/smoke

# 重跑并替换这一输出目录内的七个生成文件。
python prepare_pdf_chunks.py --overwrite

# 单篇 PDF；默认 OpenAlex 目录下的文件仍会自动匹配元数据。
python prepare_pdf_chunks.py \
  --pdf ../data/openalex/pdf/W4411630045.pdf \
  --output data_local/single

# 自己的 PDF 文件夹，不需要 OpenAlex manifest，非递归读取 .pdf 文件。
python prepare_pdf_chunks.py --pdf-dir /path/to/pdfs --output data_local/custom

# 显式指定元数据文件，格式沿用现有 OpenAlex manifest 的 JSON 数组。
python prepare_pdf_chunks.py --pdf-dir /path/to/pdfs \
  --manifest /path/to/manifest.json --output data_local/custom

# 把检测出的 References / Bibliography 也加入 chunks。
python prepare_pdf_chunks.py --include-references --output data_local/with_references
```

非空输出目录必须显式添加 `--overwrite`；该参数只替换约定的生成文件，不清空其他内容。
所有文档处理和暂存完成后才发布输出，最后写入 manifest；它不是跨多个文件的数据库事务。
输入 PDF 和输入 manifest 不会被修改。

## 切块规则

- 按页保存文本块、原始文本、行信息和坐标，使用几何规则重建常见双栏论文的阅读顺序。
- 清洗重复页眉页脚、页面边缘的纯页码和 arXiv 侧边标记，合并段落内换行、修复常见断词和连字。
- 启发式识别章节，默认排除参考文献；检测到附录或 supplementary material 后恢复索引。
- chunk 尽量在段落或句子边界结束，超长段落按 tokenizer 的原始字符 offsets 拆分。
- 默认正文上限 **200 tokens**，重叠上限 **30 tokens**；实际重叠可因句子和章节边界而减少或为零。
- 同时验证 `title + "\n\n" + chunk.text` 加 special tokens 后不超过 **256 tokens**。
  三个参数分别为 `--chunk-size`、`--overlap`、`--max-input-tokens`。
- chunk 始终是保存的 normalized text 的精确切片；章节之间不做 overlap。
- 文本来源、PDF hash、标题、实现代码、解析器/tokenizer 版本和配置共同确定索引版本。
  相同输入与环境重跑得到相同 chunk ID 和文本；运行时间字段会变化。

切换 tokenizer 时，应同时设置对应的 `--max-input-tokens`，不要把 MiniLM 的 256 当作通用模型限制。
`token_count` 是正文 token 数，不包含标题和 special tokens；完整输入的数量见 `id_map.jsonl`。

## 输出

默认输出目录：`data_local/processed/openalex/`。

| 文件 | 内容 |
| --- | --- |
| `documents.jsonl` | 文档 ID、标题、PDF hash、切块配置、版本和 chunk 数量 |
| `pages.jsonl` | 每页原始文本块、行、bbox；保留清洗前文本以便核查 |
| `normalized_texts.jsonl` | 完整清洗文本、段落 offsets、页码、章节、是否参与切块和 OpenAlex 来源信息 |
| `chunks.jsonl` | chunk ID、正文、token 数、hash、页码和字符范围 |
| `id_map.jsonl` | chunk 对应的 OpenAlex/本地来源 ID、章节、block IDs 和完整输入 token 数 |
| `quality_report.json` | 每篇处理结果、低文字量页面、清洗统计、警告和错误 |
| `manifest.json` | 参数、版本、数量、完成状态及其余输出文件的 SHA-256 |

`source_location.page_start/page_end` 是从 **1** 开始的 PDF 物理页码，不是论文印刷页码。
`char_start/char_end` 是该文档 normalized text 的 Python 字符下标，左闭右开；
`normalized_text[char_start:char_end] == chunk.text`。
`block_id` 在一篇文档内标识原始 PDF 块；查找时同时使用 document ID 和 index version。

document/chunk 的核心字段沿用 `10.19` 的格式，`id_map.jsonl` 可供原有 corpus loader 使用。
本项目完成到 `chunked` 阶段；为了符合原有文档 schema，`documents.jsonl` 的 `status`
保留为 `processing`，`embedding_model_id` 为 `null`，不会把尚未生成 embedding 的文档标成 `ready`。
本阶段是否完成请看 `manifest.json` 的 `status` 和质量报告中的 `chunked`。

## 异常与限制

- 低文字量页记录警告，仍保存可抽取文本；扫描页不会自动 OCR。
- 损坏、加密、无可用文字、hash 不符或切块失败的 PDF 记入质量报告；其余 PDF 继续处理。
  失败文档不会留下半份 chunks。
- 退出码：`0` 全部处理成功，`1` 有文档失败，`2` 参数、依赖或文件配置错误。
  `0` 仍可能伴随低文字量警告，需要查看质量报告。
- 多栏排序、章节识别和断词修复均为启发式；复杂表格、公式、图中文字不保证完整恢复。
  References 识别也可能有误判，可用 `--include-references` 对照原始文本检查。
- 原始数据中的 `W3040807780.pdf` 有多页文字稀少，应人工检查再决定是否用于后续问答。

自己运行检查的步骤见 [RUN_TESTS.md](RUN_TESTS.md)。
