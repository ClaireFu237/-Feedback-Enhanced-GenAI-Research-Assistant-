# Feedback-Enhanced Research Assistant — Local Retrieval

## What changed

This update adds a standalone local Python retrieval/reranking module, a bundled demonstration corpus, pinned dependencies, and a model-preparation script. Four methods are available:

| Method | Description |
| --- | --- |
| `bm25` | Keyword retrieval using BM25Okapi |
| `dense` | Exact cosine search over normalized pretrained embeddings |
| `hybrid` | BM25 and dense candidates combined with Reciprocal Rank Fusion |
| `hybrid_reranker` | Hybrid candidates reordered by a pretrained cross-encoder |

The module runs without FastAPI, AWS credentials, model training, or answer generation.

## Included data

`demo_data/hotpotqa/` contains **997 documents and 997 paragraph chunks**, pooled from the existing 100-example HotpotQA validation/distractor subset. All prepared candidate paragraphs, including distractors, are included; each question searches the same shared corpus.

Document/chunk IDs, source mappings, original paragraph text, and character offsets are preserved. Exact title/body duplicates are merged at retrieval time with aliases retained; identical text under different titles remains distinct. This is a custom subset, not the official fullwiki benchmark. Answers, supporting facts, qrels, reports, and comparison outputs are excluded.

The [dataset README](demo_data/hotpotqa/README.md), manifest, attribution records, and [license notice](demo_data/hotpotqa/LICENSE.md) describe the data and its [HotpotQA source](https://hotpotqa.github.io/). The bundled data is distributed under CC BY-SA 4.0.

## Quick start: clone and run BM25

Use Python 3.12; validation used Python 3.12.2 on macOS x86_64. These commands use macOS/Linux shell syntax:

```bash
git clone https://github.com/ClaireFu237/-Feedback-Enhanced-GenAI-Research-Assistant-.git research-assistant
cd research-assistant
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-bm25.txt
mkdir -p outputs

.venv/bin/python -m services.retrieval \
  --config config/demo_retrieval.json \
  --dataset hotpotqa \
  --method bm25 \
  --query "Jalen Jones plays basketball for an NBA team that plays their home games in what arena?" \
  --top-k 5 \
  > outputs/bm25.json
```

Open `outputs/bm25.json` to inspect the ranked evidence. This sample should retrieve basketball-related passages; inspect the Jalen Jones and New Orleans Pelicans sources. It is a functional sample run, not an evaluation against answer labels.

If `python3` already points to Python 3.12, it can replace `python3.12`. On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python` and adapt the shell commands. No original sibling data directory or author-specific cache is needed.

## Run dense / hybrid / reranking

Install the full dependencies and explicitly download the pinned public models once:

```bash
.venv/bin/python -m pip install -r requirements-retrieval.txt
.venv/bin/python scripts/download_retrieval_models.py

.venv/bin/python -m services.retrieval \
  --config config/demo_retrieval.json \
  --dataset hotpotqa \
  --method hybrid_reranker \
  --query "What class of instrument does Apatim Majumdar play?" \
  --top-k 5 \
  > outputs/hybrid_reranker.json
```

Create `outputs/` first if you skipped the BM25 example. Replace the method with `dense` or `hybrid` to try either method. The example retains the question's spelling “Apatim”; the corpus spells the musician “Apratim Majumdar” and includes the instrument Sarod.

Dependencies and the first model download require internet access. Inference is local and never implicitly downloads models. The download script fetches model files, not datasets. Models are cached in `.model_cache/huggingface/hub/`; an explicit `HF_HUB_CACHE` overrides that location. Use `scripts/download_retrieval_models.py --local-only` to verify existing model files offline.

The first neural run builds corpus vectors; later runs reuse `data_local/embeddings/mentor_demo/`. Weights, vectors, virtual environments and `outputs/` are ignored by Git. The committed data is sufficient to rebuild the index.

## Inputs and outputs

**Input:** query text, method, top_k (1–20), and optional document IDs. Repeat `--document-id <document_id>` to limit the actual candidate scope before ranking. `config/demo_retrieval.json` selects the bundled corpus; the original `config/retrieval.json` remains for separately prepared developer datasets.

**Output:** JSON with a retrieval ID, corpus fingerprint, timing, and ranked `results`. Each result contains:

- `chunk_id`, `document_id`: stable IDs for future citations and feedback.
- `document_title`, `text`, `source_location`, `source`: title, original passage, character offsets, and provenance.
- `rank`, `score`, `score_type`, `stage_scores`: order and separately named method/stage scores.

Scores are ranking signals, not correctness probabilities. The output contains retrieved evidence, not a generated answer. The Python entry point is `LocalRetriever(dataset="hotpotqa", config_path="config/demo_retrieval.json").search(query=..., method=..., top_k=...)`, imported from `services.retrieval`.

## Configuration and limits

Models and exact revisions are pinned in `config/demo_retrieval.json`: MiniLM-L6-v2 for 384-dimensional mean-pooled, L2-normalized embeddings, and the MS MARCO MiniLM cross-encoder for raw pairwise logits. Dense encoding uses no query/document prefixes. BM25 casefolds Unicode word tokens without stemming or stop-word removal. Hybrid takes 50 candidates per branch with RRF constant 60; reranking scores 20 hybrid candidates before returning top_k.

Dense inputs are truncated at 256 tokens and cross-encoder pairs at 512 tokens. Original chunks remain unchanged. CPU reranking is slower than BM25/dense. Cache manifests record content/ID fingerprints, model revision, encoding settings, runtime versions and vector integrity checks. Changing the corpus or encoding configuration creates a new cache.

## Submission files

[FILES1016.txt](FILES1016.txt) lists the exact 21 files for this update: runtime modules, configuration, requirements, model setup, documentation, and the complete demonstration corpus with attribution. Reports, evaluation/test sources, model weights, generated caches and sample outputs are not required for the sample run. Existing architecture placeholder READMEs may remain from earlier commits.

Verification used a separate copy containing only these submission files, a fresh virtual environment with no inherited base packages, fresh model downloads, and a newly built vector cache. All four methods, document filtering and cache reuse passed. This verifies runnable packaging; it does not establish retrieval accuracy for new questions.
