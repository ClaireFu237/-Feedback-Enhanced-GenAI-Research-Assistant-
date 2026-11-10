# Local Retrieval Module

See the [project README](../../README.md) for the fresh-clone installation, bundled data, model preparation, CLI examples and output fields.

## Runtime Files

- `data.py`: load searchable corpus and provenance, preserve IDs/aliases, fingerprint data, and apply document scope.
- `engine.py`: build BM25 statistics, cache dense vectors, fuse ranks, and rerank hybrid candidates.
- `neural.py`: inference-only Transformers adapters using mean pooling/L2 normalization and cross-encoder logits.
- `__main__.py`: JSON-producing CLI, invoked with `python -m services.retrieval`.
- `__init__.py`: expose LocalRetriever for independent Python callers.

Only corpus documents, chunks, ID maps and manifests are loaded. Existing schema field names for query, top_k, document_ids, source locations and stable IDs are preserved; method-specific scores are named separately rather than being stored as a cosine score.

For the portable mentor demo, pass `--config config/demo_retrieval.json --dataset hotpotqa`. Original developer datasets remain accessible through `config/retrieval.json` when their local processed files are available.

`hybrid_reranker` scores the **entire BM25/dense candidate union** before
returning `top_k`. The configuration uses `reranker.candidate_scope: full_union`;
the former `reranker.candidate_k` cutoff is obsolete and has no effect.
Candidate depth is controlled by `hybrid.bm25_candidate_k` and
`hybrid.dense_candidate_k` (50 each by default). Raising these values also
increases cross-encoder CPU work.

`retrieval_ms`, `reranker_ms` and `query_ms` report query timing.
Model and index preparation is recorded separately in `setup_timings`.
