# Bundled HotpotQA Retrieval Corpus

This package includes all 997 existing prepared documents/chunks, including distractors, so a fresh clone can search locally without the original sibling data directory. Original chunk IDs, document IDs, text and source positions are retained. Answers, supporting facts and qrels are excluded.

- documents.jsonl: schema-compatible metadata with portable local text locators.
- chunks.jsonl: complete searchable paragraph records and source character offsets.
- id_map.jsonl: project IDs mapped to original prepared dataset IDs.
- attribution.jsonl and LICENSE.md: upstream source and data attribution.
- manifest.json: corpus provenance, counts, file sizes and SHA-256 checksums.

Runtime deduplication uses exact title/body pairs and preserves aliases; identical body text under different titles remains distinct. The prepared metadata's processing state reflects unpublished cloud ingestion; the standalone local retriever constructs its own index and does not require cloud lifecycle publication.

Use config/demo_retrieval.json from the project root. This is the complete local pooled-context subset, not the full HotpotQA dataset or the official fullwiki setting. See the [project README](../../README.md) for installation and commands.
