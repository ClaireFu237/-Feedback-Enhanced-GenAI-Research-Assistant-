# Local PDF ingestion

Implemented by merging the standalone `11.4` pipeline into this repository.

- `extract.py`: PDF blocks, reading order, normalization, sections and page warnings.
- `chunk.py`: exact normalized-text slices, tokenizer budgets and overlap.
- `pipeline.py`: metadata, stable IDs/versions, JSONL outputs and reports.
- Repository CLI: `scripts/prepare_pdf_chunks.py`.

From the repository root, run `python scripts/prepare_pdf_chunks.py --overwrite`,
then `python -m services.retrieval --dataset openalex --query "retrieval"`.
Default source: `../data/openalex/pdf/`; default output: `data_local/processed/openalex/`.
No code is imported from the old `11.4` folder.

Default limits: 200 body tokens, 30 overlap tokens, and 256 total tokens including
paper title and special tokens. All chunks preserve page and normalized-text offsets.
OCR and full table/formula reconstruction are not implemented. Review page warnings.

See [RUN_TESTS_AND_IO.md](../../RUN_TESTS_AND_IO.md) for usage, status semantics,
input/output fields and runnable checks.
