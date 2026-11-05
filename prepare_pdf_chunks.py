#!/usr/bin/env python3
"""Prepare local PDFs as positioned text and reproducible, token-bounded chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PDFS = ROOT.parent / "data/openalex/pdf"
DEFAULT_MANIFEST = ROOT.parent / "data/openalex/manifest.json"
VERSION = "pdf-to-chunks-v1"
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:capstone:feedback-enhanced-research-assistant")
OUTPUTS = ("documents.jsonl", "pages.jsonl", "normalized_texts.jsonl", "chunks.jsonl",
           "id_map.jsonl", "quality_report.json", "manifest.json")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(prefix, value):
    return prefix + "_" + str(uuid.uuid5(NAMESPACE, value))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=Path, help="Process one PDF")
    source.add_argument("--pdf-dir", type=Path, help="PDF directory (non-recursive); defaults to ../data/openalex/pdf")
    parser.add_argument("--manifest", type=Path, help="Optional OpenAlex JSON manifest; auto-used for the default PDF folder")
    parser.add_argument("--output", type=Path, default=ROOT / "data_local/processed/openalex")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted PDFs")
    parser.add_argument("--chunk-size", type=int, default=200, help="Maximum body tokens (default: 200)")
    parser.add_argument("--overlap", type=int, default=30, help="Maximum overlap tokens (default: 30)")
    parser.add_argument("--max-input-tokens", type=int, default=256, help="Title + body + special-token limit")
    parser.add_argument("--include-references", action="store_true", help="Also chunk detected bibliography sections")
    parser.add_argument("--tokenizer-file", type=Path, help="Use a local Hugging Face tokenizer.json")
    parser.add_argument("--download-tokenizer", action="store_true", help="Allow downloading the pinned MiniLM tokenizer")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in an existing output folder")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (0 <= args.overlap < args.chunk_size) or args.max_input_tokens < 4:
        parser.error("Require 0 <= overlap < chunk-size and max-input-tokens >= 4")
    return args


def inputs(args):
    directory = (args.pdf_dir or DEFAULT_PDFS).resolve()
    if args.pdf:
        files = [args.pdf.resolve()]
        directory = files[0].parent
    else:
        if not directory.is_dir():
            raise ValueError(f"PDF directory does not exist: {directory}")
        files = sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No PDF files found")
    if len(files) != len(set(files)):
        raise ValueError("Input contains duplicate PDF paths")
    if any(not p.is_file() or p.suffix.lower() != ".pdf" for p in files):
        raise ValueError("Every input must be an existing .pdf file")
    metadata_path = args.manifest
    if metadata_path is None and directory == DEFAULT_PDFS.resolve() and DEFAULT_MANIFEST.is_file():
        metadata_path = DEFAULT_MANIFEST
    metadata = {}
    if metadata_path:
        metadata_path = metadata_path.resolve()
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Input manifest must be a JSON list of document records")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str) or not row["doc_id"].strip():
                raise ValueError("Each input manifest record must have a nonempty string doc_id")
            if row["doc_id"] in metadata:
                raise ValueError(f"Duplicate doc_id in input manifest: {row['doc_id']}")
            metadata[row["doc_id"]] = row
    return files, metadata, metadata_path


def process_document(path, metadata, budget, recipe, timestamp, output, extract_pdf, make_chunks):
    pdf_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = metadata.get("pdf_sha256")
    if expected and pdf_hash != expected:
        raise ValueError("PDF SHA-256 differs from input manifest")
    title = metadata.get("title") or path.stem
    if not isinstance(title, str) or not title.strip() or len(title) > 500:
        raise ValueError("Document title must contain 1–500 characters")
    title = title.strip()
    source_id = metadata.get("doc_id") or "sha256:" + pdf_hash
    source = "openalex" if metadata.get("source") == "openalex" or metadata.get("openalex_id") else "local-pdf"
    document_id = stable_id("doc", f"{source}:document:{source_id}")
    version = VERSION + "-" + digest(json_text({"pdf_hash": pdf_hash, "title": title, "recipe": recipe}))[:24]
    pages, text, paragraphs, report = extract_pdf(path, recipe["include_references"])
    raw_chunks = make_chunks(text, paragraphs, title, budget)
    chunks, mappings = [], []
    for index, item in enumerate(raw_chunks):
        chunk_id = stable_id("chk", f"{document_id}:{version}:{index}")
        chunks.append({"chunk_id": chunk_id, "document_id": document_id, "index_version": version,
                       "chunk_index": index, "text": item["text"], "source_location": item["source_location"],
                       "content_hash": digest(item["text"]), "token_count": item["token_count"],
                       "created_at": timestamp, "schema_version": 1})
        mappings.append({"source_document_id": source_id, "document_id": document_id,
                         "source_chunk_id": f"{source_id}:{version}:chunk:{index}", "chunk_id": chunk_id,
                         "source": source, "index_version": version, "section": item["section"],
                         "block_ids": item["block_ids"], "input_token_count": item["input_token_count"]})
    document = {
        "document_id": document_id, "title": title, "filename": path.name,
        "content_type": "application/pdf", "size_bytes": path.stat().st_size,
        # The shared 10.19 schema reserves 'ready' for a complete embedded index.
        "status": "processing", "source_object_key": str(path), "content_hash": pdf_hash,
        "processed_text_key": f"local-jsonl:{output / 'normalized_texts.jsonl'}#document_id={document_id}",
        "ingestion_version": VERSION, "chunking_config": {
            "chunk_size": budget.chunk_size, "unit": "tokens", "overlap": budget.overlap,
            "parser_version": recipe["parser_version"], "tokenizer_version": budget.fingerprint},
        "embedding_model_id": None, "index_version": version, "chunk_count": len(chunks),
        "processing_attempt_id": version, "error": None,
        "created_at": timestamp, "updated_at": timestamp, "schema_version": 1}
    normalized = {"document_id": document_id, "index_version": version, "text": text,
                  "paragraphs": paragraphs, "source_document_id": source_id,
                  "source_metadata": {k: metadata[k] for k in ("doi", "authors", "publication_date",
                                      "landing_page_url", "pdf_source_url", "best_oa_license") if k in metadata}}
    page_rows = [{"document_id": document_id, "index_version": version, **p} for p in pages]
    report.update({"filename": path.name, "document_id": document_id, "source_document_id": source_id,
                   "status": "chunked", "chunk_count": len(chunks), "normalized_char_count": len(text),
                   "max_body_tokens": max(c["token_count"] for c in chunks),
                   "max_input_tokens": max(m["input_token_count"] for m in mappings)})
    return document, page_rows, normalized, chunks, mappings, report


def run(args):
    # Keep --help usable even before installing dependencies.
    import pymupdf
    import tokenizers
    from pdf_chunks.chunk import TokenBudget, VERSION as CHUNK_VERSION, make_chunks
    from pdf_chunks.extract import VERSION as PARSER_VERSION, extract_pdf

    files, metadata, input_manifest = inputs(args)
    output = args.output.resolve()
    if input_manifest and (output / "manifest.json").resolve() == input_manifest:
        raise ValueError("Output would overwrite the input manifest; choose a separate output directory")
    if output.exists() and not output.is_dir():
        raise ValueError("--output must be a directory")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("Output directory is not empty. Use --overwrite or choose a new --output")
    for name in OUTPUTS:
        target = output / name
        if target.is_symlink() or target.is_dir():
            raise ValueError(f"Refusing output symlink or directory: {target}")
    budget = TokenBudget(args.tokenizer_file, args.download_tokenizer, args.chunk_size,
                         args.overlap, args.max_input_tokens)
    recipe = {"pipeline_version": VERSION, "parser_version": PARSER_VERSION,
              "implementation_sha256": digest("".join(p.read_text(encoding="utf-8") for p in
                  [Path(__file__), ROOT / "pdf_chunks/extract.py", ROOT / "pdf_chunks/chunk.py"])),
              "pymupdf_version": pymupdf.VersionBind, "chunker_version": CHUNK_VERSION,
              "tokenizers_version": tokenizers.__version__, "tokenizer_sha256": budget.fingerprint,
              "chunk_size": args.chunk_size, "overlap": args.overlap,
              "max_input_tokens": args.max_input_tokens, "include_references": args.include_references,
              "input_text_recipe": "title-double-newline-body"}
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    reports, counts, seen_docs, seen_chunks = [], {"documents": 0, "pages": 0, "chunks": 0}, set(), set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pdf-chunks-", dir=output.parent) as temporary:
        staging = Path(temporary)
        streams = {name: (staging / name).open("w", encoding="utf-8") for name in OUTPUTS if name.endswith(".jsonl")}
        try:
            for i, path in enumerate(files, 1):
                print(f"[{i}/{len(files)}] {path.name}", flush=True)
                try:
                    document, pages, text, chunks, mappings, report = process_document(
                        path, metadata.get(path.stem, {}), budget, recipe, timestamp, output, extract_pdf, make_chunks)
                    if document["document_id"] in seen_docs:
                        raise ValueError("Duplicate document identity (same PDF content or source ID)")
                    if any(c["chunk_id"] in seen_chunks for c in chunks):
                        raise ValueError("Duplicate chunk identity")
                except Exception as exc:
                    reports.append({"filename": path.name, "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                    continue
                seen_docs.add(document["document_id"])
                seen_chunks.update(c["chunk_id"] for c in chunks)
                batches = {"documents.jsonl": [document], "pages.jsonl": pages,
                           "normalized_texts.jsonl": [text], "chunks.jsonl": chunks, "id_map.jsonl": mappings}
                for name, rows in batches.items():
                    for row in rows:
                        streams[name].write(json_text(row) + "\n")
                counts["documents"] += 1
                counts["pages"] += len(pages)
                counts["chunks"] += len(chunks)
                reports.append(report)
                print(f"  {len(pages)} pages -> {len(chunks)} chunks" +
                      (f"; review low-text pages {report['low_text_pages']}" if report["low_text_pages"] else ""), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        failures = sum(r["status"] == "failed" for r in reports)
        status = "partial_failure" if failures else "complete"
        quality = {"status": status, "documents": reports, "failed_document_count": failures,
                   "warning_document_count": sum(bool(r.get("warnings")) for r in reports)}
        (staging / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {"schema_version": 1, "stage": "pdf_to_chunks", "status": status,
                    "created_at": timestamp, "recipe": recipe, "counts": counts,
                    "attempted_document_count": len(files), "failed_document_count": failures,
                    "input_manifest": str(input_manifest) if input_manifest else None,
                    "files": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                              for name in OUTPUTS if name != "manifest.json"}}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output.mkdir(exist_ok=True)
        # All processing finishes before replacing artifacts; publish manifest last.
        for name in OUTPUTS:
            os.replace(staging / name, output / name)
    print(f"{status}: {counts['documents']} documents, {counts['chunks']} chunks. Output: {output}")
    return 1 if failures else 0


def main():
    args = arguments()
    try:
        return run(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: python -m pip install -r {ROOT / 'requirements.txt'}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
