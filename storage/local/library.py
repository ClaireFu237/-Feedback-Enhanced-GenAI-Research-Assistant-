"""Local library adapter: immutable corpus snapshots and atomic publication."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from services.retrieval.data import Corpus, read_jsonl


class LibraryStorage(Protocol):
    def snapshot(self) -> Path | None: ...

    def publish(self, source: Path) -> dict: ...


class LocalLibraryStorage:
    """First publication wins per document_id, including its original chunk IDs.

    Readers resolve the library symlink once. Old snapshots remain readable during
    publication; the pointer replacement commits all four files together.
    """

    names = ("documents.jsonl", "chunks.jsonl", "id_map.jsonl")

    def __init__(self, path):
        self.path = Path(path)

    def snapshot(self):
        if not self.path.exists():
            return None
        return self.path.resolve(strict=True)

    def publish(self, source):
        source = Path(source).resolve(strict=True)
        Corpus.load("publication", source)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Also serialize separate server processes publishing to this library.
        with (self.path.parent / ("." + self.path.name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self.snapshot()
            if current:
                Corpus.load("library", current)
            rows = {name: read_jsonl(current / name) if current else [] for name in self.names}
            existing = {r["document_id"] for r in rows["documents.jsonl"]}
            incoming = read_jsonl(source / "documents.jsonl")
            added = {r["document_id"] for r in incoming} - existing
            response = {"dataset": "library", "added_document_ids": sorted(added),
                        "skipped_document_ids": sorted({r["document_id"] for r in incoming} & existing)}
            if added:
                if self.path.exists() and not self.path.is_symlink():
                    raise ValueError("Library path must be managed by the library storage adapter")
                versions = self.path.parent / ("." + self.path.name + "-versions")
                versions.mkdir(exist_ok=True)
                staging = versions / uuid.uuid4().hex
                staging.mkdir()
                pointer = self.path.parent / ("." + self.path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    hashes = {}
                    for name in self.names:
                        rows[name].extend(r for r in read_jsonl(source / name) if r["document_id"] in added)
                        content = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows[name])
                        with (staging / name).open("wb") as stream:
                            stream.write(content.encode("utf-8"))
                            stream.flush()
                            os.fsync(stream.fileno())
                        hashes[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    manifest = {"schema_version": 1, "stage": "library", "status": "complete",
                                "created_at": datetime.now(timezone.utc).isoformat(), "files": hashes,
                                "counts": {"documents": len(rows["documents.jsonl"]),
                                           "chunks": len(rows["chunks.jsonl"])}}
                    with (staging / "manifest.json").open("w", encoding="utf-8") as stream:
                        json.dump(manifest, stream, ensure_ascii=False, allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    Corpus.load("library", staging)
                    pointer.symlink_to(staging.relative_to(self.path.parent), target_is_directory=True)
                    pointer.replace(self.path)
                except BaseException:
                    pointer.unlink(missing_ok=True)
                    if self.snapshot() != staging:
                        shutil.rmtree(staging)
                    raise
            response.update(document_count=len(rows["documents.jsonl"]), chunk_count=len(rows["chunks.jsonl"]))
            return response
