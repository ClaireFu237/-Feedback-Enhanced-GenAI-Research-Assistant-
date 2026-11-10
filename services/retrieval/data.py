"""Load searchable corpus records and source provenance."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def fingerprint(value):
    """Hash a canonical JSON value without platform-dependent formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    """Read a UTF-8 JSONL corpus or auxiliary provenance file."""
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_config(path=None):
    """Resolve all project-relative paths against the project root."""
    config = json.loads(Path(path or PROJECT / 'config/retrieval.json').read_text())
    if config['schema_version'] != 1:
        raise ValueError('Unsupported retrieval configuration version')
    hybrid = config['hybrid']
    if min(hybrid['bm25_candidate_k'], hybrid['dense_candidate_k']) < 20:
        raise ValueError('Each branch must supply at least the maximum public top_k (20)')
    if config['reranker'].get('candidate_scope', 'full_union') != 'full_union' or hybrid['rrf_constant'] <= 0:
        raise ValueError('Require positive RRF constant and full_union reranker candidate scope')
    if config['dense']['normalize_embeddings'] is not True:
        raise ValueError('This engine requires normalized cosine embeddings')
    return config


@dataclass
class Corpus:
    """Canonical units plus lossless aliases, document lookups, and provenance."""
    dataset: str
    units: list
    original_unit_count: int
    document_aliases: dict
    fingerprint: str
    manifest: dict
    scope_ids: list

    @classmethod
    def load(cls, dataset, folder, text_recipe='title-double-newline-body-v1'):
        # Pin one immutable library snapshot across all file reads.
        folder = Path(folder).resolve()
        if not (folder / 'manifest.json').is_file():
            raise ValueError(f'Corpus is not prepared: {folder}. Run scripts/prepare_pdf_chunks.py for PDF corpora, or select a prepared dataset/config.')
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest.get('stage') in ('pdf_to_chunks', 'text_to_chunks', 'library'):
            if manifest.get('status') != 'complete':
                raise ValueError('PDF ingestion is incomplete; inspect quality_report.json and rerun ingestion')
            # A failed/overlapping publication must not silently mix old and new
            # document/chunk identities. Legacy demo manifests keep their format.
            for name in ('documents.jsonl', 'chunks.jsonl', 'id_map.jsonl'):
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest['files'][name]:
                    raise ValueError(f'Ingestion artifact checksum mismatch: {name}; rerun ingestion')
        documents = read_jsonl(folder / 'documents.jsonl')
        chunks = read_jsonl(folder / 'chunks.jsonl')
        if not documents or not chunks:
            raise ValueError('Searchable corpus must contain documents and chunks')
        docs = {d['document_id']: d for d in documents}
        if len(docs) != len(documents) or len({r['chunk_id'] for r in chunks}) != len(chunks):
            raise ValueError('Duplicate stable IDs in input corpus')
        mapping_rows = read_jsonl(folder / 'id_map.jsonl')
        provenance = {r['chunk_id']: r for r in mapping_rows}
        if text_recipe not in ('title-double-newline-body-v1', 'title-section-body-v1'):
            raise ValueError('Unsupported text recipe')
        if len(provenance) != len(mapping_rows) or set(provenance) != {r['chunk_id'] for r in chunks}:
            raise ValueError('Incomplete or duplicate source-ID mapping')
        groups = {}
        for chunk in chunks:
            doc = docs.get(chunk['document_id'])
            if doc is None or provenance[chunk['chunk_id']]['document_id'] != chunk['document_id']:
                raise ValueError('Broken document or source-ID mapping')
            if not chunk['text'].strip():
                raise ValueError('Empty searchable text')
            if hashlib.sha256(chunk['text'].encode()).hexdigest() != chunk['content_hash']:
                raise ValueError('Chunk text hash mismatch')
            # Preserve distinct titles even when short paragraphs are identical.
            # Exact duplicated title/body pairs share one retrieval unit.
            key = (doc['title'], chunk['text'])
            row = {**chunk, 'document_title': doc['title'], 'source': {
                'dataset': dataset,
                'source_document_id': provenance[chunk['chunk_id']]['source_document_id'],
                'source_chunk_id': provenance[chunk['chunk_id']]['source_chunk_id']}}
            section = str(provenance[chunk['chunk_id']].get('section') or '').strip()
            # Known extractor artifacts are voltage labels, not section headings.
            row['section'] = '' if section == 'Unsectioned' or re.fullmatch(r'[+-]?\d+(?:\.\d+)?\s*V', section) else section
            groups.setdefault(key, []).append(row)
        units, alias_map = [], {}
        for aliases in groups.values():
            aliases.sort(key=lambda r: r['chunk_id'])
            canonical = dict(aliases[0])
            canonical['aliases'] = aliases
            canonical['search_text'] = canonical['document_title'] + '\n\n' + canonical['text']
            if text_recipe == 'title-section-body-v1' and canonical['section']:
                canonical['search_text'] = canonical['document_title'] + '\n\n' + canonical['section'] + '\n\n' + canonical['text']
            units.append(canonical)
        units.sort(key=lambda r: r['chunk_id'])
        for i, unit in enumerate(units):
            for alias in unit['aliases']:
                alias_map.setdefault(alias['document_id'], set()).add(i)
        payload = [{'chunk_id': r['chunk_id'], 'index_version': r['index_version'],
                    'search_text': r['search_text'], 'source_location': r['source_location'],
                    'aliases': [{'chunk_id': a['chunk_id'], 'document_id': a['document_id'], 'source': a['source']} for a in r['aliases']]}
                   for r in units]
        identity = fingerprint({'dataset': dataset, 'deduplication': 'exact-title-and-text-v1', 'units': payload})
        return cls(dataset, units, len(chunks), alias_map, identity, manifest, sorted(docs))

    def eligible(self, document_ids=None):
        """Apply document scope before any branch selects its candidates."""
        if document_ids is None:
            return list(range(len(self.units)))
        if not isinstance(document_ids, list) or not all(isinstance(d, str) for d in document_ids):
            raise ValueError('document_ids must be a list of strings')
        if not document_ids or len(document_ids) != len(set(document_ids)):
            raise ValueError('document_ids must be a nonempty list of unique IDs')
        unknown = set(document_ids) - set(self.document_aliases)
        if unknown:
            raise ValueError('Unknown document IDs: ' + ', '.join(sorted(unknown)))
        return sorted(set().union(*(self.document_aliases[d] for d in document_ids)))

    def display(self, index, document_ids=None):
        """Return an in-scope original identity while retaining canonical aliases."""
        unit = self.units[index]
        selected = unit if document_ids is None else next(a for a in unit['aliases'] if a['document_id'] in document_ids)
        return {k: selected[k] for k in ['chunk_id', 'document_id', 'document_title', 'text', 'source_location', 'source']}

    def mapping(self):
        """Export deduplication decisions without discarding original stable IDs."""
        return [{'canonical_chunk_id': r['chunk_id'], 'canonical_document_id': r['document_id'],
                 'aliases': [{'chunk_id': a['chunk_id'], 'document_id': a['document_id'], 'source': a['source']} for a in r['aliases']]}
                for r in self.units]
