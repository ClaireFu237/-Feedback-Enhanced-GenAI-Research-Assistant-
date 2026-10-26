"""BM25, normalized dense search, RRF, and candidate-only cross-encoder ranking."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .data import PROJECT, Corpus, fingerprint, load_config

METHODS = ('bm25', 'dense', 'hybrid', 'hybrid_reranker')


def tokenize(text):
    """Casefold Unicode word tokens; no stemming or stop-word removal."""
    return re.findall(r'\w+', text.casefold(), flags=re.UNICODE)


def fuse_rankings(rankings, constant):
    """Fuse ordered stable unit IDs using reciprocal ranks, never raw scores."""
    scores = {}
    for ranking in rankings:
        # An accidental repeated unit in one branch contributes only once.
        seen = set()
        for rank, unit_id in enumerate(ranking, 1):
            if unit_id not in seen:
                scores[unit_id] = scores.get(unit_id, 0.0) + 1.0 / (constant + rank)
                seen.add(unit_id)
    return scores


def model_cache_root(configured=None):
    """Choose the explicit environment cache, project cache, or legacy cache."""
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE']).expanduser().resolve()
    if configured:
        return (PROJECT / configured).resolve()
    return Path.home() / '.cache/huggingface/hub'


def resolve_model(model, cache_root=None):
    """Require a pinned local snapshot; never download models implicitly."""
    cache = model_cache_root(cache_root)
    path = cache / ('models--' + model['model_id'].replace('/', '--')) / 'snapshots' / model['revision']
    if not (path / 'config.json').is_file() or not (path / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Model snapshot is unavailable locally: {path}. Download this exact revision explicitly before running neural methods.')
    return path


class LocalRetriever:
    """Independent synchronous retriever; no API server, LLM, or cloud clients."""
    def __init__(self, dataset='scifact', config_path=None, config=None, corpus=None):
        self.config = config if config is not None else load_config(config_path)
        self.corpus = corpus or Corpus.load(dataset, PROJECT / self.config['datasets'][dataset])
        if self.config['text_recipe'] != 'title-double-newline-body-v1' or self.config['deduplication'] != 'exact-title-and-text-v1':
            raise ValueError('Unsupported text or deduplication recipe')
        self.timings = {}
        self.bm25 = None
        self.encoder = None
        self.reranker = None
        self.vectors = None
        self.dense_manifest = None
        self.cache_root = PROJECT / self.config['cache_root'] / self.corpus.dataset
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def prepare(self, method):
        """Build/load required indexes and models before normal query timing."""
        if method not in METHODS:
            raise ValueError('Unknown retrieval method')
        if method in ('bm25', 'hybrid', 'hybrid_reranker') and self.bm25 is None:
            started = time.perf_counter()
            cfg = self.config['bm25']
            if cfg['tokenizer'] != 'unicode-word-casefold-v1':
                raise ValueError('Unsupported BM25 tokenizer')
            self.bm25 = BM25Okapi([tokenize(r['search_text']) for r in self.corpus.units], k1=cfg['k1'], b=cfg['b'], epsilon=cfg['epsilon'])
            self.timings['bm25_index_build_s'] = time.perf_counter() - started
        if method != 'bm25' and self.vectors is None:
            self._prepare_dense()
        if method == 'hybrid_reranker' and self.reranker is None:
            started = time.perf_counter()
            from .neural import CrossReranker
            self.reranker = CrossReranker(resolve_model(self.config['reranker'], self.config.get('model_cache_root')),
                device=self.config['device'], max_length=self.config['reranker']['max_length'])
            self.timings['reranker_model_load_s'] = time.perf_counter() - started

    def _prepare_dense(self):
        """Load a pinned encoder and a content-addressed normalized vector cache."""
        started = time.perf_counter()
        import torch
        from .neural import Encoder
        self.timings['neural_import_s'] = time.perf_counter() - started
        torch.set_num_threads(self.config['torch_threads'])
        torch.manual_seed(self.config['seed'])
        started = time.perf_counter()
        settings = self.config['dense']
        self.encoder = Encoder(resolve_model(settings, self.config.get('model_cache_root')), device=self.config['device'], max_length=settings['max_length'])
        self.timings['dense_model_load_s'] = time.perf_counter() - started
        identity = {
            'cache_schema_version': 1, 'corpus_fingerprint': self.corpus.fingerprint,
            'encoding': settings, 'text_recipe': self.config['text_recipe'], 'pooling': 'attention-mask-mean-l2-v1', 'dtype': 'float32',
            'libraries': {x: importlib.metadata.version(x) for x in ['transformers', 'huggingface-hub', 'torch', 'numpy']},
        }
        cache_id = fingerprint(identity)
        folder = self.cache_root / cache_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path, vector_path = folder / 'manifest.json', folder / 'vectors.npy'
        ids = [r['chunk_id'] for r in self.corpus.units]
        started = time.perf_counter()
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text())
            if metadata['identity'] != identity or metadata['chunk_ids'] != ids:
                raise ValueError('Incompatible dense cache metadata or ID ordering')
            vectors = np.load(vector_path, allow_pickle=False)
            if fingerprint(vectors.tolist()) != metadata['vector_fingerprint']:
                raise ValueError('Dense cache checksum mismatch')
            self.timings['dense_cache_load_s'] = time.perf_counter() - started
            self.timings['dense_cache_hit'] = True
        else:
            texts = [settings['document_prefix'] + r['search_text'] for r in self.corpus.units]
            lengths = [len(x) for x in self.encoder.tokenizer(texts, truncation=False, padding=False, verbose=False)['input_ids']]
            vectors = self.encoder.encode(texts, batch_size=self.config['batch_size'])
            metadata = {
                'identity': identity, 'chunk_ids': ids, 'shape': list(vectors.shape),
                'max_input_tokens': max(lengths), 'truncated_document_count': sum(n > settings['max_length'] for n in lengths),
                'tokenization_policy': 'native tokenizer; truncate to configured model max_length; original chunks retained',
                'vector_fingerprint': fingerprint(vectors.tolist()),
                'deduplication_map': self.corpus.mapping(),
                'embedding_build_s': time.perf_counter() - started,
            }
            temp_vector = folder / 'vectors.tmp.npy'
            np.save(temp_vector, vectors, allow_pickle=False)
            temp_vector.replace(vector_path)
            temp_manifest = folder / 'manifest.tmp.json'
            temp_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            temp_manifest.replace(manifest_path)
            self.timings['dense_embedding_build_s'] = metadata['embedding_build_s']
            self.timings['dense_cache_hit'] = False
        if vectors.dtype != np.float32 or vectors.shape != (len(ids), self.encoder.get_sentence_embedding_dimension()):
            raise ValueError('Dense cache vector shape/dtype mismatch')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Dense vectors must be finite, nonzero, and unit normalized')
        self.vectors, self.dense_manifest = vectors, metadata

    def _query_vector(self, query):
        """Encode only the query; corpus vectors are already cached."""
        return self.encoder.encode([self.config['dense']['query_prefix'] + query])[0]

    def _rank(self, eligible, scores, limit):
        order = sorted(range(len(eligible)), key=lambda j: (-float(scores[j]), self.corpus.units[eligible[j]]['chunk_id']))
        return [(eligible[j], float(scores[j])) for j in order[:limit]]

    def _branches(self, query, eligible, want_bm25, want_dense):
        bm, dense = [], []
        cfg = self.config['hybrid']
        if want_bm25:
            scores = self.bm25.get_batch_scores(tokenize(query), eligible)
            bm = self._rank(eligible, scores, cfg['bm25_candidate_k'])
        if want_dense:
            scores = np.clip(self.vectors[eligible] @ self._query_vector(query), -1.0, 1.0)
            dense = self._rank(eligible, scores, cfg['dense_candidate_k'])
        return bm, dense

    def search(self, query, top_k=5, document_ids=None, method='bm25', mode='baseline'):
        """Retrieve within the actual document scope using one of four methods.

        mode preserves the existing feedback contract; only baseline is supported.
        method selects the algorithm independently of future feedback learning.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError('query must contain 1-4000 characters with non-whitespace text')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError('top_k must be an integer in [1,20]')
        if method not in METHODS or mode != 'baseline':
            raise ValueError('Unsupported method or feedback mode; this module implements baseline only')
        eligible = self.corpus.eligible(document_ids)
        self.prepare(method)
        started = time.perf_counter()
        bm, dense = self._branches(query.strip(), eligible, method != 'dense', method != 'bm25')
        stages = {}
        for idx, score in bm:
            stages.setdefault(idx, {})['bm25'] = score
        for idx, score in dense:
            stages.setdefault(idx, {})['dense_cosine'] = score
        pool = []
        if method == 'bm25':
            ranked, score_type = bm[:top_k], 'bm25'
        elif method == 'dense':
            ranked, score_type = dense[:top_k], 'dense_cosine'
        else:
            ids = [[self.corpus.units[i]['chunk_id'] for i, _ in r] for r in [bm, dense]]
            rrf = fuse_rankings(ids, self.config['hybrid']['rrf_constant'])
            positions = {self.corpus.units[i]['chunk_id']: i for i in stages}
            fused = sorted([(positions[c], score) for c, score in rrf.items()], key=lambda item: (-item[1], self.corpus.units[item[0]]['chunk_id']))
            for idx, score in fused:
                stages[idx]['rrf'] = score
            if method == 'hybrid':
                ranked, score_type = fused[:top_k], 'rrf'
            else:
                pool = fused[:self.config['reranker']['candidate_k']]
                texts = [self.corpus.units[i]['search_text'] for i, _ in pool]
                scores = self.reranker.predict([(query.strip(), text) for text in texts],
                    batch_size=self.config['reranker']['batch_size'])
                scores = np.asarray(scores).reshape(-1)
                if len(scores) != len(pool) or not np.isfinite(scores).all():
                    raise ValueError('Invalid cross-encoder scores')
                ranked = sorted([(item[0], float(score)) for item, score in zip(pool, scores)],
                    key=lambda item: (-item[1], self.corpus.units[item[0]]['chunk_id']))[:top_k]
                for idx, score in zip([i for i, _ in pool], scores):
                    stages[idx]['cross_encoder'] = float(score)
                score_type = 'cross_encoder_logit'
        results = [{**self.corpus.display(idx, document_ids), 'rank': rank, 'score': float(score),
                    'score_type': score_type, 'stage_scores': stages[idx]} for rank, (idx, score) in enumerate(ranked, 1)]
        candidate_docs = [self.corpus.display(idx, document_ids)['document_id'] for idx, _ in pool]
        return {'retrieval_id': 'ret_' + str(uuid.uuid4()), 'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'query': query.strip(), 'method': method, 'mode': mode, 'top_k': top_k,
                'corpus_fingerprint': self.corpus.fingerprint, 'eligible_unit_count': len(eligible),
                'results': results, 'candidate_document_ids': candidate_docs,
                'candidate_k': len(pool),
                'truncated_reranker_pairs': self.reranker.last_truncated_pair_count if method == 'hybrid_reranker' else 0,
                'query_ms': (time.perf_counter() - started) * 1000}
