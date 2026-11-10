"""Versioned retrieval settings; no global retrieval defaults are modified."""
from dataclasses import dataclass
import copy
import json
from pathlib import Path

from services.retrieval.data import fingerprint


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    dataset: str
    method: str
    bm25_k: int
    dense_k: int
    rrf_k: int
    top_k: int
    text_recipe: str
    max_chunks_per_document: int | None
    bm25: dict
    dense: dict
    deduplication: str

    @classmethod
    def load(cls, path):
        obj = cls(**json.loads(Path(path).read_text(encoding='utf-8')))
        # This named baseline is intentionally fixed. New behavior requires a new version.
        expected = (obj.name, obj.dataset, obj.method, obj.bm25_k, obj.dense_k, obj.rrf_k,
                    obj.top_k, obj.text_recipe, obj.max_chunks_per_document, obj.deduplication)
        if expected != ('hybrid_v1', 'library', 'hybrid', 50, 50, 60, 10,
                        'title-double-newline-body-v1', None, 'exact-title-and-text-v1'):
            raise ValueError('Unsupported retrieval profile; hybrid_v1 settings must remain fixed')
        if obj.dense['normalize_embeddings'] is not True:
            raise ValueError('Profile requires normalized dense embeddings')
        return obj

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @property
    def fingerprint(self):
        return fingerprint(self.to_dict())

    def engine_config(self, base):
        cfg = copy.deepcopy(base)
        cfg.update(text_recipe=self.text_recipe, deduplication=self.deduplication,
                   bm25=copy.deepcopy(self.bm25), dense=copy.deepcopy(self.dense))
        cfg['hybrid'] = {'bm25_candidate_k': self.bm25_k, 'dense_candidate_k': self.dense_k, 'rrf_constant': self.rrf_k}
        return cfg
