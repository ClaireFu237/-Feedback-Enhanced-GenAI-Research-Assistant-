"""Regression: a low-RRF candidate must reach the cross-encoder and can win."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from services.retrieval.engine import LocalRetriever


class FullUnionTests(unittest.TestCase):
    def engine(self):
        engine = LocalRetriever.__new__(LocalRetriever)
        units = [{'chunk_id': f'c{i:02}', 'search_text': f'evidence {i}'} for i in range(35)]
        engine.corpus = SimpleNamespace(units=units, fingerprint='test', eligible=lambda ids: list(range(35)),
            display=lambda i, ids: {'chunk_id': units[i]['chunk_id'], 'document_id': f'd{i // 5}', 'text': units[i]['search_text']})
        engine.config = {'hybrid': {'rrf_constant': 60}, 'reranker': {'batch_size': 8, 'candidate_k': 20}}
        engine.prepare = Mock()
        engine._branches = Mock(return_value=([(i, 35-i) for i in range(30)], [(i, 35-i) for i in range(5,35)]))
        engine.reranker = SimpleNamespace(last_truncated_pair_count=0,
            predict=Mock(side_effect=lambda pairs, batch_size: np.array([100 if text=='evidence 34' else 0 for _,text in pairs])))
        return engine

    def test_entire_union_is_reranked_before_public_top_k(self):
        engine = self.engine()
        result = engine.search('query',top_k=5,method='hybrid_reranker')
        self.assertEqual(result['candidate_k'],35)
        self.assertEqual(len(engine.reranker.predict.call_args.args[0]),35)
        self.assertEqual(result['results'][0]['chunk_id'],'c34')
        self.assertEqual(len(result['results']),5)
        self.assertGreaterEqual(result['reranker_ms'],0)

    def test_nonfinite_scores_are_rejected(self):
        engine = self.engine()
        engine.reranker.predict.side_effect=lambda pairs,batch_size: np.full(len(pairs),np.nan)
        with self.assertRaisesRegex(ValueError,'Invalid cross-encoder'):
            engine.search('query',method='hybrid_reranker')


if __name__=='__main__':
    unittest.main()
