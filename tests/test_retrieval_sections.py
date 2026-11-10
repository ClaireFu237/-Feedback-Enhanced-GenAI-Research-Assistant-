"""Check section indexing without changing evidence text or stable identities."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from services.retrieval.data import Corpus


class SectionTests(unittest.TestCase):
    def test_section_recipe_preserves_body_and_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{"stage": "test"}')
            documents = [{'document_id': 'd', 'title': 'Paper'}]
            chunks, mapping = [], []
            for i, section in enumerate(['Methods', 'Unsectioned', '0.2 V']):
                body = f'Original evidence {i}'
                chunks.append({'chunk_id': str(i), 'document_id': 'd', 'text': body,
                    'content_hash': hashlib.sha256(body.encode()).hexdigest(),
                    'index_version': 'v1', 'source_location': {'page_start': 1}})
                mapping.append({'chunk_id': str(i), 'document_id': 'd', 'source_document_id': 'source',
                    'source_chunk_id': str(i), 'section': section})
            for name, rows in [('documents', documents), ('chunks', chunks), ('id_map', mapping)]:
                (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            base = Corpus.load('test', root)
            section = Corpus.load('test', root, 'title-section-body-v1')
            self.assertEqual([u['chunk_id'] for u in base.units], [u['chunk_id'] for u in section.units])
            self.assertEqual([u['text'] for u in base.units], [u['text'] for u in section.units])
            self.assertEqual(section.units[0]['search_text'], 'Paper\n\nMethods\n\nOriginal evidence 0')
            self.assertEqual(section.units[1]['search_text'], base.units[1]['search_text'])
            self.assertEqual(section.units[2]['search_text'], base.units[2]['search_text'])
            self.assertNotEqual(base.fingerprint, section.fingerprint)



if __name__ == '__main__':
    unittest.main()
