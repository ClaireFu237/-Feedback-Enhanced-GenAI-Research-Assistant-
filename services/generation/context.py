"""Bind short source IDs to immutable retrieval provenance."""
import json
from pathlib import Path

from services.retrieval.data import PROJECT


def build_context(question, retrieval, max_chars, template=None):
    template = template or (PROJECT / 'prompts/context_v1.txt').read_text(encoding='utf-8')
    sources, omitted = [], []
    for item in retrieval['results']:
        source = {'source_id': f'S{len(sources)+1}', 'document_id': item['document_id'],
                  'chunk_id': item['chunk_id'], 'title': item['document_title'],
                  'pages': item['source_location'], 'text': item['text']}
        trial = template.format(question=question, sources_json=json.dumps(sources + [source], ensure_ascii=False))
        if len(trial) > max_chars:
            omitted.append({'chunk_id': item['chunk_id'], 'reason': 'context_character_budget'})
            continue
        sources.append(source)
    text = template.format(question=question, sources_json=json.dumps(sources, ensure_ascii=False))
    if len(text) > max_chars:
        raise ValueError('Question and context template exceed the configured character budget')
    return {'version': 'context_v1', 'text': text, 'sources': sources, 'omitted': omitted,
            'budget_policy': 'Whole chunks only; character guard, not a model token count',
            'characters': len(text)}
