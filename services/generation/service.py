"""Reusable retrieval-to-Bedrock orchestration with a pinned corpus and audit trail."""
import copy
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from services.retrieval.data import PROJECT, load_config, fingerprint
from services.retrieval.engine import LocalRetriever
from .client import build_request, make_client, parse_response
from .context import build_context


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class GenerationService:
    def __init__(self, *, profile, config, retrieval_config, output_root=None,
                 client=None, retriever_factory=LocalRetriever):
        # Explicit profile is mandatory. Capture it; do not let later config mutation change a run.
        self.profile = copy.deepcopy(profile)
        self.profile_data = self.profile.to_dict()
        self.config = copy.deepcopy(config)
        engine_config = self.profile.engine_config(retrieval_config)
        self.retriever = retriever_factory(dataset=self.profile.dataset, config=engine_config)
        self.engine_config = copy.deepcopy(engine_config)
        self.client = client if client is not None else make_client(self.config)
        self.output_root = Path(output_root or PROJECT / self.config['output_root'])
        self.system = (PROJECT / 'prompts/grounded_answer_v1.txt').read_text(encoding='utf-8')
        self.template = (PROJECT / 'prompts/context_v1.txt').read_text(encoding='utf-8')
        self.lock = threading.Lock()
        self.prepared = False

    def answer(self, question, document_ids=None):
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError('Question must contain 1–4000 characters')
        question = question.strip()
        # One retriever/model per service; serialize local inference to avoid lazy-init races.
        with self.lock:
            return self._answer(question, document_ids)

    def _answer(self, question, document_ids):
        started = time.perf_counter()
        answer_id, query_id = 'ans_' + uuid.uuid4().hex, 'qry_' + uuid.uuid4().hex
        folder = self.output_root / answer_id
        folder.mkdir(parents=True)
        record = {'answer_id': answer_id, 'query_id': query_id, 'question': question,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(), 'state': 'running',
                  'retrieval_profile': self.profile_data, 'retrieval_profile_fingerprint': fingerprint(self.profile_data),
                  'generation_config': self.config, 'effective_retrieval_config': self.engine_config,
                  'prompt_version': 'grounded_answer_v1', 'context_version': 'context_v1',
                  'output_schema_version': 'grounded_answer_v1'}
        write_json(folder / 'record.json', record)
        try:
            preparation_s = 0.0
            if not self.prepared:
                prep = time.perf_counter()
                self.retriever.prepare(self.profile.method)
                preparation_s = time.perf_counter() - prep
                self.prepared = True
            retrieval = self.retriever.search(question, method=self.profile.method,
                top_k=self.profile.top_k, document_ids=document_ids)
            write_json(folder / 'retrieval.json', retrieval)
            record.update(retrieval_id=retrieval['retrieval_id'], corpus_fingerprint=retrieval['corpus_fingerprint'])
            context = build_context(question, retrieval, self.config['context_max_chars'], self.template)
            write_json(folder / 'context.json', context)
            if not context['sources']:
                raise ValueError('No source chunks fit the context budget; LLM was not called')
            request = build_request(self.config, self.system, context)
            write_json(folder / 'request.json', request)
            llm_started = time.perf_counter()
            response = self.client.converse(**request)
            llm_ms = (time.perf_counter() - llm_started) * 1000
            # Strip HTTP header metadata; preserve model output, usage, stop reason and request ID.
            response = {k: v for k, v in response.items() if k != 'ResponseMetadata'} | {
                'request_id': response.get('ResponseMetadata', {}).get('RequestId')}
            write_json(folder / 'response.json', response)
            answer = parse_response(response, context['sources'])
            record.update(state='complete', answer=answer, usage=response.get('usage', {}),
                model_id=self.config['model_id'], stop_reason=response.get('stopReason'),
                timings={'preparation_s': preparation_s, 'retrieval_ms': retrieval['query_ms'],
                         'llm_ms': llm_ms, 'total_ms': (time.perf_counter()-started)*1000},
                context_source_count=len(context['sources']), omitted_chunks=context['omitted'])
            write_json(folder / 'record.json', record)
            return record
        except Exception as exc:
            # Keep completed retrieval/request artifacts even if AWS rejects the request.
            record.update(state='failed', error={'type': type(exc).__name__, 'message': str(exc)},
                          total_ms=(time.perf_counter()-started)*1000)
            write_json(folder / 'record.json', record)
            raise


def create_service(*, profile_path, config_path=None, output_root=None):
    from .profile import RetrievalProfile
    profile = RetrievalProfile.load(profile_path)
    config = json.loads(Path(config_path or PROJECT/'config/generation.json').read_text(encoding='utf-8'))
    return GenerationService(profile=profile, config=config, retrieval_config=load_config(), output_root=output_root)
