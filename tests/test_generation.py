"""Offline generation contract checks without cloud calls."""
import copy
import json
import tempfile
import unittest
import time
from pathlib import Path
from unittest.mock import Mock

from jsonschema import ValidationError
from services.generation.client import build_request, parse_response
from services.generation.context import build_context
from services.generation.profile import RetrievalProfile
from services.generation.service import GenerationService
from services.retrieval.data import PROJECT, load_config


def retrieval():
    return {'retrieval_id':'ret_test','corpus_fingerprint':'corpus','query_ms':1.2,'results':[
        {'document_id':'d1','chunk_id':'c1','document_title':'Paper','text':'The measured value is 42.',
         'source_location':{'page_start':3,'page_end':3}}]}


def response():
    return {'stopReason':'tool_use','output':{'message':{'role':'assistant','content':[{'toolUse':{
        'name':'submit_grounded_answer','toolUseId':'t1','input':{'status':'answered',
            'claims':[{'text':'The value is 42.','source_ids':['S1']}], 'missing_information':[]}}}]}},
        'usage':{'inputTokens':20,'outputTokens':10}}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.profile=RetrievalProfile.load(PROJECT/'config/retrieval_profiles/hybrid_v1.json')
        self.config=json.loads((PROJECT/'config/generation.json').read_text())
        self.context=build_context('What value?',retrieval(),40000)

    def test_forced_tool_and_only_supplied_source_ids(self):
        request=build_request(self.config,'English system prompt',self.context)
        self.assertEqual(request['toolConfig']['toolChoice'],{'tool':{'name':'submit_grounded_answer'}})
        schema=request['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
        self.assertEqual(schema['properties']['claims']['items']['properties']['source_ids']['items']['enum'],['S1'])
        answer=parse_response(response(),self.context['sources'])
        self.assertEqual(answer['citations'][0]['chunk_id'],'c1')
        self.assertNotIn('text',answer['citations'][0])

    def test_unknown_citation_and_plaintext_and_truncation_are_rejected(self):
        bad=response();bad['output']['message']['content'][0]['toolUse']['input']['claims'][0]['source_ids']=['S999']
        with self.assertRaisesRegex(ValueError,'not sent'):parse_response(bad,self.context['sources'])
        for reason in ('end_turn','max_tokens'):
            bad=response();bad['stopReason']=reason
            with self.assertRaises(ValueError):parse_response(bad,self.context['sources'])

    def test_partial_and_insufficient_answer_consistency(self):
        bad=response(); value=bad['output']['message']['content'][0]['toolUse']['input'];value['status']='partial'
        with self.assertRaises(ValueError):parse_response(bad,self.context['sources'])
        value.update(status='insufficient_evidence',claims=[],missing_information=['No measurement was provided.'])
        self.assertEqual(parse_response(bad,self.context['sources'])['status'],'insufficient_evidence')
        value['claims']=[{'text':'Uncited','source_ids':[]}]
        with self.assertRaises(ValidationError):parse_response(bad,self.context['sources'])

    def test_context_budget_does_not_silently_cut_chunks(self):
        r=retrieval();r['results'][0]['text']='x'*50000
        c=build_context('Question',r,40000)
        self.assertEqual(c['sources'],[])
        self.assertEqual(c['omitted'][0]['chunk_id'],'c1')

    def test_profile_overrides_base_without_mutating_it(self):
        base=load_config(); original=copy.deepcopy(base); base['hybrid']['bm25_candidate_k']=200
        cfg=self.profile.engine_config(base)
        self.assertEqual(cfg['hybrid']['bm25_candidate_k'],50)
        self.assertEqual(base['hybrid']['bm25_candidate_k'],200)
        self.assertEqual(original['text_recipe'],cfg['text_recipe'])

    def test_service_reuses_one_retriever_and_persists_real_request(self):
        fake=Mock();fake.search.return_value=retrieval(); factory=Mock(return_value=fake)
        client=Mock();client.converse.return_value=response()
        with tempfile.TemporaryDirectory() as tmp:
            service=GenerationService(profile=self.profile,config=self.config,retrieval_config=load_config(),
                output_root=tmp,client=client,retriever_factory=factory)
            first=service.answer('Question one');second=service.answer('Question two')
            factory.assert_called_once();fake.prepare.assert_called_once_with('hybrid')
            self.assertEqual(fake.search.call_args.kwargs,{'method':'hybrid','top_k':10,'document_ids':None})
            self.assertEqual(client.converse.call_count,2)
            self.assertNotEqual(first['answer_id'],second['answer_id'])
            folder=Path(tmp)/first['answer_id']
            for name in ('retrieval','context','request','response','record'):
                self.assertTrue((folder/f'{name}.json').is_file())
            self.assertEqual(first['state'],'complete')
            self.assertEqual(first['retrieval_profile']['name'],'hybrid_v1')

    def test_failed_cloud_call_keeps_context_and_no_false_success(self):
        fake=Mock();fake.search.return_value=retrieval(); client=Mock();client.converse.side_effect=RuntimeError('Cloud rejected request')
        with tempfile.TemporaryDirectory() as tmp:
            service=GenerationService(profile=self.profile,config=self.config,retrieval_config=load_config(),
                output_root=tmp,client=client,retriever_factory=Mock(return_value=fake))
            with self.assertRaises(RuntimeError):service.answer('Question')
            folder=next(Path(tmp).iterdir());record=json.loads((folder/'record.json').read_text())
            self.assertEqual(record['state'],'failed');self.assertTrue((folder/'request.json').is_file())

    def test_api_requires_profile_and_reuses_service_without_cloud(self):
        from fastapi.testclient import TestClient
        from main import create_app, SearchOptions
        self.assertEqual(SearchOptions(query='Question').method, 'bm25')
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(tmp)) as client:
            self.assertEqual(client.post('/api/answers', json={'query':'Question'}).status_code,422)
            self.assertEqual(client.post('/api/answers', json={'query':'Question','retrieval_profile':'unknown'}).status_code,422)
            fake=Mock();fake.answer.return_value={'answer_id':'ans_test','state':'complete'}
            client.app.state.generation_service=fake
            for question in ('Question one','Question two'):
                response=client.post('/api/answers',json={'query':question,'retrieval_profile':'hybrid_v1'})
                self.assertEqual(response.status_code,202)
                task_id=response.json()['task_id']
                for _ in range(100):
                    task=client.get(f'/api/tasks/{task_id}').json()
                    if task['status'] in ('completed','failed'):break
                    time.sleep(0.01)
                self.assertEqual(task['status'],'completed')
                self.assertEqual(task['result']['answer_id'],'ans_test')
            self.assertEqual(fake.answer.call_count,2)


if __name__=='__main__':unittest.main()
