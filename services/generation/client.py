"""One Converse request with a forced answer-submission tool, validated locally."""
import copy
from jsonschema import Draft202012Validator

TOOL_NAME = 'submit_grounded_answer'
ANSWER_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'type': 'string', 'enum': ['answered', 'partial', 'insufficient_evidence']},
        'claims': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'text': {'type': 'string', 'minLength': 1},
                           'source_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'uniqueItems': True}},
            'required': ['text', 'source_ids']}},
        'missing_information': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}
    }, 'required': ['status', 'claims', 'missing_information']
}


def make_client(config):
    import boto3
    from botocore.config import Config
    return boto3.Session().client('bedrock-runtime', region_name=config['region'], config=Config(
        connect_timeout=config['connect_timeout_s'], read_timeout=config['read_timeout_s'],
        retries={'total_max_attempts': 1}))


def build_request(config, system, context):
    schema = copy.deepcopy(ANSWER_SCHEMA)
    source_ids = [s['source_id'] for s in context['sources']]
    if not source_ids:
        raise ValueError('Cannot invoke generation without sources')
    schema['properties']['claims']['items']['properties']['source_ids']['items']['enum'] = source_ids
    return {'modelId': config['model_id'], 'system': [{'text': system}],
            'messages': [{'role': 'user', 'content': [{'text': context['text']}]}],
            'inferenceConfig': {'maxTokens': config['max_output_tokens'], 'temperature': config['temperature']},
            'toolConfig': {'tools': [{'toolSpec': {'name': TOOL_NAME,
                'description': 'Submit the final evidence-grounded answer with claim-level source citations and explicit missing information.',
                'inputSchema': {'json': schema}}}], 'toolChoice': {'tool': {'name': TOOL_NAME}}}}


def parse_response(response, sources):
    if response.get('stopReason') != 'tool_use':
        raise ValueError(f"Expected a complete answer tool call; stopReason={response.get('stopReason')}")
    blocks = response.get('output', {}).get('message', {}).get('content', [])
    tools = [b['toolUse'] for b in blocks if 'toolUse' in b]
    if len(tools) != 1 or tools[0].get('name') != TOOL_NAME:
        raise ValueError('Expected exactly one submit_grounded_answer tool call')
    answer = tools[0].get('input')
    Draft202012Validator(ANSWER_SCHEMA).validate(answer)
    by_id = {s['source_id']: s for s in sources}
    for claim in answer['claims']:
        if any(s not in by_id for s in claim['source_ids']):
            raise ValueError('Model cited a source that was not sent in this request')
        if not claim['text'].strip():
            raise ValueError('Empty answer claim')
    if answer['status'] == 'insufficient_evidence':
        if answer['claims'] or not answer['missing_information']:
            raise ValueError('Insufficient evidence requires no claims and an explanation')
    elif not answer['claims']:
        raise ValueError('Answered/partial results require cited claims')
    if answer['status'] == 'answered' and answer['missing_information']:
        raise ValueError('Answered results cannot have missing information')
    if answer['status'] == 'partial' and not answer['missing_information']:
        raise ValueError('Partial results must identify missing information')
    cited = {s for c in answer['claims'] for s in c['source_ids']}
    return {**answer, 'citations': [{k: v for k, v in source.items() if k != 'text'}
                                    for source in sources if source['source_id'] in cited]}
