# Grounded generation through Amazon Bedrock

This is the first working retrieval-to-LLM integration. It uses the existing local
hybrid retriever and one Bedrock Converse call per answer. It does not migrate
embeddings or deploy AWS infrastructure.

## Explicit retrieval contract

Generation requires a `RetrievalProfile`; there is no implicit profile fallback.
`config/retrieval_profiles/hybrid_v1.json` pins library retrieval, BM25/dense top50,
RRF 60, output top10, title+body indexing, no document cap, BM25 parameters and
the dense model/revision. The profile overrides the base engine's algorithm
settings. Base configuration still supplies dataset locations, cache paths and
CPU settings. Every run records both the profile fingerprint and effective config.

The service owns one `LocalRetriever` and one Bedrock client. It prepares its
indexes/model once and reuses them for subsequent questions. Local inference is
serialized by a lock. The corpus is pinned to the library snapshot loaded when
the service starts; restart the generation service after publishing new papers.
SearchOptions, the retrieval CLI and LocalRetriever.search defaults remain BM25.

## Run a real answer

Install the updated requirements into the project venv. Configure AWS credentials
using the standard SDK credential chain and allow `bedrock:InvokeModel` for the
selected model/inference profile and any destination models required by its
cross-region inference profile. No keys belong in repository configuration.

```bash
.venv/bin/python -m services.generation \
  --profile config/retrieval_profiles/hybrid_v1.json \
  --query "Which temperature settings were tested in the truss design experiments?"
```

`config/generation.json` selects region, model/inference-profile ID and generation
parameters. The initial choice is `us.amazon.nova-lite-v1:0` in `us-east-1`.
Model access and availability depend on the account. There is no automatic model
fallback or extra LLM repair call. Network, access and schema errors fail the run
and preserve any completed artifacts. SDK retry attempts are limited to one.

## Structured answer

The English system prompt and context template are versioned under `prompts/`.
Retrieved sources are serialized as JSON, with source IDs assigned by code.
`toolConfig.toolChoice.tool` forces `submit_grounded_answer`. The tool's input
is the final answer; no external tool is executed and no second model round is
needed. The runtime schema enumerates only source IDs actually sent to the model.

```json
{
  "status": "answered",
  "claims": [{"text": "A supported statement.", "source_ids": ["S1"]}],
  "missing_information": []
}
```

Supported statuses are `answered`, `partial`, `insufficient_evidence`. Every claim
requires citations. Partial/insufficient answers must explain missing information;
insufficient answers contain no claims. Code validates schema, complete tool-use
termination and source IDs, then resolves citations to original document/chunk
IDs, titles and locations. These checks do **not** verify semantic entailment.

Context includes whole original chunks, never silently cut bodies. The configured
character budget is a conservative input guard, not a model tokenizer count.
Omitted chunks are recorded explicitly. The generation model's output limit is
independent of the embedding encoder's 256-token limit.

## HTTP endpoint

Start the existing FastAPI app, then submit:

```http
POST /api/answers
Content-Type: application/json

{"query":"Which temperature settings were tested?","retrieval_profile":"hybrid_v1"}
```

Optional `document_ids` limits retrieval to explicitly selected papers. This is
user scope. The endpoint returns HTTP 202 with a task ID;
poll `GET /api/tasks/{task_id}`. The task result contains the answer record.
Generation uses the existing bounded serial task queue and a single lazily created
service per application process; health endpoints remain responsive.

## Artifacts

CLI artifacts are stored in `outputs/generation/ans_<id>/`; API artifacts in
`data_local/web_runs/answers/ans_<id>/` (or the custom workbench root).

- `record.json`: IDs, state, exact configs/profile, corpus fingerprint, prompt and
  schema versions, structured answer, usage and timings.
- `retrieval.json`: the full ranked top10 returned by retrieval.
- `context.json`: actual text/sources supplied and omitted chunks.
- `request.json`: exact Converse payload, including English prompts/tool schema.
- `response.json`: model response, stop reason, usage and AWS request ID;
  HTTP headers are removed. Never stores AWS credentials.

A failed call is recorded as failed and never presented as a successful answer.
The answer's source citations can be connected to the proposal's later feedback
workflow through answer_id, query_id, retrieval_id and stable chunk IDs.

AWS reference: https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html
