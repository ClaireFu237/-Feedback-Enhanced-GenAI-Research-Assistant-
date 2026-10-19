# Amazon DynamoDB Storage

Persist document and chunk metadata, small-corpus embeddings, query and answer records, feedback events, and aggregate scores.

Document table keys and query access patterns before implementing adapters. Store vectors with model/version metadata and account for item-size limits; use S3 or local snapshots when appropriate. Use conditional or atomic writes for duplicate-event handling and feedback aggregates. Similarity computation remains in the retrieval service.

Status: architecture scaffold; implementation will be added here.
