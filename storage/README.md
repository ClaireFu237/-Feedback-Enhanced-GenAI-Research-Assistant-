# Persistence Adapters

Isolate AWS and local persistence from application logic.

Expose document, chunk, vector, query-log, answer, and feedback access methods. S3 stores files and artifacts; DynamoDB stores structured records and feedback; local persistence supports development and small-corpus vector snapshots. Keep serialization and key conventions consistent across adapters.

Status: architecture scaffold; implementation will be added here.
