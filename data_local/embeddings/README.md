# Embedding Cache

Hold generated local vector files or manifests for cached embeddings.

Record embedding model ID, dimensions, source chunk IDs, and content hashes. Reuse cached vectors for unchanged content to reduce Bedrock calls. Keep generated vectors out of source control by default.

Status: architecture scaffold; implementation will be added here.
