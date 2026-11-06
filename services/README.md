# Application Services

Coordinate the ingestion, retrieval, generation, and feedback workflows.

Subdirectories separate the major RAG capabilities. Services consume schemas and storage interfaces; routes and the frontend should not implement these workflows. AWS SDK persistence calls belong in storage adapters. Bedrock inference integration belongs in embedding and generation services.

Implemented locally: ingestion (PDF to chunks) and retrieval (BM25, dense, hybrid and cross-encoder ranking). AWS embedding/generation and feedback remain scaffolds. See the root README.

`workbench.py` coordinates local web runs, isolated artifacts and the bounded job queue. The FastAPI entry is `main.py`.
