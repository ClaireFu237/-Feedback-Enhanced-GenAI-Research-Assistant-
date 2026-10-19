# Application Services

Coordinate the ingestion, retrieval, generation, and feedback workflows.

Subdirectories separate the major RAG capabilities. Services consume schemas and storage interfaces; routes and the frontend should not implement these workflows. AWS SDK persistence calls belong in storage adapters. Bedrock inference integration belongs in embedding and generation services.

Status: architecture scaffold; implementation will be added here.
