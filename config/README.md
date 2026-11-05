# Application Configuration

Centralize environment-specific settings for AWS region, S3 buckets, DynamoDB tables, Bedrock model IDs, chunk size, overlap, retrieval top-k, and feedback weight.

Planned files: `settings.py` and an environment-variable example. Validate settings at startup; keep credentials outside the repository. Supply configuration to services and storage adapters rather than reading environment variables throughout the codebase.

Status: architecture scaffold; implementation will be added here.
