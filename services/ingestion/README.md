# Document Ingestion

Extract text from uploaded PDF or text documents and split it into searchable chunks.

Planned steps: validate document type, extract text with page provenance, normalize text, create overlapping chunks, request embeddings, and persist artifacts and metadata. Track processing status and errors so repeated events do not create duplicate chunks. Preserve source locations for downstream citations.

Status: architecture scaffold; implementation will be added here.
