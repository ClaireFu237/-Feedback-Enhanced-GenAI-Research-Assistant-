# Semantic Retrieval and Reranking

Retrieve relevant chunks using cosine similarity and optionally rerank them using historical feedback.

Maintain a semantic-only baseline and a feedback-enhanced mode. Start with `final_score = semantic_similarity + feedback_weight * historical_feedback_score`. Define score normalization and aggregation explicitly. For the small initial corpus, load vectors through storage and compute similarity in Python; DynamoDB is persistence, not a native vector-search engine. Return scores and chunk IDs for traceability.

Status: architecture scaffold; implementation will be added here.
