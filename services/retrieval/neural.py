"""Inference-only adapters for locally cached Transformers checkpoints.

The encoder uses the model card's attention-mask mean pooling and L2
normalization. The reranker returns raw sequence-classification logits.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class Encoder:
    """English MiniLM sentence encoder with no query/document prefix."""
    def __init__(self, path, device, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device = device
        self.max_seq_length = max_length

    def get_sentence_embedding_dimension(self):
        return self.model.config.hidden_size

    def encode(self, texts, batch_size=16):
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                    max_length=self.max_seq_length, return_tensors='pt').to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


class CrossReranker:
    """Score only supplied query/passage pairs with a pretrained cross-encoder."""
    def __init__(self, path, device, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.last_truncated_pair_count = 0

    def predict(self, pairs, batch_size):
        scores = []
        self.last_truncated_pair_count = 0
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start+batch_size]
                queries, texts = zip(*batch)
                lengths = self.tokenizer(list(queries), list(texts), truncation=False, padding=False, verbose=False)['input_ids']
                self.last_truncated_pair_count += sum(len(tokens)>self.max_length for tokens in lengths)
                encoded = self.tokenizer(list(queries), list(texts), padding=True,
                    truncation='longest_first', max_length=self.max_length, return_tensors='pt').to(self.device)
                logits = self.model(**encoded).logits.reshape(-1)
                scores.extend(logits.cpu().tolist())
        return np.asarray(scores,dtype=np.float32)
