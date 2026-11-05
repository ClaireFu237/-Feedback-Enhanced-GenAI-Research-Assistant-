"""Cut exact normalized-text slices using the actual embedding tokenizer."""
from __future__ import annotations

import bisect
import hashlib
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
VERSION = "paragraph-sentence-token-v1"


class TokenBudget:
    def __init__(self, tokenizer_file=None, download=False, chunk_size=200, overlap=30, max_input=256):
        if not (0 <= overlap < chunk_size and max_input >= 4):
            raise ValueError("Require 0 <= overlap < chunk-size and max-input-tokens >= 4")
        if tokenizer_file is None:
            try:
                tokenizer_file = hf_hub_download(MODEL, "tokenizer.json", revision=REVISION,
                                                local_files_only=not download)
            except Exception as exc:
                raise ValueError("Tokenizer not available. Use --download-tokenizer once, or "
                                 "--tokenizer-file /path/to/tokenizer.json") from exc
        try:
            self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        except Exception as exc:
            raise ValueError(f"Cannot load tokenizer.json: {exc}") from exc
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.fingerprint = hashlib.sha256(Path(tokenizer_file).read_bytes()).hexdigest()
        self.chunk_size, self.overlap, self.max_input = chunk_size, overlap, max_input

    def encode(self, text, special=False):
        return self.tokenizer.encode(text, add_special_tokens=special)

    def count(self, text, special=False):
        return len(self.encode(text, special).ids)

    def fits(self, title, text):
        return (self.count(text) <= self.chunk_size and
                self.count(title + "\n\n" + text, special=True) <= self.max_input)

    def ranges(self, title, text):
        """Yield advancing windows covering all non-whitespace characters.

        Prefer paragraph/sentence endings near the budget. Overlap is at most
        the configured token count; splitting never decodes/reconstructs text.
        """
        reserve = self.count(title + "\n\n", special=True)
        capacity = min(self.chunk_size, self.max_input - reserve)
        if capacity < 1:
            raise ValueError("Title leaves no token budget for PDF text")
        encoding = self.encode(text)
        offsets = encoding.offsets
        if not offsets:
            # Some standalone diacritics are removed by the BERT normalizer.
            # Keep the exact source slice rather than silently omitting it.
            if text.strip():
                if not self.fits(title, text):
                    raise ValueError("Text cannot fit the configured token budget")
                yield 0, len(text)
            return
        starts = [a for a, _ in offsets]
        boundaries = sorted(set(m.end() for m in re.finditer(r"\n\n|[.!?][\"')\]]*\s+", text)))
        start = 0
        while start < len(text):
            while start < len(text) and text[start].isspace():
                start += 1
            if start == len(text):
                break
            first = bisect.bisect_left(starts, start)
            stop = min(len(offsets), first + capacity)
            end = starts[stop] if stop < len(offsets) else len(text)
            # Local retokenization can differ from tokenization of the full section.
            while end > start and not self.fits(title, text[start:end].rstrip()):
                stop -= 1
                end = starts[stop] if stop > first else start
            if end <= start:
                # With a very small budget, a continuation WordPiece can split
                # into several tokens when used at the beginning of a window.
                # Find a fitting character prefix without dropping that word.
                probe = offsets[first][1] if first < len(offsets) else len(text)
                for candidate in range(probe, start, -1):
                    if self.fits(title, text[start:candidate]):
                        end = candidate
                        break
                if end <= start:
                    raise ValueError("Cannot fit a text character with the title; increase the token budget")
            if end < len(text):
                lower = start + int((end - start) * .6)
                candidates = boundaries[bisect.bisect_right(boundaries, lower):
                                        bisect.bisect_right(boundaries, end)]
                if candidates:
                    end = candidates[-1]
            end = len(text[:end].rstrip())
            if end <= start or not self.fits(title, text[start:end]):
                raise ValueError("Chunk budget invariant failed")
            yield start, end
            if not text[end:].strip():
                break
            if self.overlap:
                local_offsets = self.encode(text[start:end]).offsets
                # Never retain the whole previous window, even for tiny capacities.
                keep = min(self.overlap, max(0, len(local_offsets) - 1))
                next_start = start + local_offsets[-keep][0] if keep else end
                sentence_starts = boundaries[bisect.bisect_left(boundaries, next_start):
                                             bisect.bisect_left(boundaries, end)]
                if sentence_starts:
                    next_start = sentence_starts[0]
                # A suffix beginning inside a WordPiece word can tokenize
                # differently from the original chunk. Enforce the overlap
                # bound on the actual suffix as well.
                while next_start < end and self.count(text[next_start:end]) > self.overlap:
                    suffix = self.encode(text[next_start:end])
                    next_start += max(1, suffix.offsets[0][1])
                start = next_start if next_start > start else end
            else:
                start = end


def make_chunks(text, paragraphs, title, budget):
    """Keep section boundaries and omit references without bridging across them."""
    groups = []
    current = []
    for paragraph in paragraphs:
        if not paragraph["searchable"] or (current and paragraph["section_id"] != current[-1]["section_id"]):
            if current:
                groups.append(current)
                current = []
        if paragraph["searchable"]:
            current.append(paragraph)
    if current:
        groups.append(current)
    result = []
    for group in groups:
        base, limit = group[0]["char_start"], group[-1]["char_end"]
        previous_end = base
        for a, b in budget.ranges(title, text[base:limit]):
            start, end = base + a, base + b
            if text[previous_end:start].strip():
                raise ValueError("Chunking omitted non-whitespace text")
            matching = [p for p in group if p["char_start"] < end and p["char_end"] > start]
            result.append({"text": text[start:end], "source_location": {
                "page_start": min(p["page_number"] for p in matching),
                "page_end": max(p["page_number"] for p in matching),
                "char_start": start, "char_end": end}, "section": group[0]["section"],
                "block_ids": list(dict.fromkeys(p["block_id"] for p in matching)),
                "token_count": budget.count(text[start:end]),
                "input_token_count": budget.count(title + "\n\n" + text[start:end], special=True)})
            previous_end = max(previous_end, end)
        if text[previous_end:limit].strip():
            raise ValueError("Chunking omitted the end of a section")
    if not result:
        raise ValueError("No searchable chunks remain after normalization")
    return result
