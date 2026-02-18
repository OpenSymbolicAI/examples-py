"""Text utilities for semantic-aware truncation."""

from __future__ import annotations

import re

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Paragraph break, then sentence-ending punctuation
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def truncate(text: str, max_tokens: int = 2000) -> str:
    """Truncate text to a token limit respecting paragraph and sentence boundaries.

    Strategy:
    1. Split into paragraphs (double newline).
    2. Accumulate whole paragraphs while under the token limit.
    3. If a single paragraph exceeds the remaining budget, split it by
       sentences and accumulate those instead.
    4. Never cuts mid-sentence.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        Truncated text that ends at a clean boundary.
    """
    if not text:
        return text

    if count_tokens(text) <= max_tokens:
        return text

    paragraphs = _PARAGRAPH_SPLIT.split(text)
    result: list[str] = []
    used = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = count_tokens(para)

        if used + para_tokens <= max_tokens:
            result.append(para)
            used += para_tokens
            continue

        # Paragraph too large — split by sentences, keep as one paragraph
        remaining = max_tokens - used
        if remaining <= 0:
            break

        partial_sentences: list[str] = []
        sentences = _SENTENCE_SPLIT.split(para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_tokens = count_tokens(sentence)
            if used + sent_tokens <= max_tokens:
                partial_sentences.append(sentence)
                used += sent_tokens
            else:
                break
        if partial_sentences:
            result.append(" ".join(partial_sentences))
        break

    return "\n\n".join(result)
