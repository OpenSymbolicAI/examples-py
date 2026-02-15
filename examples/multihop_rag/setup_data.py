#!/usr/bin/env python3
"""
One-command data setup for Multihop RAG Agent.

Downloads the MultiHop-RAG corpus from HuggingFace and loads it into ChromaDB.
Run this before using the multihop RAG agent.

Usage:
    # Load all 609 articles
    uv run python setup_data.py

    # Quick setup with first 50 articles
    uv run python setup_data.py --quick

    # Limit to N articles
    uv run python setup_data.py --max-articles 100

    # Clear and reload
    uv run python setup_data.py --clear
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def _split_long_paragraph(text: str, chunk_size: int) -> list[str]:
    """Word-count split for a single paragraph that exceeds chunk_size."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 300,
    paragraph_overlap: int = 1,
) -> list[str]:
    """Split text into chunks respecting paragraph boundaries.

    Accumulates paragraphs until reaching the target word count, then
    starts a new chunk. Keeps the last ``paragraph_overlap`` paragraphs
    as overlap for continuity. Falls back to word-count splitting for
    single paragraphs that exceed the target.

    Args:
        text: The text to split.
        chunk_size: Target number of words per chunk.
        paragraph_overlap: Number of paragraphs to carry over between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= 20 else []

    # Split into paragraph units
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If the text has no paragraph breaks, fall back to word-count splitting
    if len(paragraphs) <= 1:
        raw = _split_long_paragraph(text, chunk_size)
        return [c for c in raw if len(c.split()) >= 20]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds chunk_size, split it separately
        if para_words > chunk_size:
            # Flush current accumulator first
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            chunks.extend(_split_long_paragraph(para, chunk_size))
            continue

        # Would this paragraph push us over the limit?
        if current_words + para_words > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last N paragraphs for continuity
            current = current[-paragraph_overlap:] if paragraph_overlap > 0 else []
            current_words = sum(len(p.split()) for p in current)

        current.append(para)
        current_words += para_words

    # Flush remaining
    if current:
        chunks.append("\n\n".join(current))

    # Filter too-small chunks
    return [c for c in chunks if len(c.split()) >= 20 and len(c) >= 100]


def load_corpus(
    max_articles: int | None = None,
    chunk_size: int = 300,
    max_chunks_per_article: int = 20,
    clear: bool = False,
) -> int:
    """Download MultiHop-RAG corpus and load into ChromaDB.

    Args:
        max_articles: Maximum number of articles to load (None = all).
        chunk_size: Words per chunk.
        max_chunks_per_article: Maximum chunks per article.
        clear: Clear existing data before loading.

    Returns:
        Total number of document chunks loaded.
    """
    from datasets import load_dataset

    from multihop_rag.retriever import ChromaRetriever

    retriever = ChromaRetriever()

    if clear:
        print("Clearing existing knowledge base...")
        retriever.clear()
        retriever.clear_initialization()

    current_count = retriever.count()
    if current_count > 0 and not clear:
        print(f"Existing knowledge base has {current_count} documents.")
        response = input("Add more data? [y/N]: ").strip().lower()
        if response != "y":
            print("Keeping existing data. Run with --clear to reload.")
            return current_count

    print("Downloading MultiHop-RAG corpus from HuggingFace...")
    dataset = load_dataset("yixuantt/MultiHopRAG", "corpus")
    corpus = dataset["train"]

    total_articles = len(corpus)
    if max_articles is not None:
        total_articles = min(total_articles, max_articles)

    print(f"Processing {total_articles} articles...")
    total_chunks = 0

    for i in range(total_articles):
        article = corpus[i]
        title = article.get("title", f"article_{i}")
        body = article.get("body", "")
        author = article.get("author", "")
        category = article.get("category", "")
        published_at = article.get("published_at", "")
        source = article.get("source", "")
        url = article.get("url", "")

        if not body:
            continue

        chunks = chunk_text(body, chunk_size=chunk_size)
        if not chunks:
            continue

        # Limit chunks per article
        chunks = chunks[:max_chunks_per_article]

        # Prepend article header so embeddings carry source context
        header = f"[{title or 'Untitled'}] ({source or 'unknown'}, {published_at or 'n/a'})"
        chunks = [f"{header}\n\n{chunk}" for chunk in chunks]

        # Parse published_at to epoch seconds for ChromaDB numeric filtering
        published_ts = 0
        if published_at:
            try:
                dt = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                )
                published_ts = int(dt.timestamp())
            except ValueError:
                published_ts = 0

        base_meta = {
            "title": title or "",
            "author": author or "",
            "category": category or "",
            "published_at": published_at or "",
            "published_ts": published_ts,
            "source": source or "",
            "url": url or "",
        }
        metadatas = [
            {**base_meta, "chunk_index": ci} for ci in range(len(chunks))
        ]

        retriever.add_documents(
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

        if (i + 1) % 50 == 0 or (i + 1) == total_articles:
            print(f"  Processed {i + 1}/{total_articles} articles ({total_chunks} chunks)")

    retriever.mark_initialized()
    return retriever.count()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup Multihop RAG knowledge base with MultiHop-RAG corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with first 50 articles",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to load",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before loading",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Words per chunk (default: 300)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Maximum chunks per article (default: 20)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Multihop RAG Agent - Knowledge Base Setup")
    print("=" * 60)

    max_articles = args.max_articles
    if args.quick:
        max_articles = 50
        print("\nQuick setup mode - loading first 50 articles...")

    total = load_corpus(
        max_articles=max_articles,
        chunk_size=args.chunk_size,
        max_chunks_per_article=args.max_chunks,
        clear=args.clear,
    )

    print("\n" + "=" * 60)
    print(f"Setup complete! Knowledge base now has {total} documents.")
    print("=" * 60)
    print("\nYou can now run the multihop RAG agent:")
    print("  uv run python -m multihop_rag.main")
    print("\nOr run demo queries:")
    print("  uv run python -m multihop_rag.main --demo")

    return 0


if __name__ == "__main__":
    sys.exit(main())
