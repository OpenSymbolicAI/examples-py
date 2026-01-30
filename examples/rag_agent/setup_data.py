#!/usr/bin/env python3
"""
One-command data setup for RAG Agent.

This script loads Wikipedia articles into the ChromaDB knowledge base.
Run this before using the RAG agent.

Usage:
    # Load default curated topics (recommended for first setup)
    uv run python setup_data.py

    # Quick setup with just 3 essential topics
    uv run python setup_data.py --quick

    # Load specific topics
    uv run python setup_data.py --topics "Python (programming language)" "Machine learning"

    # Load a topic and its related articles
    uv run python setup_data.py --seed "Artificial intelligence" --max-related 10

    # Clear and reload
    uv run python setup_data.py --clear
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup RAG knowledge base with Wikipedia data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific Wikipedia topics to load",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with minimal topics (Python, ML, AI)",
    )
    parser.add_argument(
        "--seed",
        help="Load a topic and its related/linked articles",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=5,
        help="Maximum related articles when using --seed (default: 5)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before loading",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Words per chunk (default: 400)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=30,
        help="Maximum chunks per topic (default: 30)",
    )
    args = parser.parse_args()

    from rag_agent.retriever import ChromaRetriever
    from rag_agent.wikipedia_loader import (
        load_wikipedia_topics,
        search_and_load_related,
    )

    print("=" * 60)
    print("RAG Agent - Knowledge Base Setup")
    print("=" * 60)

    retriever = ChromaRetriever()

    if args.clear:
        print("\nClearing existing knowledge base...")
        retriever.clear()
        print("Done.")

    current_count = retriever.count()
    if current_count > 0:
        print(f"\nExisting knowledge base has {current_count} documents.")
        if not args.clear and not args.topics and not args.seed:
            response = input("Add more data? [y/N]: ").strip().lower()
            if response != "y":
                print("Keeping existing data. Run with --clear to reload.")
                return 0

    print()

    if args.seed:
        print(f"Loading '{args.seed}' and up to {args.max_related} related articles...\n")
        search_and_load_related(
            args.seed,
            retriever=retriever,
            max_related=args.max_related,
        )

    elif args.quick:
        quick_topics = [
            "Python (programming language)",
            "Machine learning",
            "Artificial intelligence",
        ]
        print("Quick setup mode - loading 3 essential topics...\n")
        load_wikipedia_topics(
            topics=quick_topics,
            retriever=retriever,
            chunk_size=args.chunk_size,
            max_chunks_per_topic=args.max_chunks,
        )

    else:
        if args.topics:
            print(f"Loading {len(args.topics)} specified topics...\n")
        else:
            print("Loading default curated topic list...\n")

        load_wikipedia_topics(
            topics=args.topics,
            retriever=retriever,
            chunk_size=args.chunk_size,
            max_chunks_per_topic=args.max_chunks,
        )

    print("\n" + "=" * 60)
    print(f"Setup complete! Knowledge base now has {retriever.count()} documents.")
    print("=" * 60)
    print("\nYou can now run the RAG agent:")
    print("  uv run python -m rag_agent.main")
    print("\nOr run demo queries:")
    print("  uv run python -m rag_agent.main --demo")

    return 0


if __name__ == "__main__":
    sys.exit(main())
