"""Wikipedia data loader for RAG knowledge base."""

from __future__ import annotations

import re
from typing import Any

import wikipediaapi

from rag_agent.retriever import ChromaRetriever


# Default topics covering diverse knowledge areas
DEFAULT_TOPICS = [
    # Technology & Programming
    "Python (programming language)",
    "Rust (programming language)",
    "JavaScript",
    "Go (programming language)",
    "Machine learning",
    "Artificial intelligence",
    "Neural network",
    "Transformer (deep learning architecture)",
    "Large language model",
    "Natural language processing",
    # Science
    "Quantum computing",
    "Climate change",
    "Photosynthesis",
    "DNA",
    "Theory of relativity",
    # Companies & People
    "OpenAI",
    "Google",
    "Tesla, Inc.",
    "Elon Musk",
    "Sam Altman",
    # Health & Lifestyle
    "Green tea",
    "Meditation",
    "Exercise",
    # Databases & Infrastructure
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Docker (software)",
    "Kubernetes",
]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split
        chunk_size: Target size of each chunk in words
        chunk_overlap: Number of overlapping words between chunks

    Returns:
        List of text chunks
    """
    # Clean the text
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        if len(words) > 20:  # Minimum chunk size
            chunks.append(text)
        return chunks

    step = chunk_size - chunk_overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)

        # Only add chunks with substantial content
        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


def extract_sections(page: wikipediaapi.WikipediaPage) -> list[dict[str, str]]:
    """
    Extract sections from a Wikipedia page.

    Returns:
        List of dicts with 'title' and 'content' keys
    """
    sections = []

    def process_section(
        section: wikipediaapi.WikipediaPageSection,
        parent_title: str = "",
    ) -> None:
        title = f"{parent_title} > {section.title}" if parent_title else section.title
        if section.text.strip():
            sections.append({"title": title, "content": section.text})
        for subsection in section.sections:
            process_section(subsection, title)

    # Add the summary as the first section
    if page.summary.strip():
        sections.append({"title": "Summary", "content": page.summary})

    # Process all sections
    for section in page.sections:
        process_section(section)

    return sections


def load_wikipedia_topics(
    topics: list[str] | None = None,
    retriever: ChromaRetriever | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    max_chunks_per_topic: int = 30,
    verbose: bool = True,
) -> ChromaRetriever:
    """
    Load Wikipedia articles into a ChromaDB retriever.

    Args:
        topics: List of Wikipedia article titles to load.
                If None, uses DEFAULT_TOPICS.
        retriever: Existing retriever to add to. If None, creates a new one.
        chunk_size: Number of words per chunk.
        chunk_overlap: Overlapping words between chunks.
        max_chunks_per_topic: Maximum chunks to store per topic.
        verbose: Whether to print progress.

    Returns:
        The ChromaRetriever with loaded documents.
    """
    if topics is None:
        topics = DEFAULT_TOPICS

    if retriever is None:
        retriever = ChromaRetriever()

    wiki = wikipediaapi.Wikipedia(
        user_agent="RAGAgent/1.0 (https://github.com/OpenSymbolicAI/examples-py)",
        language="en",
    )

    total_chunks = 0
    failed_topics = []

    for topic in topics:
        if verbose:
            print(f"Loading: {topic}...", end=" ", flush=True)

        page = wiki.page(topic)

        if not page.exists():
            if verbose:
                print("NOT FOUND")
            failed_topics.append(topic)
            continue

        # Extract sections for better metadata
        sections = extract_sections(page)

        topic_chunks = 0
        for section in sections:
            if topic_chunks >= max_chunks_per_topic:
                break

            chunks = chunk_text(
                section["content"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for i, chunk in enumerate(chunks):
                if topic_chunks >= max_chunks_per_topic:
                    break

                doc_id = f"{topic.replace(' ', '_')}_{section['title'].replace(' ', '_')}_{i}"
                # Sanitize the ID
                doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)[:100]

                metadata: dict[str, Any] = {
                    "source": "wikipedia",
                    "topic": topic,
                    "section": section["title"],
                    "url": page.fullurl,
                }

                retriever.add_documents(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[doc_id],
                )

                topic_chunks += 1

        total_chunks += topic_chunks
        if verbose:
            print(f"{topic_chunks} chunks")

    if verbose:
        print(f"\nTotal: {total_chunks} chunks loaded")
        if failed_topics:
            print(f"Failed to load: {', '.join(failed_topics)}")
        print(f"Collection now has {retriever.count()} documents")

    return retriever


def search_and_load_related(
    seed_topic: str,
    retriever: ChromaRetriever | None = None,
    max_related: int = 5,
    verbose: bool = True,
) -> ChromaRetriever:
    """
    Load a topic and its related/linked articles.

    Args:
        seed_topic: Starting Wikipedia article
        retriever: Existing retriever to add to
        max_related: Maximum number of related articles to load
        verbose: Whether to print progress

    Returns:
        The ChromaRetriever with loaded documents
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="RAGAgent/1.0 (https://github.com/OpenSymbolicAI/examples-py)",
        language="en",
    )

    page = wiki.page(seed_topic)
    if not page.exists():
        raise ValueError(f"Topic not found: {seed_topic}")

    # Get linked articles
    links = list(page.links.keys())[:max_related]
    topics = [seed_topic] + links

    if verbose:
        print(f"Loading {seed_topic} and {len(links)} related articles...")

    return load_wikipedia_topics(topics, retriever=retriever, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Wikipedia articles into RAG knowledge base")
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Wikipedia topics to load (default: curated list)",
    )
    parser.add_argument(
        "--seed",
        help="Load a topic and its related articles",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=5,
        help="Max related articles when using --seed",
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
        help="Max chunks per topic (default: 30)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before loading",
    )
    args = parser.parse_args()

    retriever = ChromaRetriever()

    if args.clear:
        print("Clearing existing data...")
        retriever.clear()

    if args.seed:
        search_and_load_related(
            args.seed,
            retriever=retriever,
            max_related=args.max_related,
        )
    else:
        load_wikipedia_topics(
            topics=args.topics,
            retriever=retriever,
            chunk_size=args.chunk_size,
            max_chunks_per_topic=args.max_chunks,
        )
