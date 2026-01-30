"""Main entry point for the RAG Agent example."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from opensymbolicai.llm import LLMConfig, Provider

# Load environment variables from .env file
load_dotenv()

from rag_agent.agent import RAGAgent
from rag_agent.retriever import ChromaRetriever
from rag_agent.wikipedia_loader import load_wikipedia_topics


def setup_knowledge_base(quick: bool = False) -> ChromaRetriever:
    """Set up the knowledge base with Wikipedia data."""
    retriever = ChromaRetriever()

    if retriever.count() == 0:
        print("Knowledge base is empty. Loading Wikipedia articles...")
        if quick:
            # Quick setup with fewer topics
            topics = [
                "Python (programming language)",
                "Machine learning",
                "Artificial intelligence",
            ]
        else:
            topics = None  # Use default comprehensive list

        load_wikipedia_topics(topics=topics, retriever=retriever)
    else:
        print(f"Using existing knowledge base with {retriever.count()} documents")

    return retriever


def interactive_mode(agent: RAGAgent) -> None:
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("RAG Agent Interactive Mode")
    print("=" * 60)
    print("Ask questions about the knowledge base.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...", flush=True)

        try:
            result = agent.run(query)

            print(f"\n{'─' * 40}")
            print("Generated Plan:")
            print(f"{'─' * 40}")
            print(result.plan)
            print(f"\n{'─' * 40}")
            print("Answer:")
            print(f"{'─' * 40}")
            print(result.result)
            print()

            if result.metrics:
                print(
                    f"[Tokens: {result.metrics.plan_tokens.total_tokens} | "
                    f"Time: {result.metrics.total_time_seconds:.2f}s]"
                )
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


def demo_queries(agent: RAGAgent) -> None:
    """Run a set of demo queries showcasing different RAG strategies."""
    queries = [
        # Simple QA - should use retrieve -> extract
        ("What are neural networks?", "Simple factual query"),
        # Comparison - should use parallel retrieve -> compare
        (
            "How do supervised and unsupervised learning differ?",
            "Comparison query",
        ),
        # Multi-hop - should use chained retrieval
        (
            "What is deep learning and what breakthroughs has it enabled?",
            "Multi-hop query",
        ),
        # Summarization - should use retrieve -> summarize
        ("Summarize the key applications of artificial intelligence", "Summarization query"),
    ]

    print("\n" + "=" * 60)
    print("RAG Agent Demo - Showcasing Different Strategies")
    print("=" * 60)

    for query, description in queries:
        print(f"\n{'─' * 60}")
        print(f"Query Type: {description}")
        print(f"Question: {query}")
        print("─" * 60)

        try:
            result = agent.run(query)

            print("\nPlan (showing strategy selection):")
            print(result.plan)
            print("\nAnswer:")
            print(result.result)

            if result.metrics:
                print(
                    f"\n[Tokens: {result.metrics.plan_tokens.total_tokens} | "
                    f"Time: {result.metrics.total_time_seconds:.2f}s]"
                )

        except Exception as e:
            print(f"\nError: {e}")

        print()
        input("Press Enter for next query...")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Agent with behavior-based decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default settings
  uv run python -m rag_agent.main

  # Quick setup (fewer Wikipedia articles)
  uv run python -m rag_agent.main --quick

  # Run demo queries
  uv run python -m rag_agent.main --demo

  # Single query
  uv run python -m rag_agent.main --query "What is machine learning?"

  # Use a different model
  uv run python -m rag_agent.main --model llama3.2
        """,
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model to use (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "groq"],
        default="groq",
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Single query to run (non-interactive)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo queries showcasing different strategies",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with fewer Wikipedia articles",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Clear and reload the knowledge base",
    )
    args = parser.parse_args()

    # Set up retriever
    retriever = ChromaRetriever()

    if args.reload:
        print("Clearing knowledge base...")
        retriever.clear()

    retriever = setup_knowledge_base(quick=args.quick)

    # Set up LLM config
    provider_map = {
        "ollama": Provider.OLLAMA,
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "groq": Provider.GROQ,
    }
    config = LLMConfig(
        provider=provider_map[args.provider],
        model=args.model,
    )

    print(f"\nUsing model: {args.provider}/{args.model}")

    # Create agent
    agent = RAGAgent(llm=config, retriever=retriever)

    # Run mode
    if args.query:
        # Single query mode
        print(f"\nQuery: {args.query}")
        result = agent.run(args.query)
        print(f"\nPlan:\n{result.plan}")
        print(f"\nAnswer:\n{result.result}")
        return 0

    elif args.demo:
        demo_queries(agent)
        return 0

    else:
        interactive_mode(agent)
        return 0


if __name__ == "__main__":
    sys.exit(main())
