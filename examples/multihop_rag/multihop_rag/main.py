"""Main entry point for the Multihop RAG Agent example."""

from __future__ import annotations

import argparse
import sys

from dotenv import find_dotenv, load_dotenv
from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import GoalSeekingResult

load_dotenv(dotenv_path=find_dotenv(usecwd=True))

from multihop_rag.agent import MultiHopRAGAgent  # noqa: E402
from multihop_rag.retriever import ChromaRetriever  # noqa: E402

# Demo queries showcasing different multi-hop patterns
DEMO_QUERIES = [
    (
        "Who is the individual associated with the cryptocurrency industry "
        "that was recently found guilty?",
        "Two-hop inference",
    ),
    (
        "What are the latest developments in AI chatbots?",
        "Single retrieval",
    ),
    (
        "How did technology companies and sports organizations approach "
        "their challenges differently in late 2023?",
        "Comparison across angles",
    ),
]


def setup_knowledge_base(quick: bool = False, reinit: bool = False) -> ChromaRetriever:
    """Set up the knowledge base with MultiHop-RAG corpus."""
    from setup_data import load_corpus

    retriever = ChromaRetriever()

    if reinit:
        print("Reinitializing knowledge base...")
        retriever.clear()
        retriever.clear_initialization()

    if retriever.count() == 0:
        print("Knowledge base is empty. Loading MultiHop-RAG corpus...")
        max_articles = 50 if quick else None
        load_corpus(max_articles=max_articles, clear=False)
        retriever = ChromaRetriever()
    else:
        doc_count = retriever.count()
        print(f"Using existing knowledge base with {doc_count} documents")
        print("(Use --reinit to reload the knowledge base)")

    return retriever


def display_result(result: GoalSeekingResult, query_type: str | None = None) -> None:
    """Display a GoalSeekingResult with iteration summary."""
    if query_type:
        print(f"  Type: {query_type}")

    for iteration in result.iterations:
        n = iteration.iteration_number
        plan = iteration.plan_result.plan
        plan_preview = plan[:200] + "..." if len(plan) > 200 else plan
        achieved = "achieved" if iteration.evaluation.goal_achieved else "continuing"
        print(f"\n  Iteration {n}: {achieved}")
        print(f"    Plan: {plan_preview}")

    print(f"\n  {'─' * 50}")
    print(f"  Answer: {result.final_answer}")
    print(f"  Iterations: {result.iteration_count} | Status: {result.status.value}")


def interactive_mode(agent: MultiHopRAGAgent) -> None:
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("Multihop RAG Agent - Interactive Mode (GoalSeeking)")
    print("=" * 60)
    print("Ask multi-hop questions about the news corpus.")
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

        print("\nSeeking answer...", flush=True)

        try:
            result = agent.seek(query)
            print(f"\n{'─' * 60}")
            display_result(result)
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


def demo_mode(agent: MultiHopRAGAgent) -> None:
    """Run demo queries showcasing different multi-hop patterns."""
    print(f"\n{'=' * 60}")
    print(f"Multihop RAG Agent - Demo ({len(DEMO_QUERIES)} queries)")
    print(f"{'=' * 60}")

    for i, (query, query_type) in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(DEMO_QUERIES)}] {query}")
        print(f"{'─' * 60}", flush=True)

        try:
            result = agent.seek(query)
            display_result(result, query_type=query_type)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print("Demo complete.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multihop RAG Agent with GoalSeeking pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  uv run python -m multihop_rag.main

  # Quick corpus setup (first 50 articles)
  uv run python -m multihop_rag.main --quick

  # Run demo queries
  uv run python -m multihop_rag.main --demo

  # Single query
  uv run python -m multihop_rag.main --query "Who was found guilty in the crypto trial?"

  # Use a different provider/model
  uv run python -m multihop_rag.main --model llama3.2 --provider ollama
        """,
    )
    parser.add_argument(
        "--model",
        default="accounts/fireworks/models/gpt-oss-120b",
        help="Model to use (default: accounts/fireworks/models/gpt-oss-120b)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "groq", "fireworks"],
        default="fireworks",
        help="LLM provider (default: fireworks)",
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to run (non-interactive)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo queries showcasing different multi-hop patterns",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick corpus setup (first 50 articles)",
    )
    parser.add_argument(
        "--reinit",
        action="store_true",
        help="Clear and reload the knowledge base from scratch",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum GoalSeeking iterations per query (default: 5)",
    )
    args = parser.parse_args()

    # Set up retriever
    retriever = setup_knowledge_base(quick=args.quick, reinit=args.reinit)

    # Set up LLM config
    provider_map = {
        "ollama": Provider.OLLAMA,
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "groq": Provider.GROQ,
        "fireworks": Provider.FIREWORKS,
    }
    config = LLMConfig(
        provider=provider_map[args.provider],
        model=args.model,
    )

    print(f"\nUsing model: {args.provider}/{args.model}")
    print(f"Max iterations: {args.max_iterations}")

    # Create agent
    agent = MultiHopRAGAgent(
        llm=config,
        retriever=retriever,
        max_iterations=args.max_iterations,
    )

    # Run mode
    if args.query:
        print(f"\nQuery: {args.query}")
        print("Seeking answer...\n")
        result = agent.seek(args.query)
        display_result(result)
        return 0

    elif args.demo:
        demo_mode(agent)
        return 0

    else:
        interactive_mode(agent)
        return 0


if __name__ == "__main__":
    sys.exit(main())
