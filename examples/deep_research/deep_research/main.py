"""Main entry point for the Deep Research Agent example."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import find_dotenv, load_dotenv
from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import GoalSeekingResult

load_dotenv(dotenv_path=find_dotenv(usecwd=True))

from deep_research.agent import DeepResearchAgent  # noqa: E402
from deep_research.models import Searcher  # noqa: E402
from deep_research.searcher import TavilySearcher  # noqa: E402

# Demo queries showcasing different research patterns
DEMO_QUERIES = [
    (
        "What are the latest breakthroughs and challenges in nuclear fusion energy research?",
        "Broad topic research",
    ),
    (
        "How do the AI regulation approaches of the EU, US, and China compare?",
        "Comparison research",
    ),
    (
        "What is the current state of mRNA vaccine technology beyond COVID-19?",
        "Deep dive research",
    ),
]


def display_result(result: GoalSeekingResult, query_type: str | None = None) -> None:
    """Display a GoalSeekingResult with the full report."""
    if query_type:
        print(f"  Type: {query_type}")

    for iteration in result.iterations:
        n = iteration.iteration_number
        plan = iteration.plan_result.plan
        plan_preview = plan[:200] + "..." if len(plan) > 200 else plan
        achieved = "achieved" if iteration.evaluation.goal_achieved else "continuing"
        print(f"\n  Iteration {n}: {achieved}")
        print(f"    Plan: {plan_preview}")

    print(f"\n  {'=' * 60}")
    print(f"  Status: {result.status.value} | Iterations: {result.iteration_count}")
    print(f"  {'=' * 60}")

    if result.final_answer:
        print(f"\n{result.final_answer}")
    else:
        print("\n  No report generated.")


def interactive_mode(agent: DeepResearchAgent) -> None:
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("Deep Research Agent - Interactive Mode (GoalSeeking)")
    print("=" * 60)
    print("Enter a research topic or question.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Research topic: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nResearching...", flush=True)

        try:
            result = agent.seek(query)
            print(f"\n{'=' * 60}")
            display_result(result)
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


def demo_mode(agent: DeepResearchAgent) -> None:
    """Run demo queries showcasing different research patterns."""
    print(f"\n{'=' * 60}")
    print(f"Deep Research Agent - Demo ({len(DEMO_QUERIES)} queries)")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Deep Research Agent with GoalSeeking pattern and Tavily search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  uv run python -m deep_research.main

  # Run demo queries
  uv run python -m deep_research.main --demo

  # Single query
  uv run python -m deep_research.main -q "What is CRISPR gene editing?"

  # Use a different provider/model
  uv run python -m deep_research.main --model claude-sonnet-4-5-20250929 --provider anthropic
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
        help="Single research query to run (non-interactive)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo queries showcasing different research patterns",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum GoalSeeking iterations per query (default: 8)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use pre-downloaded fixture data instead of live Tavily API",
    )
    args = parser.parse_args()

    # Build searcher: mock or live Tavily
    searcher: Searcher
    if args.mock:
        from deep_research.mock_searcher import MockSearcher

        searcher = MockSearcher()
    else:
        searcher = TavilySearcher()

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
    agent = DeepResearchAgent(
        llm=config,
        searcher=searcher,
        max_iterations=args.max_iterations,
    )

    # Run mode
    if args.query:
        print(f"\nResearch topic: {args.query}")
        print("Researching...\n")
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
