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
from rag_agent.tool_call_agent import ToolCallRAGAgent
from rag_agent.wikipedia_loader import load_wikipedia_topics


def setup_knowledge_base(quick: bool = False, reinit: bool = False) -> ChromaRetriever:
    """Set up the knowledge base with Wikipedia data."""
    retriever = ChromaRetriever()

    if reinit:
        print("Reinitializing knowledge base...")
        retriever.clear()
        retriever.clear_initialization()

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
        retriever.mark_initialized()
        print("\nInitialization complete!")
    else:
        doc_count = retriever.count()
        print(f"Using existing knowledge base with {doc_count} documents")
        print("(Use --reinit to reload the knowledge base)")
        retriever.mark_initialized()

    return retriever


def interactive_mode(agent: RAGAgent | ToolCallRAGAgent, mode: str = "behaviour") -> None:
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print(f"RAG Agent Interactive Mode ({mode})")
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

            if mode == "tool-call":
                print(f"\n{'─' * 40}")
                print("Tool Calls:")
                print(f"{'─' * 40}")
                for tc in result.tool_calls:
                    print(f"  - {tc.get('tool')}")
                print(f"\n{'─' * 40}")
                print("Answer:")
                print(f"{'─' * 40}")
                print(result.answer)
                print()
                print(
                    f"[LLM Calls: {result.metrics.llm_calls} | "
                    f"Tokens: {result.metrics.total_tokens}]"
                )
            else:
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


def demo_queries(agent: RAGAgent | ToolCallRAGAgent, mode: str = "behaviour") -> None:
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
    print(f"RAG Agent Demo ({mode}) - Showcasing Different Strategies")
    print("=" * 60)

    for query, description in queries:
        print(f"\n{'─' * 60}")
        print(f"Query Type: {description}")
        print(f"Question: {query}")
        print("─" * 60)

        try:
            result = agent.run(query)

            if mode == "tool-call":
                print("\nTool calls:")
                for tc in result.tool_calls:
                    print(f"  - {tc.get('tool')}")
                print("\nAnswer:")
                print(result.answer)
                print(
                    f"\n[LLM Calls: {result.metrics.llm_calls} | "
                    f"Tokens: {result.metrics.total_tokens}]"
                )
            else:
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

  # Reinitialize the knowledge base from scratch
  uv run python -m rag_agent.main --reinit
        """,
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model to use (default: openai/gpt-oss-120b)",
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
        "--reinit",
        action="store_true",
        help="Clear and reload the knowledge base from scratch",
    )
    parser.add_argument(
        "--mode",
        choices=["behaviour", "tool-call"],
        default="behaviour",
        help="Agent mode: 'behaviour' (plan-execute) or 'tool-call' (agentic loop)",
    )
    args = parser.parse_args()

    # Set up retriever with knowledge base
    retriever = setup_knowledge_base(quick=args.quick, reinit=args.reinit)

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
    print(f"Agent mode: {args.mode}")

    # Create agent based on mode
    if args.mode == "tool-call":
        agent = ToolCallRAGAgent(llm=config, retriever=retriever)
    else:
        agent = RAGAgent(llm=config, retriever=retriever)

    # Run mode
    if args.query:
        # Single query mode
        print(f"\nQuery: {args.query}")
        result = agent.run(args.query)

        if args.mode == "tool-call":
            print(f"\nTool calls: {[tc.get('tool') for tc in result.tool_calls]}")
            print(f"\nAnswer:\n{result.answer}")
            print(f"\n[LLM Calls: {result.metrics.llm_calls} | Tokens: {result.metrics.total_tokens}]")
        else:
            print(f"\nPlan:\n{result.plan}")
            print(f"\nAnswer:\n{result.result}")
            if result.metrics:
                print(f"\n[Tokens: {result.metrics.plan_tokens.total_tokens}]")
        return 0

    elif args.demo:
        demo_queries(agent, mode=args.mode)
        return 0

    else:
        interactive_mode(agent, mode=args.mode)
        return 0


if __name__ == "__main__":
    sys.exit(main())
