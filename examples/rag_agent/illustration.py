"""
Illustration comparing Behaviour Programming vs Tool-Calling approaches.

This illustration measures:
- LLM calls: How many times the LLM is invoked
- Token usage: Input and output tokens consumed
- Time: Execution time for each approach

Key insight from "LLM Attention is Precious":
- Behaviour programming uses ONE planning call + N execution calls
- Tool-calling uses N planning calls (one per tool decision) + M execution calls

The difference is in the orchestration overhead:
- Behaviour: LLM plans once, then primitives execute
- Tool-call: LLM re-plans after every tool result
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from opensymbolicai.llm import LLMConfig, Provider

from rag_agent.agent import RAGAgent
from rag_agent.retriever import ChromaRetriever
from rag_agent.tool_call_agent import ToolCallRAGAgent, LLMUsageMetrics


@dataclass
class QueryResult:
    """Result from running a single query."""

    query: str
    answer: str
    llm_calls: int
    planning_calls: int
    execution_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    execution_time_seconds: float
    plan: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ApproachResult:
    """Aggregated results for an approach."""

    approach: str
    total_queries: int
    total_llm_calls: int
    total_planning_calls: int
    total_execution_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_time_seconds: float
    avg_llm_calls_per_query: float
    avg_tokens_per_query: float
    avg_time_per_query: float
    query_results: list[QueryResult] = field(default_factory=list)


@dataclass
class IllustrationReport:
    """Complete illustration comparison report."""

    timestamp: str
    model: str
    provider: str
    queries: list[str]
    behaviour_results: ApproachResult
    tool_call_results: ApproachResult
    summary: dict[str, Any] = field(default_factory=dict)


# Test queries that exercise different RAG strategies
ILLUSTRATION_QUERIES = [
    # Simple QA (retrieve -> extract)
    "What is machine learning?",
    "What is artificial intelligence?",
    # Comparison (parallel retrieve -> compare)
    "How do supervised and unsupervised learning differ?",
    # Multi-concept
    "What is deep learning and how does it relate to neural networks?",
    # Summarization
    "Give me an overview of Python programming language",
]


def run_behaviour_approach(
    agent: RAGAgent,
    queries: list[str],
) -> ApproachResult:
    """Run queries using the behaviour programming approach."""
    query_results: list[QueryResult] = []

    for query in queries:
        start_time = time.time()

        try:
            result = agent.run(query)
            elapsed = time.time() - start_time

            # Extract metrics from behaviour agent
            plan_metrics = result.metrics
            exec_metrics = agent.execution_metrics

            # Planning phase metrics (from PlanExecute base class)
            plan_input_tokens = 0
            plan_output_tokens = 0
            if plan_metrics and plan_metrics.plan_tokens:
                plan_input_tokens = plan_metrics.plan_tokens.input_tokens
                plan_output_tokens = plan_metrics.plan_tokens.output_tokens

            # Execution phase metrics (from our tracking in _llm_generate)
            exec_calls = exec_metrics.llm_calls
            exec_input_tokens = exec_metrics.input_tokens
            exec_output_tokens = exec_metrics.output_tokens

            # Total metrics: 1 planning call + N execution calls
            planning_calls = 1
            total_input_tokens = plan_input_tokens + exec_input_tokens
            total_output_tokens = plan_output_tokens + exec_output_tokens

            query_results.append(
                QueryResult(
                    query=query,
                    answer=str(result.result)[:500],
                    llm_calls=planning_calls + exec_calls,
                    planning_calls=planning_calls,
                    execution_calls=exec_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_input_tokens + total_output_tokens,
                    execution_time_seconds=elapsed,
                    plan=result.plan,
                )
            )

        except Exception as e:
            elapsed = time.time() - start_time
            query_results.append(
                QueryResult(
                    query=query,
                    answer=f"Error: {e}",
                    llm_calls=0,
                    planning_calls=0,
                    execution_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    execution_time_seconds=elapsed,
                )
            )

    # Aggregate results
    total_llm_calls = sum(r.llm_calls for r in query_results)
    total_planning = sum(r.planning_calls for r in query_results)
    total_execution = sum(r.execution_calls for r in query_results)
    total_input = sum(r.input_tokens for r in query_results)
    total_output = sum(r.output_tokens for r in query_results)
    total_tokens = sum(r.total_tokens for r in query_results)
    total_time = sum(r.execution_time_seconds for r in query_results)
    n = len(queries)

    return ApproachResult(
        approach="behaviour_programming",
        total_queries=n,
        total_llm_calls=total_llm_calls,
        total_planning_calls=total_planning,
        total_execution_calls=total_execution,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        avg_llm_calls_per_query=total_llm_calls / n if n else 0,
        avg_tokens_per_query=total_tokens / n if n else 0,
        avg_time_per_query=total_time / n if n else 0,
        query_results=query_results,
    )


def run_tool_call_approach(
    agent: ToolCallRAGAgent,
    queries: list[str],
) -> ApproachResult:
    """Run queries using the tool-calling approach."""
    query_results: list[QueryResult] = []

    for query in queries:
        start_time = time.time()

        try:
            result = agent.run(query)
            elapsed = time.time() - start_time

            metrics = result.metrics
            query_results.append(
                QueryResult(
                    query=query,
                    answer=result.answer[:500],
                    llm_calls=metrics.llm_calls,
                    planning_calls=metrics.planning_calls,
                    execution_calls=metrics.execution_calls,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    total_tokens=metrics.total_tokens,
                    execution_time_seconds=elapsed,
                    tool_calls=result.tool_calls,
                )
            )

        except Exception as e:
            elapsed = time.time() - start_time
            import traceback
            print(f"    ERROR in tool-call: {e}")
            if "-v" in sys.argv or "--verbose" in sys.argv:
                traceback.print_exc()
            query_results.append(
                QueryResult(
                    query=query,
                    answer=f"Error: {e}",
                    llm_calls=0,
                    planning_calls=0,
                    execution_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    execution_time_seconds=elapsed,
                )
            )

    # Aggregate results
    total_llm_calls = sum(r.llm_calls for r in query_results)
    total_planning = sum(r.planning_calls for r in query_results)
    total_execution = sum(r.execution_calls for r in query_results)
    total_input = sum(r.input_tokens for r in query_results)
    total_output = sum(r.output_tokens for r in query_results)
    total_tokens = sum(r.total_tokens for r in query_results)
    total_time = sum(r.execution_time_seconds for r in query_results)
    n = len(queries)

    return ApproachResult(
        approach="tool_calling",
        total_queries=n,
        total_llm_calls=total_llm_calls,
        total_planning_calls=total_planning,
        total_execution_calls=total_execution,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        avg_llm_calls_per_query=total_llm_calls / n if n else 0,
        avg_tokens_per_query=total_tokens / n if n else 0,
        avg_time_per_query=total_time / n if n else 0,
        query_results=query_results,
    )


def generate_report(report: IllustrationReport) -> str:
    """Generate a readable comparison report."""
    lines = [
        "=" * 80,
        "BEHAVIOUR PROGRAMMING vs TOOL-CALLING ILLUSTRATION",
        "=" * 80,
        f"Timestamp: {report.timestamp}",
        f"Model: {report.provider}/{report.model}",
        f"Queries tested: {len(report.queries)}",
        "",
        "-" * 80,
        "SUMMARY COMPARISON",
        "-" * 80,
        "",
        f"{'Metric':<35} {'Behaviour':<20} {'Tool-Call':<20}",
        "-" * 80,
    ]

    b = report.behaviour_results
    t = report.tool_call_results

    metrics = [
        ("Total LLM Calls", b.total_llm_calls, t.total_llm_calls),
        ("  - Planning Calls", b.total_planning_calls, t.total_planning_calls),
        ("  - Execution Calls", b.total_execution_calls, t.total_execution_calls),
        ("Avg LLM Calls/Query", f"{b.avg_llm_calls_per_query:.1f}", f"{t.avg_llm_calls_per_query:.1f}"),
        ("Total Tokens", b.total_tokens, t.total_tokens),
        ("  - Input Tokens", b.total_input_tokens, t.total_input_tokens),
        ("  - Output Tokens", b.total_output_tokens, t.total_output_tokens),
        ("Avg Tokens/Query", f"{b.avg_tokens_per_query:.0f}", f"{t.avg_tokens_per_query:.0f}"),
        ("Total Time (s)", f"{b.total_time_seconds:.2f}", f"{t.total_time_seconds:.2f}"),
        ("Avg Time/Query (s)", f"{b.avg_time_per_query:.2f}", f"{t.avg_time_per_query:.2f}"),
    ]

    for name, bval, tval in metrics:
        lines.append(f"{name:<35} {str(bval):<20} {str(tval):<20}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("KEY INSIGHT: LLM Attention is Precious")
    lines.append("-" * 80)
    lines.append("")

    # Calculate savings
    if t.total_planning_calls > 0 and b.total_planning_calls > 0:
        planning_reduction = (1 - b.total_planning_calls / t.total_planning_calls) * 100
        lines.append(f"Planning calls reduced by: {planning_reduction:.0f}%")
        lines.append(f"  Behaviour: {b.total_planning_calls} planning calls (1 per query)")
        lines.append(f"  Tool-call: {t.total_planning_calls} planning calls (1 per tool decision)")

    if t.total_tokens > 0 and b.total_tokens > 0:
        token_diff = t.total_tokens - b.total_tokens
        if token_diff > 0:
            lines.append(f"\nToken savings: {token_diff} tokens ({token_diff / t.total_tokens * 100:.0f}% reduction)")
        else:
            lines.append(f"\nToken overhead: {abs(token_diff)} tokens more for behaviour approach")

    lines.append("")
    lines.append("The behaviour approach saves LLM attention by:")
    lines.append("1. Planning once upfront instead of after each tool result")
    lines.append("2. Avoiding context accumulation in the agentic loop")
    lines.append("3. Keeping tool execution local (no LLM round-trip for orchestration)")
    lines.append("")

    # Per-query details
    lines.append("=" * 80)
    lines.append("PER-QUERY DETAILS")
    lines.append("=" * 80)

    for i, query in enumerate(report.queries):
        b_result = b.query_results[i] if i < len(b.query_results) else None
        t_result = t.query_results[i] if i < len(t.query_results) else None

        lines.append(f"\nQuery {i+1}: {query[:60]}...")
        lines.append("-" * 60)

        if b_result:
            lines.append(f"  Behaviour: {b_result.llm_calls} LLM calls, {b_result.total_tokens} tokens, {b_result.execution_time_seconds:.2f}s")
            if b_result.plan:
                plan_preview = b_result.plan.replace("\n", " ")[:80]
                lines.append(f"    Plan: {plan_preview}...")

        if t_result:
            lines.append(f"  Tool-call: {t_result.llm_calls} LLM calls, {t_result.total_tokens} tokens, {t_result.execution_time_seconds:.2f}s")
            if t_result.tool_calls:
                tool_names = [tc.get("tool", "?") for tc in t_result.tool_calls]
                lines.append(f"    Tools: {' -> '.join(tool_names)}")

    return "\n".join(lines)


def main() -> int:
    """Run the illustration comparison."""
    parser = argparse.ArgumentParser(
        description="Compare Behaviour Programming vs Tool-Calling for RAG"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model to use (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "groq", "fireworks"],
        default="groq",
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Custom queries to test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    # Set up retriever
    retriever = ChromaRetriever()
    if retriever.count() == 0:
        print("ERROR: Knowledge base is empty. Run setup_data.py first:")
        print("  python setup_data.py --quick")
        return 1

    print(f"Knowledge base: {retriever.count()} documents")

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

    print(f"Using model: {args.provider}/{args.model}")

    # Use custom queries or defaults
    queries = args.queries or ILLUSTRATION_QUERIES

    print(f"\nRunning {len(queries)} queries with both approaches...")
    print()

    # Create agents
    behaviour_agent = RAGAgent(llm=config, retriever=retriever)
    tool_call_agent = ToolCallRAGAgent(llm=config, retriever=retriever)

    # Run behaviour approach
    print("=" * 60)
    print("Running BEHAVIOUR PROGRAMMING approach...")
    print("=" * 60)
    behaviour_results = run_behaviour_approach(behaviour_agent, queries)

    for i, r in enumerate(behaviour_results.query_results):
        status = "OK" if "Error" not in r.answer else "ERR"
        print(f"  [{status}] Query {i+1}: {r.llm_calls} calls, {r.total_tokens} tokens, {r.execution_time_seconds:.2f}s")

    print()

    # Run tool-call approach
    print("=" * 60)
    print("Running TOOL-CALLING approach...")
    print("=" * 60)
    tool_call_results = run_tool_call_approach(tool_call_agent, queries)

    for i, r in enumerate(tool_call_results.query_results):
        status = "OK" if "Error" not in r.answer else "ERR"
        print(f"  [{status}] Query {i+1}: {r.llm_calls} calls, {r.total_tokens} tokens, {r.execution_time_seconds:.2f}s")

    print()

    # Generate report
    report = IllustrationReport(
        timestamp=datetime.now().isoformat(),
        model=args.model,
        provider=args.provider,
        queries=queries,
        behaviour_results=behaviour_results,
        tool_call_results=tool_call_results,
        summary={
            "behaviour_total_calls": behaviour_results.total_llm_calls,
            "tool_call_total_calls": tool_call_results.total_llm_calls,
            "behaviour_total_tokens": behaviour_results.total_tokens,
            "tool_call_total_tokens": tool_call_results.total_tokens,
        },
    )

    print(generate_report(report))

    # Save JSON report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nJSON report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
