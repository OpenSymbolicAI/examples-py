# Multihop RAG Agent

A multi-hop question-answering agent built with the **GoalSeeking** blueprint from `opensymbolicai-core`. It answers complex questions that require reasoning across multiple documents by iteratively retrieving evidence — each retrieval "hop" is one iteration in the GoalSeeking loop.

Uses the [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) benchmark dataset (609 news articles across tech, sports, entertainment, business, science, and health).

## Why GoalSeeking?

The [existing `rag_agent`](../rag_agent/) uses **PlanExecute** — it generates the entire multi-hop plan in one shot. This is brittle because the planner must anticipate all hops before seeing any evidence.

**GoalSeeking** makes each hop adaptive:

```
seek("Who is the individual linked to crypto that was found guilty?")

  Iteration 1:  retrieve("crypto individual guilty") → extract evidence
                → update_context: found "Sam Bankman-Fried", gap: "verdict details"
                → evaluate: no synthesized answer yet → CONTINUE

  Iteration 2:  retrieve("Bankman-Fried trial verdict") → extract → synthesize
                → update_context: answer ready
                → evaluate: answer synthesized → ACHIEVED

  → "Sam Bankman-Fried"
```

The planner sees accumulated knowledge (not raw results) and decides what to search next. The evaluator checks if an answer has been synthesized.

## Architecture

```
User Query
    │
    ▼
MultiHopRAGAgent.seek(query)
    │
    ├── create_context() → MultiHopContext(evidence=[], queries_tried=[], ...)
    │
    └── LOOP (max 5 iterations):
        │
        ├── 1. plan_iteration()     ← LLM sees accumulated evidence, plans next hop
        ├── 2. execute()            ← runs primitives (retrieve, extract, synthesize)
        ├── 3. update_context()     ← INTROSPECTION BOUNDARY: raw → structured insights
        ├── 4. evaluate()           ← @evaluator checks: answer synthesized?
        └── 5. should_continue()    ← stop if achieved or max iterations
            │
            ▼
        GoalSeekingResult(answer, iterations, status)
```

### The Introspection Boundary

`update_context()` is the key architectural feature. It converts raw `ExecutionResult` into structured fields on `MultiHopContext`:

| Primitive Called | Context Updated |
|---|---|
| `retrieve` | `queries_tried` — tracks search angles used |
| `extract_evidence` | `evidence` — accumulates extracted facts |
| `synthesize_answer` | `current_answer` + `sufficient` flag |

The planner and evaluator only see these structured fields — never the raw execution results.

### Primitives

| Primitive | Purpose |
|---|---|
| `retrieve(query, k)` | Semantic search over the news corpus |
| `combine_contexts(documents)` | Merge documents into a context string |
| `extract_evidence(context, question)` | Pull relevant facts from retrieved text |
| `generate_next_query(question, evidence)` | Plan the next retrieval hop |
| `synthesize_answer(question, evidence)` | Combine multi-source evidence into answer |

### Decompositions (few-shot examples)

Three patterns teach the LLM planner how to compose primitives:

1. **Two-hop inference** — `retrieve → extract → generate_next_query → retrieve → synthesize`
2. **Single retrieval** — `retrieve → extract → synthesize`
3. **Comparison** — `retrieve(A) → retrieve(B) → extract both → synthesize`

## Quick Start

```bash
cd examples/multihop_rag

# Install dependencies
uv sync

# Set up environment (requires FIREWORKS_API_KEY for embeddings + LLM)
cp .env.example .env  # then add your API key

# Load corpus into ChromaDB (quick: 50 articles, full: all 609)
uv run python setup_data.py --quick

# Interactive mode
uv run python -m multihop_rag.main

# Single query
uv run python -m multihop_rag.main --query "Who was found guilty in the crypto trial?"

# Demo: run 3 showcase queries
uv run python -m multihop_rag.main --demo
```

## CLI Reference

```
--model MODEL        LLM model name (default: accounts/fireworks/models/gpt-oss-120b)
--provider PROVIDER  ollama | openai | anthropic | groq | fireworks (default: fireworks)
--query / -q QUERY   Single query mode
--demo               Run demo queries showcasing different multi-hop patterns
--max-iterations N   GoalSeeking max iterations (default: 5)
--quick              Quick corpus setup (50 articles)
--reinit             Clear and reload corpus
```

## Comparison with `rag_agent`

| | `rag_agent` (PlanExecute) | `multihop_rag` (GoalSeeking) |
|---|---|---|
| Planning | One-shot: entire plan upfront | Iterative: one hop per iteration |
| Adaptivity | None — plan is fixed | Full — each hop informed by prior evidence |
| When to stop | When plan finishes | When evaluator says "answer ready" |
| Context | Raw execution result | Structured `MultiHopContext` |
| Dataset | Wikipedia (general) | MultiHop-RAG (news articles) |
