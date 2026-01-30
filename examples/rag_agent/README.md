# RAG Agent with Behavior-Based Decomposition

This example demonstrates how to build an adaptive RAG (Retrieval-Augmented Generation) agent using OpenSymbolicAI's behavior-based decomposition pattern.

## Key Concept

The agent uses **decomposition behaviors** to teach the LLM different retrieval strategies. When you ask a question, the agent automatically selects the appropriate strategy based on query similarity to the examples.

### Strategies Learned via Decomposition

| Strategy | Pattern | When Used |
|----------|---------|-----------|
| **Simple QA** | `retrieve → extract` | Factual questions |
| **Reranked QA** | `retrieve → rerank → extract` | Complex technical queries |
| **Summarization** | `retrieve → summarize` | Overview requests |
| **Multi-hop** | `retrieve → extract → followup → retrieve → aggregate` | Questions requiring chained reasoning |
| **Comparison** | `retrieve(A) → retrieve(B) → compare` | Compare X vs Y |
| **Validated** | `retrieve → extract → validate` | Accuracy-critical queries |
| **Filtered** | `retrieve_filtered → extract` | Source-specific queries |

## Quick Start

### 1. Install Dependencies

```bash
cd examples/rag_agent
uv sync
```

### 2. Load Wikipedia Data

```bash
# Quick setup (3 topics, fast)
uv run python setup_data.py --quick

# Full setup (27 curated topics)
uv run python setup_data.py

# Custom topics
uv run python setup_data.py --topics "Quantum computing" "Climate change" "CRISPR"
```

### 3. Run the Agent

```bash
# Interactive mode
uv run python -m rag_agent.main

# Demo showcasing different strategies
uv run python -m rag_agent.main --demo

# Single query
uv run python -m rag_agent.main --query "What is machine learning?"
```

## Example Session

```
$ uv run python -m rag_agent.main --demo

Query Type: Simple factual query
Question: What is Python?
────────────────────────────────────────

Plan (showing strategy selection):
docs = self.retrieve("Python programming language", k=3)
context = self.combine_contexts(docs)
answer = self.extract_answer(context, "What is Python?")

Answer:
Python is a high-level, general-purpose programming language known for its
readability and versatility. It was created by Guido van Rossum and first
released in 1991...
```

```
Query Type: Comparison query
Question: Compare Python and Rust
────────────────────────────────────────

Plan (showing strategy selection):
python_docs = self.retrieve("Python programming features performance", k=3)
rust_docs = self.retrieve("Rust programming features performance", k=3)
python_context = self.combine_contexts(python_docs)
rust_context = self.combine_contexts(rust_docs)
comparison = self.compare_topics(python_context, rust_context, "Python", "Rust", ["performance", "memory safety", "ease of learning"])

Answer:
**Performance**: Rust offers near-C performance with zero-cost abstractions...
**Memory Safety**: Rust's ownership system prevents memory errors at compile time...
**Ease of Learning**: Python has a gentler learning curve...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│              "Compare Python and Rust"                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  RAGAgent.run(query) │
              │  Uses PlanExecute    │
              └──────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    ┌─────────┐                   ┌──────────────┐
    │ PLANNING│                   │  EXECUTION   │
    └─────────┘                   └──────────────┘
         │                               │
    LLM sees:                      Executes plan:
    - 10 primitives               - retrieve("Python...")
    - 7 decomposition             - retrieve("Rust...")
      examples                     - compare_topics(...)
    - Selects strategy                   │
      based on similarity                ▼
                                  ┌──────────────┐
                                  │   ANSWER     │
                                  └──────────────┘
```

## Primitives

The agent has these atomic operations:

| Primitive | Purpose |
|-----------|---------|
| `retrieve(query, k)` | Semantic search for relevant documents |
| `retrieve_filtered(query, source, topic, k)` | Search with metadata filters |
| `rerank(documents, query, k)` | LLM-based relevance reranking |
| `combine_contexts(documents)` | Merge documents into single context |
| `extract_answer(context, question)` | Extract answer from context |
| `summarize(text, max_sentences)` | Condense text to key points |
| `generate_followup_query(question, partial)` | Generate follow-up for multi-hop |
| `aggregate_answers(answers, question)` | Combine partial answers |
| `compare_topics(ctx1, ctx2, name1, name2, aspects)` | Structured comparison |
| `validate_answer(answer, context)` | Check answer is supported by sources |

## Customization

### Add Your Own Data

```python
from rag_agent.retriever import ChromaRetriever

retriever = ChromaRetriever()

# Add custom documents
retriever.add_documents(
    documents=["Your document text here..."],
    metadatas=[{"source": "custom", "topic": "my-topic"}],
)
```

### Use Different LLM Providers

```bash
# OpenAI
export OPENAI_API_KEY=your-key
uv run python -m rag_agent.main --provider openai --model gpt-4

# Anthropic
export ANTHROPIC_API_KEY=your-key
uv run python -m rag_agent.main --provider anthropic --model claude-3-sonnet-20240229

# Local Ollama (default)
uv run python -m rag_agent.main --model gpt-oss:20b
```

### Add Custom Decomposition Behaviors

```python
from rag_agent import RAGAgent
from opensymbolicai.core import decomposition

class MyRAGAgent(RAGAgent):
    @decomposition(
        intent="Find recent news about a topic",
        expanded_intent="Filter by date, retrieve recent content, summarize"
    )
    def _recent_news(self) -> str:
        docs = self.retrieve_filtered(
            query="topic developments",
            source="news",
            k=5
        )
        context = self.combine_contexts(docs)
        return self.summarize(context, max_sentences=3)
```

## Files

```
examples/rag_agent/
├── pyproject.toml           # Dependencies
├── setup_data.py            # Data loading script
├── README.md                # This file
└── rag_agent/
    ├── __init__.py
    ├── agent.py             # RAGAgent with decomposition behaviors
    ├── main.py              # CLI entry point
    ├── models.py            # Document, ValidationResult
    ├── retriever.py         # ChromaDB wrapper
    └── wikipedia_loader.py  # Wikipedia ingestion
```
