# RAG Agent with Behavior-Based Decomposition

This example demonstrates how to build an adaptive RAG (Retrieval-Augmented Generation) agent using OpenSymbolicAI's behavior-based decomposition pattern. It also includes an **illustration comparing behaviour programming vs tool-calling** to show why [LLM attention is precious](https://opensymbolicai.com/blog/llm-attention-is-precious).

![RAG Agent Demo](assets/demo.gif)

## Key Concept

The agent uses **decomposition behaviors** to teach the LLM different retrieval strategies. When you ask a question, the agent automatically selects the appropriate strategy based on query similarity to the examples.

**Why this matters:** Traditional tool-calling agents make the LLM re-read all previous tool results on every call, wasting tokens exponentially. Behaviour programming plans once, then executes in Python.

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
# Interactive mode (behaviour programming - default)
uv run python -m rag_agent.main

# Tool-calling mode (for comparison)
uv run python -m rag_agent.main --mode tool-call

# Demo showcasing different strategies
uv run python -m rag_agent.main --demo

# Single query
uv run python -m rag_agent.main --query "What is machine learning?"
```

## Behaviour Programming vs Tool-Calling Illustration

This example includes both approaches so you can measure the difference:

```bash
# Run the comparison illustration
uv run python illustration.py

# With custom model
uv run python illustration.py --model openai/gpt-oss-20b --provider groq
```

### Why LLM Attention is Precious

With **tool calling**, every step goes through the LLM. Each call includes all previous results:

```
┌─────────────────────────────────────────────────────────┐
│  CALL 1: "Get ML docs"         →  3,010 tokens read    │
│  CALL 2: "Get DL docs"         →  5,610 tokens read    │  ← re-reads ML docs
│  CALL 3: "Combine contexts"    →  8,210 tokens read    │  ← re-reads both
│  CALL 4: "Compare them"        →  9,810 tokens read    │  ← re-reads all
│  CALL 5: "Format answer"       → 10,410 tokens read    │
└─────────────────────────────────────────────────────────┘
                    TOTAL: ~37,000 tokens
```

With **behaviour programming**, the LLM plans once, then Python executes:

```
┌─────────────────────────────────────────────────────────┐
│  PLANNING CALL                 →  1,010 tokens read    │
│  (LLM outputs the plan)                                 │
└─────────────────────────────────────────────────────────┘
                    ↓ Python executes plan
┌─────────────────────────────────────────────────────────┐
│  COMPARE PRIMITIVE             →  3,100 tokens read    │
│  (Only when LLM actually needs to compare)              │
└─────────────────────────────────────────────────────────┘
                    TOTAL: ~5,000 tokens
```

### Illustration Results

| Metric | Behaviour | Tool-Calling |
|--------|-----------|--------------|
| LLM Calls | 2 | 5 |
| Tokens Processed | ~5,000 | ~37,000 |
| **Savings** | | **~7x fewer tokens** |

The gap grows with:
- **More steps** = more re-reading in tool calling
- **Bigger documents** = more wasted tokens per re-read
- **Multiple agents** = each agent re-reads everything

> **Send logic to the LLM. Keep data in Python.**

Learn more: [Behaviour Programming vs Tool-Calling](https://opensymbolicai.com/blog/behaviour-programming-vs-tool-calling)

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

### LLM Providers and Models

To run this example, you need one of the following:

1. **Ollama (Local)** - Default option, no API key required
2. **Cloud Provider** - OpenAI, Anthropic, Groq, or Fireworks with API key
3. **Custom LLM** - Implement your own LLM class

#### Supported Providers

| Provider | Env Variable |
|----------|--------------|
| `ollama` | None (local) |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `fireworks` | `FIREWORKS_API_KEY` |

#### Usage Examples

```bash
# Local Ollama (default - no API key needed)
uv run python -m rag_agent.main --model gpt-oss:20b

# Groq
export GROQ_API_KEY=your-key
uv run python -m rag_agent.main --provider groq --model openai/gpt-oss-20b

# OpenAI
export OPENAI_API_KEY=your-key
uv run python -m rag_agent.main --provider openai --model gpt-4

# Anthropic
export ANTHROPIC_API_KEY=your-key
uv run python -m rag_agent.main --provider anthropic --model claude-3-sonnet-20240229
```

#### Custom LLM Implementation

You can implement your own LLM class by extending the base `LLM` interface:

```python
from opensymbolicai.llm import LLM, LLMResponse

class MyCustomLLM(LLM):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation here
        response_text = call_your_model(prompt)
        return LLMResponse(content=response_text)

# Use with the agent
agent = RAGAgent(llm=MyCustomLLM(), retriever=retriever)
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
├── illustration.py          # Behaviour vs Tool-Calling comparison
├── README.md                # This file
└── rag_agent/
    ├── __init__.py
    ├── agent.py             # RAGAgent with decomposition behaviors
    ├── tool_call_agent.py   # ToolCallRAGAgent for comparison
    ├── main.py              # CLI entry point (supports --mode)
    ├── models.py            # Document, ValidationResult
    ├── retriever.py         # ChromaDB wrapper
    └── wikipedia_loader.py  # Wikipedia ingestion
```

## Related Reading

- [LLM Attention Is Precious: Why Tool Calling Wastes It](https://opensymbolicai.com/blog/llm-attention-is-precious)
- [Behaviour Programming vs Tool-Calling](https://opensymbolicai.com/blog/behaviour-programming-vs-tool-calling)
