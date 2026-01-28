# Date Agent

A minimal example demonstrating how to build an agent with OpenSymbolicAI that calculates days between dates.

## What This Example Shows

This example demonstrates the core concepts of OpenSymbolicAI:

- **`PlanExecute` blueprint**: A base class for agents that plan before executing
- **`@primitive`**: Marks methods as atomic operations the agent can use
- **`@decomposition`**: Provides example task breakdowns to guide the LLM

## How It Works

The `DateAgent` class defines two primitives:

1. `today()` - Returns the current date in ISO format
2. `days_between(start, end)` - Calculates the number of days between two dates

When you run a query like "How many days from Jan 1, 2026 to Valentine's Day 2026?", the agent:

1. Creates a plan using the available primitives
2. Executes the plan step by step
3. Returns the result

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama running locally (or configure a different LLM provider)

## Run

```bash
cd examples/date_agent
uv sync
uv run python -m date_agent.main
```

## Expected Output

```
Plan:
1. Parse "Jan 1, 2026" as start date
2. Parse "Valentine's Day 2026" as 2026-02-14
3. Call days_between("2026-01-01", "2026-02-14")

Result: 44
```

## Customization

To use a different LLM provider, modify the config in `main.py`:

```python
# For OpenAI
config = LLMConfig(provider=Provider.OPENAI, model="gpt-4")

# For Anthropic
config = LLMConfig(provider=Provider.ANTHROPIC, model="claude-3-opus")
```
