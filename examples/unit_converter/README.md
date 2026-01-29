# Unit Converter

A cooking/volume measurement converter using OpenSymbolicAI's PlanExecute blueprint.

## Primitives

The agent has 14 primitives for adjacent unit conversions:

```
tsp ↔ tbsp ↔ cups ↔ pints ↔ quarts ↔ gallons
               ↓
              ml ↔ liters
```

Multi-step conversions (e.g., teaspoons to cups) require chaining multiple primitives.

## Run

```bash
cd examples/unit_converter
uv run python unit_converter/main.py
```

## Benchmark

Run all 10 intents against an Ollama model:

```bash
cd examples/unit_converter
uv run python benchmark.py --models qwen3:4b
```

Options:

```
--models MODEL [MODEL ...]   Ollama models to test
--limit N                    Run only first N intents
--category simple|complex    Filter by category
-v, --verbose                Show pass/fail for each intent
-p, --parallel N             Run N intents in parallel
```

Examples:

```bash
# Run only complex (multi-step) intents
uv run python benchmark.py --models qwen3:4b --category complex

# Compare multiple models
uv run python benchmark.py --models qwen3:4b llama3.2

# Verbose output with 4 parallel workers
uv run python benchmark.py --models qwen3:4b -v -p 4
```

Results are saved to `results/` (gitignored).
