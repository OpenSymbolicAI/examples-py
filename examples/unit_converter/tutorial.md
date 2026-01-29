# Unit Converter Tutorial

This tutorial walks you through the Unit Converter agent, demonstrating how primitives and decompositions work in OpenSymbolicAI.

## Prerequisites

Make sure you're in the `examples/unit_converter` directory:

```bash
cd examples/unit_converter
```

## Step 1: Run a Simple Conversion

The agent is already set up to convert 20 gallons to liters. Run it with:

```bash
uv run python unit_converter/main.py
```

This executes the query `"20 gallons in liters"`. The agent will chain multiple primitives together:
- gallons → quarts → pints → cups → ml → liters

Observe how the plan shows the multi-step conversion path.

## Step 2: Run the Benchmark

To see how the agent performs across various conversion scenarios, run the benchmark:

```bash
uv run python benchmark.py --models qwen3:4b
```

This tests 10 different intents against the model and reports accuracy.

## Step 3: Try an Unsupported Conversion

Now let's try something the agent can't handle yet. Open `unit_converter/main.py` and change the query to:

```python
response = agent.run("Convert 1 hogshead to liters")
```

Run it again:

```bash
uv run python unit_converter/main.py
```

This will fail because the agent doesn't have primitives for hogshead conversions.

### Fix: Add Hogshead Support

In `unit_converter/main.py`, find the commented hogshead primitives (around lines 86-95) and uncomment them:

```python
# Gallons <-> Hogsheads (63 gallons = 1 hogshead)
@primitive(read_only=True)
def gallons_to_hogsheads(self, gallons: float) -> float:
    """Convert gallons to hogsheads."""
    return gallons / 63

@primitive(read_only=True)
def hogsheads_to_gallons(self, hogsheads: float) -> float:
    """Convert hogsheads to gallons."""
    return hogsheads * 63
```

Run the agent again - it should now successfully convert hogsheads to liters.

## Step 4: Try Multiple Conversions

What happens when we ask the agent to do two conversions at once? Change the query to:

```python
response = agent.run("Convert 3 cups of milk to liters and separately convert 2 beer pints to teaspoons")
```

Run it:

```bash
uv run python unit_converter/main.py
```

You'll notice the agent struggles with this. It may only complete one conversion or get confused.

### Fix: Add a Decomposition Example

The agent needs a decomposition example to understand how to handle multiple conversions in one request.

In `unit_converter/main.py`, find the commented decomposition (around lines 106-122) and uncomment it:

```python
@decomposition(
    intent="Convert 4 tablespoons of honey to milliliters and 2 quarts of juice to cups",
    expanded_intent="Convert 4 tablespoons to cups then to milliliters for honey; convert 2 quarts to pints then to cups for juice. Return both results in a dictionary with labels and units.",
)
def _dual_conversion(self) -> dict:
    # Convert 4 tablespoons of honey to milliliters
    cups_from_tbsp = self.tbsp_to_cups(4)
    honey_ml = self.cups_to_ml(cups_from_tbsp)

    # Convert 2 quarts of juice to cups
    pints_from_quarts = self.quarts_to_pints(2)
    juice_cups = self.pints_to_cups(pints_from_quarts)
    result = {
        "honey": {"value": honey_ml, "unit": "milliliters"},
        "juice": {"value": juice_cups, "unit": "cups"},
    }
    return result
```

Run the agent again:

```bash
uv run python unit_converter/main.py
```

Now the agent understands how to structure multiple conversions in a single request.

## Summary

| Concept | What You Learned |
|---------|------------------|
| **Primitives** | Basic unit conversion functions that the agent can call |
| **Chaining** | The agent automatically chains primitives for multi-step conversions |
| **Decompositions** | Examples that teach the agent how to handle complex, multi-part queries |

By adding primitives, you extend what units the agent can convert. By adding decompositions, you teach the agent new patterns for handling complex requests.
