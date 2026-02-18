# Deep Research Agent

A deep web research agent built with the **GoalSeeking** blueprint from `opensymbolicai-core`. It answers complex research questions by iteratively searching the web, extracting findings, identifying knowledge gaps, and synthesizing a comprehensive markdown report with citations.

Uses [Tavily](https://tavily.com/) for web search and page extraction (or a built-in mock for offline demos).

## Why GoalSeeking?

A single-shot plan can't anticipate what the web will return. **GoalSeeking** makes each research "hop" adaptive:

```
seek("What are the latest breakthroughs in nuclear fusion energy research?")

  Iteration 1:  decompose_question() → 5 sub-questions
                → update_context: research_plan set, gaps = all 5
                → evaluate: gaps remain → CONTINUE

  Iteration 2:  get_research_state() → gaps[0]
                generate_search_query() → search_web() → extract_findings()
                identify_gaps() → 3 remaining
                → evaluate: depth_score=0.4, gaps=3 → CONTINUE

  ...

  Iteration 6:  identify_gaps() → 0 remaining
                build_outline() → synthesize_report()
                → evaluate: depth=1.0, gaps=0, report=✓ → ACHIEVED

  → Full markdown report with citations
```

The planner sees accumulated knowledge (findings, gaps, depth score) and decides what to research next. The evaluator checks if all gaps are filled and a report exists.

## Architecture

```
User Query
    │
    ▼
DeepResearchAgent.seek(query)
    │
    ├── create_context() → ResearchContext(gaps=[], findings=[], ...)
    │
    └── LOOP (max 8 iterations):
        │
        ├── 1. plan_iteration()     ← LLM sees gaps + findings, plans next hop
        ├── 2. execute()            ← runs primitives (search, extract, identify gaps)
        ├── 3. update_context()     ← INTROSPECTION BOUNDARY: raw → structured insights
        ├── 4. evaluate()           ← @evaluator checks: gaps=0 AND report exists?
        └── 5. should_continue()    ← stop if achieved or max iterations
            │
            ▼
        GoalSeekingResult(report, iterations, status)
```

### The Introspection Boundary

`update_context()` converts raw `ExecutionResult` into structured fields on `ResearchContext`:

| Primitive Called | Context Updated |
|---|---|
| `search_web` | `queries_tried`, `sources` — tracks searches and discovered URLs |
| `read_page` | `sources` — adds full-page source |
| `decompose_question` | `research_plan`, `gaps` — initial sub-questions become gaps |
| `extract_findings` | `findings` — accumulates evidence with source attribution |
| `identify_gaps` | `gaps` — remaining unanswered sub-questions |
| `build_outline` | `outline` — report structure |
| `synthesize_report` | `draft_report`, `sufficient` — final report ready |

The planner and evaluator only see these structured fields — never the raw execution results.

### Primitives

| Primitive | Purpose |
|---|---|
| `get_research_state()` | Returns current gaps, plan, findings count — used by planner to decide next action |
| `get_findings_text()` | Returns all accumulated findings as text — passed to identify_gaps/synthesize |
| `decompose_question(question)` | Breaks research question into 3-6 focused sub-questions |
| `search_web(query, max_results)` | Web search via Tavily (or mock) |
| `read_page(url)` | Extract full page content from a URL |
| `extract_findings(content, question)` | Pull key facts and evidence from content |
| `identify_gaps(question, findings, plan)` | Determine which sub-questions still need research |
| `generate_search_query(gap, prior_findings)` | Create a targeted query to fill a specific gap |
| `build_outline(question, findings)` | Create report outline from findings |
| `write_section(title, findings, question)` | Write one report section with citations |
| `synthesize_report(question, findings, outline)` | Produce the full markdown report |

### Decompositions (few-shot examples)

Four patterns teach the LLM planner how to compose primitives:

1. **Broad topic** — `decompose → search each aspect → extract → identify_gaps → outline → report`
2. **Comparison** — `search side A → search side B → extract both → identify_gaps → outline → report`
3. **Deep dive** — `search → read_page top result → extract → search more → identify_gaps → report`
4. **Gap filling** — `get_research_state → generate_search_query → search → extract → identify_gaps → report`

### Evaluator

The static `@evaluator` checks three conditions:

```python
goal_achieved = (
    depth_score >= 0.8          # ≥80% of sub-questions covered
    and len(gaps) == 0          # no remaining gaps
    and draft_report is not None  # report has been synthesized
)
```

If `max_iterations` is reached without a report, `_extract_final_answer` synthesizes one from whatever findings are available.

## Quick Start

```bash
# Sync workspace dependencies
uv sync --all-packages

# Set up environment
echo "TAVILY_API_KEY=your-key-here" > examples/deep_research/.env
echo "FIREWORKS_API_KEY=your-key-here" >> examples/deep_research/.env

# Run with live Tavily search
uv run python -m deep_research.main -q "What is CRISPR gene editing?"

# Run with mock data (no API keys needed for Tavily)
uv run python -m deep_research.main --mock -q "What is CRISPR gene editing?"

# Demo mode (3 showcase queries)
uv run python -m deep_research.main --mock --demo

# Interactive mode
uv run python -m deep_research.main
```

## CLI Reference

```
--model MODEL        LLM model name (default: accounts/fireworks/models/gpt-oss-120b)
--provider PROVIDER  ollama | openai | anthropic | groq | fireworks (default: fireworks)
--query / -q QUERY   Single query mode
--demo               Run 3 demo queries showcasing different research patterns
--mock               Use pre-downloaded fixture data instead of live Tavily API
--max-iterations N   GoalSeeking max iterations (default: 8)
```

## Mock Mode

The `--mock` flag uses pre-downloaded Tavily search results and page extracts stored in `fixtures.json` (570K). This enables:

- **Offline demos** — no Tavily API key needed
- **Deterministic testing** — same fixture data every run
- **Fast iteration** — no network latency for search calls

The `MockSearcher` matches incoming queries to fixtures by keyword overlap, so the agent's dynamically generated search queries still get reasonable results.

## File Structure

```
deep_research/
├── agent.py          # DeepResearchAgent — primitives, decompositions, GoalSeeking overrides
├── models.py         # Pydantic models: SearchResult, PageContent, Finding, ResearchContext
├── searcher.py       # TavilySearcher — live web search wrapper
├── mock_searcher.py  # MockSearcher — offline fixture-based search
├── fixtures.json     # Pre-downloaded Tavily results for demo queries
├── text.py           # Text utilities (truncation)
└── main.py           # CLI entry point
```

## Comparison with `multihop_rag`

| | `multihop_rag` | `deep_research` |
|---|---|---|
| Data source | Local ChromaDB corpus (609 articles) | Live web via Tavily |
| Scope | Factoid QA (short answers) | Comprehensive reports (long-form markdown) |
| Sub-questions | Implicit (multi-hop reasoning) | Explicit (`decompose_question` → research plan) |
| Gap tracking | Evidence sufficiency only | Structured gap list against research plan |
| Output | Short answer string | Full markdown report with outline, citations, sources |
| Iterations | Typically 2-3 | Typically 5-8 |
