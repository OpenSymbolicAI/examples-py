"""RAG Agent using traditional tool-calling pattern for comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from opensymbolicai.llm import LLM, LLMConfig, create_llm

from rag_agent.models import Document
from rag_agent.retriever import ChromaRetriever


@dataclass
class LLMUsageMetrics:
    """Track LLM usage across multiple calls."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    planning_calls: int = 0
    execution_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCallResult:
    """Result from the tool-call based agent."""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: LLMUsageMetrics = field(default_factory=LLMUsageMetrics)


class ToolCallRAGAgent:
    """
    RAG agent using traditional tool-calling pattern.

    This agent makes multiple LLM calls in an agentic loop:
    1. LLM decides which tool to call
    2. Tool is executed
    3. Result is fed back to LLM
    4. LLM decides next action or provides final answer

    This contrasts with the behaviour-based approach where:
    1. LLM generates a complete plan in one call
    2. Plan is executed without additional LLM calls for orchestration
    """

    TOOLS = [
        {
            "name": "retrieve",
            "description": "Retrieve documents from the knowledge base matching a query. Returns top-k most relevant documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "extract_answer",
            "description": "Extract an answer from the provided context based on the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The context to extract answer from",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to answer",
                    },
                },
                "required": ["context", "question"],
            },
        },
        {
            "name": "summarize",
            "description": "Summarize the given text into a concise form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize",
                    },
                    "max_sentences": {
                        "type": "integer",
                        "description": "Maximum sentences in summary (default 3)",
                        "default": 3,
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "compare",
            "description": "Compare two topics given their contexts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic1_name": {"type": "string"},
                    "topic1_context": {"type": "string"},
                    "topic2_name": {"type": "string"},
                    "topic2_context": {"type": "string"},
                },
                "required": [
                    "topic1_name",
                    "topic1_context",
                    "topic2_name",
                    "topic2_context",
                ],
            },
        },
        {
            "name": "final_answer",
            "description": "Provide the final answer to the user's question. Use this when you have gathered enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to provide",
                    },
                },
                "required": ["answer"],
            },
        },
    ]

    def __init__(
        self,
        llm: LLMConfig | LLM,
        retriever: ChromaRetriever | None = None,
        max_iterations: int = 10,
    ):
        """Initialize the tool-call agent."""
        if isinstance(llm, LLMConfig):
            self._llm = create_llm(llm)
        else:
            self._llm = llm

        self.retriever = retriever or ChromaRetriever()
        self.max_iterations = max_iterations

        # Map tool names to handlers
        self._tool_handlers: dict[str, Callable[..., str]] = {
            "retrieve": self._handle_retrieve,
            "extract_answer": self._handle_extract_answer,
            "summarize": self._handle_summarize,
            "compare": self._handle_compare,
        }

    def _format_tools_for_prompt(self) -> str:
        """Format tools as text for the prompt."""
        lines = ["Available tools:"]
        for tool in self.TOOLS:
            params = tool["parameters"]["properties"]
            param_list = ", ".join(
                f"{name}: {info.get('type', 'any')}"
                for name, info in params.items()
            )
            lines.append(f"- {tool['name']}({param_list}): {tool['description']}")
        return "\n".join(lines)

    def _llm_generate(self, prompt: str, metrics: LLMUsageMetrics, is_planning: bool = True) -> str:
        """Generate text and track metrics."""
        response = self._llm.generate(prompt)

        metrics.llm_calls += 1
        if is_planning:
            metrics.planning_calls += 1
        else:
            metrics.execution_calls += 1

        # Track tokens if available
        if hasattr(response, "usage") and response.usage:
            metrics.input_tokens += getattr(response.usage, "input_tokens", 0)
            metrics.output_tokens += getattr(response.usage, "output_tokens", 0)

        return response.text.strip()

    def _handle_retrieve(self, query: str, k: int = 5) -> str:
        """Handle retrieve tool call."""
        docs = self.retriever.query(query, k=k)
        if not docs:
            return "No documents found."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("topic", doc.metadata.get("source", f"doc_{i}"))
            # Use full document content - no truncation
            parts.append(f"[{i}. {source}] {doc.content}")

        return "\n\n".join(parts)

    def _handle_extract_answer(
        self, context: str, question: str, metrics: LLMUsageMetrics
    ) -> str:
        """Handle extract_answer tool call (requires LLM)."""
        prompt = f"""Based ONLY on the following context, answer the question.
Be concise and accurate. If the answer is not in the context, say "Information not found."

Context:
{context}

Question: {question}

Answer:"""

        return self._llm_generate(prompt, metrics, is_planning=False)

    def _handle_summarize(
        self, text: str, max_sentences: int, metrics: LLMUsageMetrics
    ) -> str:
        """Handle summarize tool call (requires LLM)."""
        prompt = f"""Summarize the following text in {max_sentences} sentences or fewer.

Text:
{text}

Summary:"""

        return self._llm_generate(prompt, metrics, is_planning=False)

    def _handle_compare(
        self,
        topic1_name: str,
        topic1_context: str,
        topic2_name: str,
        topic2_context: str,
        metrics: LLMUsageMetrics,
    ) -> str:
        """Handle compare tool call (requires LLM)."""
        prompt = f"""Compare {topic1_name} and {topic2_name}.

### {topic1_name}:
{topic1_context}

### {topic2_name}:
{topic2_context}

Comparison:"""

        return self._llm_generate(prompt, metrics, is_planning=False)

    def _parse_tool_call(self, response: str) -> tuple[str | None, dict[str, Any]]:
        """Parse a tool call from LLM response in ACTION: format."""
        import re

        # Look for ACTION: line
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                action_part = line[7:].strip()

                # Parse action name and arguments
                # Format: action_name ARG1="value1" ARG2="value2"
                parts = action_part.split(None, 1)
                if not parts:
                    continue

                tool_name = parts[0].lower()
                args: dict[str, Any] = {}

                if len(parts) > 1:
                    arg_str = parts[1]
                    # Parse KEY="value" or KEY=value patterns
                    pattern = r'(\w+)=(?:"([^"]*)"|(\S+))'
                    matches = re.findall(pattern, arg_str)
                    for key, quoted_val, unquoted_val in matches:
                        value = quoted_val if quoted_val else unquoted_val
                        # Try to convert to int if it looks like one
                        if value.isdigit():
                            value = int(value)
                        args[key.lower()] = value

                return tool_name, args

        # Fallback: check for JSON (for backwards compat)
        try:
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                tool_name = data.get("tool") or data.get("name")
                tool_args = data.get("arguments") or data.get("args") or {}
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                return tool_name, tool_args
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return None, {}

    def run(self, query: str) -> ToolCallResult:
        """Run the agent with tool-calling loop."""
        metrics = LLMUsageMetrics()
        tool_calls: list[dict[str, Any]] = []

        tools_description = self._format_tools_for_prompt()

        # Build conversation history as simple text
        history: list[str] = []

        system_instruction = f"""You are a RAG assistant. You answer questions using ONLY information retrieved
from the knowledge base. You have access to the tools defined below.

═══════════════════════════════════════════════════════════════════════════════
                         ⚠️  CRITICAL INSTRUCTIONS  ⚠️
═══════════════════════════════════════════════════════════════════════════════

1. NEVER hallucinate or make up information. If the retrieved documents don't
   contain the answer, say "I couldn't find information about that."

2. NEVER call extract_answer without first calling retrieve. The context
   parameter MUST contain real document content from a retrieve call.

3. ALWAYS cite your sources. Include document references in your final response.

4. DO NOT call multiple tools in parallel. Wait for each tool result before
   proceeding. Tool calls are sequential.

5. IMPORTANT: The retrieve tool returns document content directly. You must
   pass this content to extract_answer or summarize to process it.

═══════════════════════════════════════════════════════════════════════════════
                              AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════════════

{tools_description}

═══════════════════════════════════════════════════════════════════════════════
                           RESPONSE FORMAT (REQUIRED)
═══════════════════════════════════════════════════════════════════════════════

You MUST respond with exactly ONE action per response using this format:

ACTION: tool_name ARG1="value1" ARG2="value2"

Examples:
ACTION: retrieve QUERY="machine learning definition" K=3
ACTION: extract_answer CONTEXT="[document content here]" QUESTION="What is ML?"
ACTION: summarize TEXT="[content to summarize]" MAX_SENTENCES=3
ACTION: compare TOPIC1_NAME="Python" TOPIC1_CONTEXT="..." TOPIC2_NAME="Rust" TOPIC2_CONTEXT="..."
ACTION: final_answer ANSWER="Your complete answer with citations here"

IMPORTANT: Do not output JSON. Only use the ACTION: format shown above.

═══════════════════════════════════════════════════════════════════════════════
                        QUERY CLASSIFICATION & ROUTING
═══════════════════════════════════════════════════════════════════════════════

Before calling any tools, classify the user's query:

## Type 1: Simple Factual Questions
Keywords: "what is", "define", "who is", "when did"
Strategy: retrieve(k=3) → extract_answer → final_answer
Example: "What is machine learning?"

## Type 2: Complex Technical Questions
Keywords: "explain", "how does", "describe the architecture", technical jargon
Strategy: retrieve(k=5) → extract_answer (may need multiple) → final_answer
Example: "Explain the attention mechanism in transformers"
⚠️ IMPORTANT: For technical questions, you may need to retrieve more documents
and call extract_answer multiple times to get complete information.

## Type 3: Overview/Summary Requests
Keywords: "overview", "summary", "tell me about", "what are the main"
Strategy: retrieve(k=5) → summarize → final_answer
Example: "Give me an overview of recent AI developments"
⚠️ DO NOT use extract_answer for summaries. Use summarize tool instead.

## Type 4: Multi-hop Questions
Keywords: Questions with multiple parts, "and", questions about relationships
Strategy: retrieve → extract_answer → retrieve(follow-up) → extract_answer → final_answer
Example: "Who founded OpenAI and what is their current role?"
⚠️ CRITICAL: Multi-hop requires MULTIPLE retrieve calls. Do not try to answer
both parts from a single retrieval.

## Type 5: Comparison Questions
Keywords: "compare", "difference between", "vs", "better"
Strategy: retrieve(topic A) → retrieve(topic B) → compare → final_answer
Example: "Compare Python and Rust for systems programming"

═══════════════════════════════════════════════════════════════════════════════
                              CHAIN OF THOUGHT
═══════════════════════════════════════════════════════════════════════════════

For each query, think through these steps IN ORDER:

1. CLASSIFY: What type of query is this? (Type 1-5 above)
2. PLAN: What sequence of tool calls do I need?
3. SEARCH TERMS: What keywords should I use? (NOT the raw question)
4. EXECUTE: Call tools one at a time, waiting for results
5. VALIDATE: Do I have enough information? Do I need more retrieval?
6. RESPOND: Formulate answer with final_answer

═══════════════════════════════════════════════════════════════════════════════
                              COMMON MISTAKES
═══════════════════════════════════════════════════════════════════════════════

❌ DON'T: Pass user's raw question to retrieve
✅ DO: Rephrase into search-optimized keywords

❌ DON'T: Use extract_answer for summaries
✅ DO: Use summarize tool for overview requests

❌ DON'T: Answer multi-hop questions in one retrieval
✅ DO: Break into multiple retrieve → extract cycles

❌ DON'T: Provide answer without tool calls
✅ DO: Always retrieve first, even if you think you know the answer

❌ DON'T: Use JSON format for responses
✅ DO: Use ACTION: format as shown above

═══════════════════════════════════════════════════════════════════════════════

Remember: You are a RETRIEVAL assistant. Your knowledge comes from the
knowledge base, not from your training data. When in doubt, retrieve more.
"""

        for iteration in range(self.max_iterations):
            # Build prompt
            if iteration == 0:
                prompt = f"""{system_instruction}

Question: {query}

What action should you take first?"""
            else:
                prompt = f"""{system_instruction}

Question: {query}

Previous steps:
{chr(10).join(history)}

What action should you take next?"""

            # Get LLM response (this is a planning/decision call)
            response = self._llm_generate(prompt, metrics, is_planning=True)

            # Parse tool call
            tool_name, args = self._parse_tool_call(response)

            if tool_name == "final_answer":
                answer = args.get("answer", response)
                tool_calls.append({"tool": "final_answer", "arguments": args})
                return ToolCallResult(answer=answer, tool_calls=tool_calls, metrics=metrics)

            if tool_name and tool_name in self._tool_handlers:
                tool_calls.append({"tool": tool_name, "arguments": args})

                # Execute tool
                if tool_name == "retrieve":
                    result = self._handle_retrieve(
                        query=args.get("query", query),
                        k=args.get("k", 5),
                    )
                elif tool_name == "extract_answer":
                    result = self._handle_extract_answer(
                        context=args.get("context", ""),
                        question=args.get("question", query),
                        metrics=metrics,
                    )
                elif tool_name == "summarize":
                    result = self._handle_summarize(
                        text=args.get("text", ""),
                        max_sentences=args.get("max_sentences", 3),
                        metrics=metrics,
                    )
                elif tool_name == "compare":
                    result = self._handle_compare(
                        topic1_name=args.get("topic1_name", ""),
                        topic1_context=args.get("topic1_context", ""),
                        topic2_name=args.get("topic2_name", ""),
                        topic2_context=args.get("topic2_context", ""),
                        metrics=metrics,
                    )
                else:
                    result = f"Unknown tool: {tool_name}"

                # Add to history for next iteration
                history.append(f"Called {tool_name}({args}):\n{result}")

            else:
                # No valid tool call found, treat response as answer
                return ToolCallResult(answer=response, tool_calls=tool_calls, metrics=metrics)

        # Max iterations reached
        return ToolCallResult(
            answer="Max iterations reached without final answer.",
            tool_calls=tool_calls,
            metrics=metrics,
        )
