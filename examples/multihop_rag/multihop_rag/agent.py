"""Multihop RAG Agent using GoalSeeking pattern for iterative evidence gathering.

The agent pursues a question through iterative plan-execute-evaluate cycles:
1. Each iteration retrieves evidence from one angle (one "hop")
2. update_context introspects results into structured knowledge
3. The evaluator checks if enough evidence has been gathered
4. The planner sees accumulated context and decides the next hop
5. Repeats until the answer is confident or max iterations reached

This demonstrates:
- Custom GoalContext subclass (MultiHopContext with evidence and queries)
- Introspection boundary (update_context extracts structured insights)
- Static @evaluator (checks evidence sufficiency)
- @decomposition (shows multi-hop retrieval patterns)
- Iterative evidence accumulation toward answering complex questions
"""

from __future__ import annotations

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig, create_llm
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)

from multihop_rag.models import Document, MultiHopContext
from multihop_rag.retriever import ChromaRetriever


class MultiHopRAGAgent(GoalSeeking):
    """Agent that answers multi-hop questions through iterative evidence retrieval.

    Uses the GoalSeeking pattern where each iteration is one retrieval "hop".
    The agent builds up evidence across iterations until it has enough to
    synthesize an answer.

    Example:
        retriever = ChromaRetriever()
        config = LLMConfig(provider=Provider.FIREWORKS, model="...")
        agent = MultiHopRAGAgent(llm=config, retriever=retriever)

        result = agent.seek("Who is the individual linked to crypto found guilty?")
        print(result.final_answer)
        print(f"Iterations: {result.iteration_count}")
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        retriever: ChromaRetriever | None = None,
        generator_llm: LLMConfig | LLM | None = None,
        max_iterations: int = 5,
    ) -> None:
        super().__init__(
            llm=llm,
            name="MultiHopRAGAgent",
            description=(
                "Answers complex multi-hop questions by iteratively retrieving "
                "evidence from a document corpus. Each iteration searches from "
                "a different angle, building up cross-referenced evidence until "
                "the answer can be confidently synthesized."
            ),
            config=GoalSeekingConfig(max_iterations=max_iterations),
        )

        self.retriever = retriever or ChromaRetriever()

        if generator_llm is None:
            if isinstance(llm, LLMConfig):
                self._generator = create_llm(llm)
            else:
                self._generator = llm
        elif isinstance(generator_llm, LLMConfig):
            self._generator = create_llm(generator_llm)
        else:
            self._generator = generator_llm

    def _llm_generate(self, prompt: str) -> str:
        """Generate text using the generator LLM."""
        response = self._generator.generate(prompt)
        return response.text.strip()

    # =========================================================================
    # PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve top-k documents semantically similar to the query.

        Returns documents with content, relevance scores, and metadata.
        Use this to find relevant articles in the corpus.
        """
        return self.retriever.query(query, k=k)

    @primitive(read_only=True)
    def combine_contexts(self, documents: list[Document]) -> str:
        """Combine multiple documents into a single context string.

        Each document is labeled with its source for attribution.
        Use this to create context for evidence extraction.
        """
        if not documents:
            return ""

        parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", f"doc_{i}")
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {title} ({source})]\n{doc.content}")

        return "\n\n---\n\n".join(parts)

    @primitive(read_only=True)
    def extract_evidence(self, context: str, question: str) -> str:
        """Extract relevant factual evidence from retrieved documents.

        Pulls out specific facts, claims, and data points relevant to
        answering the question. Returns a concise summary of evidence found.
        Use this after retrieving documents to distill the key information.
        """
        if not context:
            return "No context provided."

        prompt = f"""Extract the key factual evidence from the following context \
that is relevant to the question.
Focus on specific facts, names, dates, numbers, and claims. Be concise.

Context:
{context}

Question: {question}

Relevant evidence:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def generate_next_query(self, question: str, evidence_so_far: str) -> str:
        """Generate a follow-up search query to find missing evidence.

        Based on the original question and what we already know, determines
        what additional information is needed and generates a focused search query.
        Use this to plan the next retrieval hop.
        """
        prompt = f"""The user asked: {question}

Evidence gathered so far:
{evidence_so_far}

What additional information do we still need to fully answer this question?
Generate a single focused search query to find the missing information.
The query should search from a DIFFERENT angle than what we already have.

Follow-up search query:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def synthesize_answer(self, question: str, evidence: str) -> str:
        """Synthesize a final answer from accumulated multi-source evidence.

        Combines evidence from multiple retrieval hops into a coherent answer.
        Resolves any contradictions and ensures the answer is well-supported.
        Use this when you have gathered enough evidence to answer the question.
        """
        if not evidence:
            return "Insufficient evidence to answer the question."

        prompt = f"""Based on the following evidence gathered from multiple sources, \
answer the question.
Be concise and accurate. Only state what is supported by the evidence.
If the evidence is insufficient, say so.

Evidence:
{evidence}

Question: {question}

Answer:"""

        return self._llm_generate(prompt)

    # =========================================================================
    # DECOMPOSITIONS
    # =========================================================================

    @decomposition(
        intent="Who created Python and what company do they work for now?",
        expanded_intent=(
            "Two-hop query: first retrieve to find the creator, extract "
            "evidence and key facts, then generate a follow-up query for "
            "their current work, retrieve again, and synthesize the answer "
            "from both hops"
        ),
    )
    def _two_hop_inference(self) -> str:
        """Two-hop inference: retrieve -> extract -> followup -> retrieve -> synthesize"""
        docs1 = self.retrieve("Python programming language creator founder", k=5)
        context1 = self.combine_contexts(docs1)
        evidence1 = self.extract_evidence(context1, "Who created Python?")

        next_q = self.generate_next_query(
            "Who created Python and what company do they work for now?",
            evidence1,
        )

        docs2 = self.retrieve(next_q, k=5)
        context2 = self.combine_contexts(docs2)
        evidence2 = self.extract_evidence(context2, next_q)

        all_evidence = f"Hop 1: {evidence1}\n\nHop 2: {evidence2}"
        answer = self.synthesize_answer(
            "Who created Python and what company do they work for now?",
            all_evidence,
        )
        return answer

    @decomposition(
        intent="What was announced at the recent tech conference about AI?",
        expanded_intent=(
            "Single retrieval query: retrieve relevant articles, combine "
            "into context, extract evidence, and synthesize the answer"
        ),
    )
    def _single_retrieval(self) -> str:
        """Single hop: retrieve -> extract -> synthesize"""
        docs = self.retrieve("tech conference AI announcement", k=5)
        context = self.combine_contexts(docs)
        evidence = self.extract_evidence(context, "What was announced about AI?")
        answer = self.synthesize_answer("What was announced about AI?", evidence)
        return answer

    @decomposition(
        intent="How do the economic impacts of solar and wind energy compare?",
        expanded_intent=(
            "Comparison query: retrieve from two different angles, extract "
            "evidence from each, combine both sets of evidence, then "
            "synthesize a comparison answer"
        ),
    )
    def _comparison(self) -> str:
        """Comparison: retrieve(A) -> retrieve(B) -> extract both -> synthesize"""
        docs_a = self.retrieve("solar energy economic impact", k=5)
        context_a = self.combine_contexts(docs_a)
        evidence_a = self.extract_evidence(
            context_a, "What are the economic impacts of solar energy?"
        )

        docs_b = self.retrieve("wind energy economic impact", k=5)
        context_b = self.combine_contexts(docs_b)
        evidence_b = self.extract_evidence(
            context_b, "What are the economic impacts of wind energy?"
        )

        all_evidence = f"Solar: {evidence_a}\n\nWind: {evidence_b}"
        answer = self.synthesize_answer(
            "How do the economic impacts of solar and wind energy compare?",
            all_evidence,
        )
        return answer

    # =========================================================================
    # GOALSEEKING OVERRIDES
    # =========================================================================

    def create_context(self, goal: str) -> MultiHopContext:
        """Create initial context for multi-hop evidence gathering."""
        return MultiHopContext(goal=goal)

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """THE INTROSPECTION BOUNDARY.

        Extracts structured insights from raw execution results into
        the MultiHopContext. The planner and evaluator only see what
        this method writes into context.
        """
        assert isinstance(context, MultiHopContext)

        for step in execution_result.trace.steps:
            if not step.success:
                continue

            name = step.primitive_called
            value = step.result_value

            if name == "retrieve":
                query_arg = step.args.get("query") or step.args.get("arg0")
                if query_arg:
                    context.queries_tried.append(str(query_arg.resolved_value))

            elif name == "extract_evidence" and isinstance(value, str):
                context.evidence.append(value)

            elif name == "synthesize_answer" and isinstance(value, str):
                context.current_answer = value
                context.sufficient = True

    @evaluator
    def check_answer_ready(self, goal: str, context: GoalContext) -> GoalEvaluation:
        """Goal is achieved when we have a synthesized answer."""
        assert isinstance(context, MultiHopContext)
        return GoalEvaluation(
            goal_achieved=(
                context.sufficient and context.current_answer is not None
            )
        )

    def build_goal_prompt(self, goal: str, context: GoalContext) -> str:
        """Build prompt including accumulated evidence for the planner."""
        assert isinstance(context, MultiHopContext)

        primitives = self._get_primitive_methods()
        decompositions = self._get_decomposition_methods()

        primitive_docs = [
            self._format_primitive_signature(name, method)
            for name, method in primitives
        ]

        examples = []
        for _name, method, intent, expanded in decompositions:
            source = self._get_decomposition_source(method)
            if source:
                example = f"Intent: {intent}"
                if expanded:
                    example += f"\nApproach: {expanded}"
                example += f"\nPython:\n{source}"
                examples.append(example)

        # Build accumulated knowledge section
        knowledge_section = ""
        if context.iteration_count > 0:
            hops = context.iteration_count
            knowledge_section = f"\n## Accumulated Knowledge ({hops} hop(s) completed)\n"

            if context.evidence:
                knowledge_section += "\n### Evidence gathered:\n"
                for i, ev in enumerate(context.evidence, 1):
                    preview = ev[:200] + "..." if len(ev) > 200 else ev
                    knowledge_section += f"{i}. {preview}\n"

            if context.queries_tried:
                knowledge_section += (
                    f"\n### Queries already tried: {context.queries_tried}\n"
                )

            if context.current_answer:
                knowledge_section += (
                    f"\n### Current answer: {context.current_answer}\n"
                )

        examples_section = chr(10).join(
            f"### Example {i + 1}{chr(10)}{ex}"
            for i, ex in enumerate(examples)
        ) if examples else "No examples available."

        return f"""You are {self.name}, an agent that answers \
multi-hop questions by iteratively gathering evidence.

{self.description}

## Goal

{goal}

## Available Primitive Methods

You can ONLY call these methods:

```python
{chr(10).join(primitive_docs)}
```

## Example Decompositions

{examples_section}
{knowledge_section}
## Task

Generate Python code for the NEXT step toward answering: {goal}

IMPORTANT:
- Each iteration runs in its OWN scope. Variables from previous iterations are NOT available.
- Use the "Accumulated Knowledge" section above as your source of prior evidence.
- Do NOT repeat queries you have already tried.
- When you have enough evidence, use synthesize_answer to produce the final answer.
- Use combine_contexts before extracting evidence.

## Rules

1. Output ONLY Python assignment statements
2. Each statement must assign a result to a variable
3. You can ONLY call the primitive methods listed above
4. Do NOT use imports, loops, conditionals, or function definitions
5. Do NOT use any dangerous operations (exec, eval, open, etc.)
6. The last assigned variable will be the final result

## Output

```python
"""

    def _extract_final_answer(self, context: GoalContext) -> str | None:
        """Extract the final answer from context."""
        assert isinstance(context, MultiHopContext)
        return context.current_answer
