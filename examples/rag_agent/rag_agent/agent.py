"""RAG Agent with behavior-based decomposition for adaptive retrieval strategies."""

from __future__ import annotations

from typing import Any

from opensymbolicai.blueprints import PlanExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.llm import LLM, LLMConfig, create_llm

from rag_agent.models import Document, ValidationResult
from rag_agent.retriever import ChromaRetriever


class RAGAgent(PlanExecute):
    """
    An adaptive RAG agent that uses behavior-based decomposition
    to select appropriate retrieval strategies based on query type.

    The agent learns from decomposition examples to automatically select:
    - Simple QA: retrieve -> extract (for factual questions)
    - Reranked QA: retrieve -> rerank -> extract (for complex questions)
    - Summarization: retrieve -> summarize (for overview requests)
    - Multi-hop: retrieve -> extract -> followup -> retrieve -> aggregate
    - Comparison: retrieve(A) -> retrieve(B) -> compare
    - Validated: retrieve -> extract -> validate (for accuracy-critical queries)

    Example:
        retriever = ChromaRetriever()
        config = LLMConfig(provider=Provider.OLLAMA, model="qwen3:8b")
        agent = RAGAgent(llm=config, retriever=retriever)

        result = agent.run("What is machine learning?")
        print(result.result)  # The answer
        print(result.plan)    # The generated plan showing which strategy was used
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        retriever: ChromaRetriever | None = None,
        generator_llm: LLMConfig | LLM | None = None,
    ):
        """
        Initialize the RAG agent.

        Args:
            llm: LLM configuration for planning (decomposition).
            retriever: ChromaRetriever instance for document retrieval.
                      If None, creates a new one with default settings.
            generator_llm: Optional separate LLM for answer generation.
                          If None, uses the same LLM as planning.
        """
        super().__init__(llm=llm)

        self.retriever = retriever or ChromaRetriever()

        # Set up generator LLM
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
    # RETRIEVAL PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve top-k documents semantically similar to the query.
        Returns documents with content, relevance scores, and metadata.
        Use this for finding relevant information in the knowledge base.
        """
        return self.retriever.query(query, k=k)

    @primitive(read_only=True)
    def retrieve_filtered(
        self,
        query: str,
        source: str | None = None,
        topic: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        """
        Retrieve documents matching query AND metadata filters.
        Use 'source' to filter by data source (e.g., 'wikipedia').
        Use 'topic' to filter by topic/category.
        """
        filters: dict[str, Any] = {}
        if source:
            filters["source"] = source
        if topic:
            filters["topic"] = topic

        return self.retriever.query(query, k=k, filters=filters if filters else None)

    @primitive(read_only=True)
    def rerank(self, documents: list[Document], query: str, k: int = 3) -> list[Document]:
        """
        Rerank documents by relevance to the query using LLM scoring.
        Returns top-k most relevant documents after reranking.
        Use this when initial retrieval may include less relevant results.
        """
        if not documents:
            return []

        scored: list[tuple[Document, float]] = []
        for doc in documents:
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
Only output a single number.

Query: {query}
Document: {doc.content[:500]}

Relevance score (0-10):"""

            try:
                response = self._llm_generate(prompt)
                # Extract first number from response
                score_str = "".join(c for c in response.split()[0] if c.isdigit() or c == ".")
                score = float(score_str) if score_str else doc.score * 10
            except (ValueError, IndexError):
                score = doc.score * 10  # Fallback to original score

            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]

    # =========================================================================
    # PROCESSING PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def combine_contexts(self, documents: list[Document]) -> str:
        """
        Combine multiple documents into a single context string.
        Each document is labeled with its source for attribution.
        Use this to create context for answer extraction.
        """
        if not documents:
            return ""

        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("topic", doc.metadata.get("source", f"doc_{i}"))
            parts.append(f"[Source {i}: {source}]\n{doc.content}")

        return "\n\n---\n\n".join(parts)

    @primitive(read_only=True)
    def extract_answer(self, context: str, question: str) -> str:
        """
        Extract a direct answer to the question from the given context.
        Returns the answer or 'Information not found in the provided context.'
        Use this after retrieving and combining relevant documents.
        """
        if not context:
            return "No context provided to extract answer from."

        prompt = f"""Based ONLY on the following context, answer the question.
Be concise and accurate. If the answer is not in the context, say "Information not found in the provided context."

Context:
{context}

Question: {question}

Answer:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Summarize text into a concise form with at most max_sentences sentences.
        Preserves key information while being brief.
        Use this for creating overviews of retrieved content.
        """
        if not text:
            return ""

        prompt = f"""Summarize the following text in {max_sentences} sentences or fewer.
Keep the most important information and be accurate.

Text:
{text}

Summary:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def generate_followup_query(self, original_question: str, partial_answer: str) -> str:
        """
        Generate a follow-up search query to find missing information.
        Use this for multi-hop reasoning when initial retrieval is incomplete.
        """
        prompt = f"""The user asked: {original_question}
We found: {partial_answer}

What additional information do we need? Generate a single focused search query to find it.

Follow-up search query:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def aggregate_answers(self, answers: list[str], question: str) -> str:
        """
        Combine multiple partial answers into a coherent final answer.
        Resolves any contradictions and synthesizes the information.
        Use this after multi-hop retrieval to create a complete answer.
        """
        if not answers:
            return "No answers to aggregate."

        combined = "\n".join(f"- {a}" for a in answers if a)

        prompt = f"""Combine these pieces of information into a single coherent answer to the question.

Question: {question}

Information found:
{combined}

Complete answer:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def compare_topics(
        self,
        topic1_context: str,
        topic2_context: str,
        topic1_name: str,
        topic2_name: str,
        aspects: list[str],
    ) -> str:
        """
        Compare two topics across specified aspects using their contexts.
        Returns a structured comparison highlighting similarities and differences.
        """
        aspects_str = ", ".join(aspects)

        prompt = f"""Compare {topic1_name} and {topic2_name} on these aspects: {aspects_str}

### {topic1_name}:
{topic1_context}

### {topic2_name}:
{topic2_context}

Provide a balanced comparison covering each aspect:"""

        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def validate_answer(self, answer: str, context: str) -> ValidationResult:
        """
        Validate that the answer is fully supported by the context.
        Returns validation result with confidence score and any issues.
        Use this for accuracy-critical queries.
        """
        prompt = f"""Is this answer fully supported by the given context?

Answer: {answer}

Context: {context}

Respond in this exact format:
SUPPORTED: [yes/no]
CONFIDENCE: [0-100]
ISSUES: [list any unsupported claims, or "none"]"""

        response = self._llm_generate(prompt)

        # Parse the response
        is_valid = "supported: yes" in response.lower()

        confidence = 0.5
        for line in response.split("\n"):
            if "confidence:" in line.lower():
                try:
                    conf_str = line.split(":")[1].strip().replace("%", "")
                    confidence = float(conf_str) / 100
                except (ValueError, IndexError):
                    pass

        issues: list[str] = []
        for line in response.split("\n"):
            if "issues:" in line.lower():
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.lower() not in ["none", "n/a", ""]:
                    issues = [i.strip() for i in issues_text.split(",") if i.strip()]

        return ValidationResult(is_valid=is_valid, confidence=confidence, issues=issues)

    # =========================================================================
    # DECOMPOSITION BEHAVIORS - Teaching RAG Strategies
    # =========================================================================

    @decomposition(
        intent="What is machine learning?",
        expanded_intent="Simple factual query: retrieve relevant documents, combine into context, extract answer directly",
    )
    def _simple_qa(self) -> str:
        """Basic RAG: retrieve -> combine -> extract"""
        docs = self.retrieve("machine learning definition basics", k=3)
        context = self.combine_contexts(docs)
        answer = self.extract_answer(context, "What is machine learning?")
        return answer

    @decomposition(
        intent="Explain the architectural innovations in transformer models",
        expanded_intent="Complex technical query needing high relevance: retrieve many documents, rerank for best matches, then extract detailed answer",
    )
    def _reranked_qa(self) -> str:
        """Reranked RAG: retrieve(many) -> rerank -> combine -> extract"""
        docs = self.retrieve("transformer architecture innovations attention mechanism", k=8)
        top_docs = self.rerank(docs, "transformer model architectural innovations", k=3)
        context = self.combine_contexts(top_docs)
        answer = self.extract_answer(
            context, "What are the architectural innovations in transformer models?"
        )
        return answer

    @decomposition(
        intent="Give me an overview of recent AI developments",
        expanded_intent="Summarization request: retrieve relevant documents, combine context, then summarize into a brief overview",
    )
    def _summarization(self) -> str:
        """Summarization RAG: retrieve -> combine -> summarize"""
        docs = self.retrieve("artificial intelligence recent developments progress", k=5)
        context = self.combine_contexts(docs)
        summary = self.summarize(context, max_sentences=4)
        return summary

    @decomposition(
        intent="Who created Python and what company does the creator work for now?",
        expanded_intent="Multi-hop query requiring chained reasoning: first retrieve to find the creator, then generate follow-up query about their current work, retrieve again, and aggregate both answers",
    )
    def _multi_hop_qa(self) -> str:
        """Multi-hop RAG: retrieve -> extract -> followup -> retrieve -> aggregate"""
        # First hop: find the creator
        docs1 = self.retrieve("Python programming language creator founder", k=3)
        context1 = self.combine_contexts(docs1)
        creator_info = self.extract_answer(context1, "Who created Python?")

        # Generate follow-up query
        followup = self.generate_followup_query(
            "Who created Python and what company does the creator work for now?",
            creator_info,
        )

        # Second hop: find current work
        docs2 = self.retrieve(followup, k=3)
        context2 = self.combine_contexts(docs2)
        work_info = self.extract_answer(context2, followup)

        # Aggregate answers
        final = self.aggregate_answers(
            [creator_info, work_info],
            "Who created Python and what company does the creator work for now?",
        )
        return final

    @decomposition(
        intent="Compare Python and Rust for systems programming",
        expanded_intent="Comparison query: retrieve context about each topic separately, then compare them across relevant aspects like performance, safety, and ease of use",
    )
    def _comparison_qa(self) -> str:
        """Comparison RAG: retrieve(topic1) -> retrieve(topic2) -> compare"""
        python_docs = self.retrieve("Python programming systems performance features", k=3)
        rust_docs = self.retrieve("Rust programming systems performance features", k=3)

        python_context = self.combine_contexts(python_docs)
        rust_context = self.combine_contexts(rust_docs)

        comparison = self.compare_topics(
            topic1_context=python_context,
            topic2_context=rust_context,
            topic1_name="Python",
            topic2_name="Rust",
            aspects=["performance", "memory safety", "ease of learning", "ecosystem"],
        )
        return comparison

    @decomposition(
        intent="What are the health benefits of green tea? Make sure the answer is accurate.",
        expanded_intent="Accuracy-critical query: retrieve documents, extract answer, then validate the answer against the sources to ensure it's fully supported",
    )
    def _validated_qa(self) -> str:
        """Validated RAG: retrieve -> extract -> validate"""
        docs = self.retrieve("green tea health benefits scientific studies", k=5)
        context = self.combine_contexts(docs)
        answer = self.extract_answer(context, "What are the health benefits of green tea?")

        validation = self.validate_answer(answer, context)

        if validation.is_valid and validation.confidence > 0.7:
            return answer
        else:
            # If validation fails, provide a more conservative answer
            conservative = self.summarize(context, max_sentences=3)
            return f"Based on available sources: {conservative}"

    @decomposition(
        intent="Tell me about quantum computing from Wikipedia sources only",
        expanded_intent="Filtered retrieval: use retrieve_filtered with source filter to get only Wikipedia content, then extract answer from filtered results",
    )
    def _filtered_qa(self) -> str:
        """Filtered RAG: retrieve_filtered -> combine -> extract"""
        docs = self.retrieve_filtered(
            query="quantum computing overview",
            source="wikipedia",
            k=4,
        )
        context = self.combine_contexts(docs)
        answer = self.extract_answer(context, "What is quantum computing?")
        return answer
