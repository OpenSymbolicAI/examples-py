"""Deep Research Agent using GoalSeeking pattern for iterative web research.

The agent conducts deep web research through iterative plan-execute-evaluate cycles:
1. Decomposes the research question into sub-questions
2. Each iteration searches the web and extracts findings for a sub-question
3. update_context introspects results into structured research state
4. The evaluator checks if all gaps are filled and a report is ready
5. Once sufficient, builds an outline and synthesizes a full markdown report

This demonstrates:
- Custom GoalContext subclass (ResearchContext with findings, gaps, and depth tracking)
- Introspection boundary (update_context extracts structured research insights)
- Static @evaluator (checks depth score, gaps, and report existence)
- @decomposition (shows broad, comparison, and deep-dive research patterns)
- Iterative web research accumulation toward a comprehensive report
"""

from __future__ import annotations

import logging

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig, create_llm
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)

from deep_research.models import (
    Finding,
    PageContent,
    ResearchContext,
    Searcher,
    SearchResult,
    Source,
)
from deep_research.searcher import TavilySearcher
from deep_research.text import truncate

log = logging.getLogger(__name__)


class DeepResearchAgent(GoalSeeking):
    """Agent that conducts deep web research and produces structured reports.

    Uses the GoalSeeking pattern where each iteration is one research "hop".
    The agent decomposes the question, iteratively searches and extracts
    findings, identifies gaps, and synthesizes a comprehensive markdown report.

    Example:
        config = LLMConfig(provider=Provider.FIREWORKS, model="...")
        agent = DeepResearchAgent(llm=config)

        result = agent.seek("What are the impacts of AI on healthcare?")
        print(result.final_answer)  # Full markdown report
        print(f"Iterations: {result.iteration_count}")
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        searcher: Searcher | None = None,
        generator_llm: LLMConfig | LLM | None = None,
        max_iterations: int = 8,
    ) -> None:
        super().__init__(
            llm=llm,
            name="DeepResearchAgent",
            description=(
                "Conducts deep web research on a topic by iteratively searching, "
                "reading pages, extracting findings, identifying gaps, and "
                "synthesizing a comprehensive structured markdown report with "
                "citations and sources."
            ),
            config=GoalSeekingConfig(max_iterations=max_iterations),
        )

        self.searcher = searcher or TavilySearcher()
        self._goal_context = ResearchContext(goal="")

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
    def search_web(
        self, query: str, max_results: int = 5, topic: str = "general"
    ) -> list[SearchResult]:
        """Search the web for information on a topic.

        Returns a list of search results with titles, URLs, and content snippets.
        Use this to discover relevant web pages about a topic or sub-question.
        """
        return self.searcher.search(query=query, max_results=max_results, topic=topic)

    @primitive(read_only=True)
    def read_page(self, url: str) -> PageContent:
        """Read the full content of a web page as markdown.

        Use this to get detailed information from a specific URL found in
        search results. Returns the full page content for deep analysis.
        """
        pages = self.searcher.extract(urls=[url])
        if not pages:
            return PageContent(url=url, content="Failed to extract page content.")
        return pages[0]

    @primitive(read_only=True)
    def get_research_state(self) -> dict[str, object]:
        """Return the current research state: gaps, findings summary, and plan.

        Call this at the start of each iteration to decide what to do next.
        Returns a dict with keys: gaps (list[str]), findings_count (int),
        research_plan (list[str]), has_outline (bool), has_report (bool).
        """
        ctx = self._goal_context
        return {
            "gaps": list(ctx.gaps),
            "findings_count": len(ctx.findings),
            "research_plan": list(ctx.research_plan),
            "has_outline": ctx.outline is not None,
            "has_report": ctx.draft_report is not None,
        }

    @primitive(read_only=True)
    def get_findings_text(self) -> str:
        """Return all findings collected so far as a single text block.

        Use this when you need to pass accumulated findings to
        identify_gaps, build_outline, or synthesize_report.
        """
        ctx = self._goal_context
        if not ctx.findings:
            return ""
        return "\n\n".join(
            f"### {f.sub_question}\n{f.evidence}" for f in ctx.findings
        )

    @primitive(read_only=True)
    def decompose_question(self, question: str) -> list[str]:
        """Break a research question into 3-6 focused sub-questions.

        Each sub-question should explore a distinct aspect of the topic.
        Use this at the start of research to create a research plan.
        """
        prompt = f"""Break the following research question into 3-6 focused sub-questions.
Each sub-question should explore a distinct aspect that, together, would provide
a comprehensive understanding of the topic.

Research question: {question}

Return ONLY a numbered list of sub-questions, one per line:
1. ...
2. ...
"""
        text = self._llm_generate(prompt)
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        sub_questions = []
        for line in lines:
            cleaned = line.lstrip("0123456789.)- ").strip()
            if cleaned:
                sub_questions.append(cleaned)
        return sub_questions

    @primitive(read_only=True)
    def extract_findings(self, content: str, question: str) -> str:
        """Extract key facts and evidence from content relevant to a question.

        Pulls out specific facts, data points, quotes, and claims relevant to
        answering the question. Returns a concise summary of findings.
        Use this after reading search results or full pages.
        """
        if not content:
            return "No content provided."

        prompt = f"""Extract the key factual findings from the following content
that are relevant to answering the question. Focus on:
- Specific facts, statistics, and data points
- Expert opinions and quotes
- Key claims with supporting evidence
- Dates, names, and concrete details

Be concise but thorough. Include source attribution where possible.

Content:
{truncate(content, max_tokens=2000)}

Question: {question}

Key findings:"""
        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def identify_gaps(
        self, question: str, findings_so_far: str, research_plan: str
    ) -> list[str]:
        """Identify what sub-questions still need research.

        Given the original question, current findings, and research plan,
        determines which aspects haven't been adequately covered.
        Returns a list of remaining gaps to investigate.
        """
        prompt = f"""Given the following research question and what has been found so far,
identify which sub-questions from the research plan still need investigation.

Research question: {question}

Research plan (sub-questions):
{truncate(research_plan, max_tokens=500)}

Findings so far:
{truncate(findings_so_far, max_tokens=1500)}

List ONLY the sub-questions that have NOT been adequately answered yet.
If all questions are answered, return "NONE".

Remaining gaps:"""
        text = self._llm_generate(prompt)
        if "NONE" in text.upper() and len(text.strip()) < 20:
            return []
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        gaps = []
        for line in lines:
            cleaned = line.lstrip("0123456789.)- ").strip()
            if cleaned and cleaned.upper() != "NONE":
                gaps.append(cleaned)
        return gaps

    @primitive(read_only=True)
    def generate_search_query(self, gap: str, prior_findings: str) -> str:
        """Generate a targeted search query to fill a specific research gap.

        Creates a web search query optimized to find information about the
        gap topic, avoiding what has already been found.
        """
        prompt = f"""Generate a single focused web search query to find information about:

Gap to fill: {gap}

What we already know:
{truncate(prior_findings, max_tokens=500)}

Generate a specific, targeted search query that would find NEW information
not already covered. Return ONLY the search query, nothing else.

Search query:"""
        return self._llm_generate(prompt).strip().strip('"').strip("'")

    @primitive(read_only=True)
    def build_outline(self, question: str, findings: str) -> str:
        """Create a structured outline for the research report.

        Based on the research question and all gathered findings, creates
        a hierarchical outline with section headings for the final report.
        """
        prompt = f"""Create a structured outline for a comprehensive research report.

Research question: {question}

Available findings:
{truncate(findings, max_tokens=1500)}

Create a clear hierarchical outline with:
- An introduction section
- 3-5 main body sections that organize the findings logically
- A conclusion/summary section

Return the outline using markdown heading format:
## Section Title
### Subsection Title
"""
        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def write_section(
        self, section_title: str, relevant_findings: str, question: str
    ) -> str:
        """Write one section of the research report with inline citations.

        Produces well-structured markdown for a single report section,
        with [Source Title](URL) inline citations.
        """
        prompt = f"""Write a detailed section for a research report.

Section title: {section_title}
Overall research question: {question}

Relevant findings and sources for this section:
{truncate(relevant_findings, max_tokens=1000)}

Write the section in markdown format:
- Use clear, professional prose
- Include inline citations as [Source Title](URL) where applicable
- Include specific data, facts, and examples from the findings
- Keep the section focused and well-organized

## {section_title}
"""
        return self._llm_generate(prompt)

    @primitive(read_only=True)
    def synthesize_report(self, question: str, findings: str, outline: str) -> str:
        """Synthesize a complete structured markdown research report.

        Combines all findings according to the outline into a final report
        with introduction, body sections, conclusion, and a sources list.
        """
        if not findings:
            return "Insufficient findings to generate a report."

        prompt = f"""Write a comprehensive, well-structured research report in markdown format.

Research question: {question}

Report outline:
{outline}

All gathered findings:
{truncate(findings, max_tokens=3000)}

Requirements:
- Start with a title (# heading) and brief introduction
- Follow the outline structure with ## and ### headings
- Include inline citations as [Source Title](URL) throughout
- Include specific facts, data, and evidence from the findings
- End with a ## Conclusion section summarizing key takeaways
- End with a ## Sources section listing all referenced URLs as a bulleted list
- Write in clear, professional prose

Report:"""
        return self._llm_generate(prompt)

    # =========================================================================
    # DECOMPOSITIONS
    # =========================================================================

    @decomposition(
        intent="What are the environmental and economic impacts of electric vehicles?",
        expanded_intent=(
            "Decompose the question into sub-questions, search each one, "
            "extract findings, identify remaining gaps, search for gap information, "
            "build an outline, and synthesize the full report"
        ),
    )
    def _broad_topic_research(self) -> str:
        """Broad topic: decompose -> search each -> extract -> gaps -> outline -> report"""
        sub_questions = self.decompose_question(
            "What are the environmental and economic impacts of electric vehicles?"
        )

        results1 = self.search_web(
            query="environmental impact electric vehicles emissions", max_results=5
        )
        context1 = "\n".join([r.content for r in results1])
        findings1 = self.extract_findings(
            context1, "What are the environmental impacts of electric vehicles?"
        )

        results2 = self.search_web(
            query="economic impact electric vehicles cost savings", max_results=5
        )
        context2 = "\n".join([r.content for r in results2])
        findings2 = self.extract_findings(
            context2, "What are the economic impacts of electric vehicles?"
        )

        all_findings = f"Environmental: {findings1}\n\nEconomic: {findings2}"
        plan_text = "\n".join(sub_questions)
        gaps = self.identify_gaps(
            "environmental and economic impacts of EVs", all_findings, plan_text
        )

        if not gaps:
            outline = self.build_outline(
                "environmental and economic impacts of electric vehicles", all_findings
            )
            report = self.synthesize_report(
                "What are the environmental and economic impacts of electric vehicles?",
                all_findings,
                outline,
            )
        else:
            report = all_findings
        return report

    @decomposition(
        intent="How do solar and nuclear energy compare for addressing climate change?",
        expanded_intent=(
            "Search for each side separately, extract findings from each, "
            "build a comparative outline, and synthesize a comparison report"
        ),
    )
    def _comparison_research(self) -> str:
        """Comparison: search A -> extract -> search B -> extract -> outline -> report"""
        results_a = self.search_web(
            query="solar energy climate change benefits limitations", max_results=5
        )
        context_a = "\n".join([r.content for r in results_a])
        findings_a = self.extract_findings(
            context_a, "How does solar energy help address climate change?"
        )

        results_b = self.search_web(
            query="nuclear energy climate change benefits limitations", max_results=5
        )
        context_b = "\n".join([r.content for r in results_b])
        findings_b = self.extract_findings(
            context_b, "How does nuclear energy help address climate change?"
        )

        all_findings = f"Solar energy: {findings_a}\n\nNuclear energy: {findings_b}"
        plan_text = (
            "1. Solar energy benefits and limitations\n"
            "2. Nuclear energy benefits and limitations"
        )
        gaps = self.identify_gaps(
            "solar vs nuclear energy for climate change", all_findings, plan_text
        )

        if not gaps:
            outline = self.build_outline(
                "solar vs nuclear energy for climate change", all_findings
            )
            report = self.synthesize_report(
                "How do solar and nuclear energy compare for addressing climate change?",
                all_findings,
                outline,
            )
        else:
            report = all_findings
        return report

    @decomposition(
        intent="What is the current state of quantum computing research and applications?",
        expanded_intent=(
            "Search broadly, read full pages of top results for deep content, "
            "extract detailed findings, identify gaps, search for remaining gaps, "
            "build outline, and synthesize comprehensive report"
        ),
    )
    def _deep_dive_research(self) -> str:
        """Deep dive: search -> read top pages -> extract -> gaps -> report"""
        results = self.search_web(
            query="quantum computing research breakthroughs", max_results=5
        )

        page = self.read_page(results[0].url)
        findings1 = self.extract_findings(
            page.content, "What is the current state of quantum computing?"
        )

        results2 = self.search_web(
            query="quantum computing practical applications industry", max_results=5
        )
        context2 = "\n".join([r.content for r in results2])
        findings2 = self.extract_findings(
            context2, "What are practical applications of quantum computing?"
        )

        all_findings = f"Research state: {findings1}\n\nApplications: {findings2}"
        plan_text = "1. Current research breakthroughs\n2. Practical applications in industry"
        gaps = self.identify_gaps(
            "quantum computing research and applications", all_findings, plan_text
        )

        if not gaps:
            outline = self.build_outline(
                "quantum computing research and applications", all_findings
            )
            report = self.synthesize_report(
                "current state of quantum computing research and applications",
                all_findings,
                outline,
            )
        else:
            report = all_findings
        return report

    @decomposition(
        intent="Fill remaining research gaps and generate the final report",
        expanded_intent=(
            "Get the current state, then if gaps remain generate a targeted "
            "search query for the first gap, search and extract findings, "
            "then identify remaining gaps. When all gaps are filled, "
            "build an outline and synthesize the report."
        ),
    )
    def _gap_filling_and_reporting(self) -> str:
        """Gap filling: state -> query -> search -> extract -> gaps -> report"""
        state = self.get_research_state()
        all_findings = self.get_findings_text()

        if state["gaps"]:
            current_gap = state["gaps"][0]
            query = self.generate_search_query(current_gap, all_findings)
            results = self.search_web(query=query, max_results=5)
            context_str = "\n".join([r.content for r in results])
            self.extract_findings(context_str, current_gap)

            all_findings = self.get_findings_text()
            plan_text = "\n".join(state["research_plan"])
            remaining = self.identify_gaps(
                "comprehensive analysis", all_findings, plan_text
            )
        else:
            remaining = []

        if not remaining:
            all_findings = self.get_findings_text()
            outline = self.build_outline("comprehensive analysis", all_findings)
            report = self.synthesize_report(
                "comprehensive analysis", all_findings, outline
            )
        else:
            report = all_findings
        return report

    # =========================================================================
    # GOALSEEKING OVERRIDES
    # =========================================================================

    def create_context(self, goal: str) -> ResearchContext:
        """Create initial context for deep research."""
        ctx = ResearchContext(goal=goal)
        self._goal_context = ctx
        return ctx

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """THE INTROSPECTION BOUNDARY.

        Extracts structured insights from raw execution results into
        the ResearchContext. The planner and evaluator only see what
        this method writes into context.
        """
        assert isinstance(context, ResearchContext)

        # Track most recent sources from search/read so we can attach to findings
        recent_sources: list[Source] = []

        for step in execution_result.trace.steps:
            if not step.success:
                continue

            name = step.primitive_called
            value = step.result_value

            if name == "search_web":
                query_arg = step.args.get("query") or step.args.get("arg0")
                if query_arg:
                    q = str(query_arg.resolved_value)
                    context.queries_tried.append(q)
                    log.info("[Tavily] query: %s", q)
                recent_sources = []
                if isinstance(value, list):
                    for result in value:
                        if hasattr(result, "url") and hasattr(result, "title"):
                            source = Source(url=result.url, title=result.title)
                            recent_sources.append(source)
                            if not any(s.url == source.url for s in context.sources):
                                context.sources.append(source)

            elif name == "read_page":
                url_arg = step.args.get("url") or step.args.get("arg0")
                if url_arg and isinstance(value, PageContent):
                    # Try to resolve title from existing sources
                    url_str = str(url_arg.resolved_value)
                    existing = next(
                        (s for s in context.sources if s.url == url_str), None
                    )
                    title = existing.title if existing else ""
                    source = Source(url=url_str, title=title)
                    recent_sources = [source]
                    if not existing:
                        context.sources.append(source)

            elif name == "decompose_question" and isinstance(value, list):
                context.research_plan = [str(q) for q in value]
                context.gaps = list(context.research_plan)
                log.info("[Plan] %d sub-questions: %s", len(context.gaps), context.gaps)

            elif name == "extract_findings" and isinstance(value, str):
                question_arg = step.args.get("question") or step.args.get("arg1")
                sub_q = str(question_arg.resolved_value) if question_arg else "unknown"
                finding = Finding(
                    sub_question=sub_q,
                    evidence=value,
                    sources=list(recent_sources),
                )
                context.findings.append(finding)

            elif name == "identify_gaps" and isinstance(value, list):
                context.gaps = [str(g) for g in value]
                if context.gaps:
                    log.info("[Gaps] %d remaining: %s", len(context.gaps), context.gaps)
                else:
                    log.info("[Gaps] all filled")

            elif name == "build_outline" and isinstance(value, str):
                context.outline = value

            elif name == "synthesize_report" and isinstance(value, str):
                context.draft_report = value
                context.sufficient = True

        # Update depth_score: proportion of research plan covered
        if context.research_plan:
            covered = len(context.research_plan) - len(context.gaps)
            context.depth_score = min(covered / len(context.research_plan), 1.0)
        elif context.findings:
            context.depth_score = min(len(context.findings) / 3.0, 1.0)

    @evaluator
    def check_research_complete(
        self, goal: str, context: GoalContext
    ) -> GoalEvaluation:
        """Goal is achieved when we have a comprehensive report."""
        assert isinstance(context, ResearchContext)
        return GoalEvaluation(
            goal_achieved=(
                context.depth_score >= 0.8
                and len(context.gaps) == 0
                and context.draft_report is not None
            )
        )

    def _extract_final_answer(self, context: GoalContext) -> str | None:
        """Extract the final answer (the report) from context.

        If max_iterations was reached without a draft report, synthesize
        one from whatever findings we have so far.
        """
        assert isinstance(context, ResearchContext)
        if context.draft_report:
            return context.draft_report

        # Fallback: synthesize a report from available findings
        if context.findings:
            log.info(
                "[Report] No draft report — synthesizing from %d findings",
                len(context.findings),
            )
            findings_text = "\n\n".join(
                f"### {f.sub_question}\n{f.evidence}" for f in context.findings
            )
            outline = self.build_outline(context.goal, findings_text)
            return self.synthesize_report(context.goal, findings_text, outline)

        return None
