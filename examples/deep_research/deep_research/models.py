"""Data models for the Deep Research agent."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from opensymbolicai.models import GoalContext
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single web search result from Tavily."""

    title: str
    url: str
    content: str  # snippet
    score: float = 0.0
    raw_content: str | None = None  # full page markdown if requested

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"SearchResult(title={self.title!r}, url={self.url!r}, content={preview!r})"


class PageContent(BaseModel):
    """Extracted full-page content from a URL via Tavily extract."""

    url: str
    content: str  # full markdown content

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"PageContent(url={self.url!r}, content={preview!r})"


@runtime_checkable
class Searcher(Protocol):
    """Protocol for web search backends (Tavily, mock, etc.)."""

    def search(
        self,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        search_depth: str = "advanced",
    ) -> list[SearchResult]: ...

    def extract(self, urls: list[str]) -> list[PageContent]: ...


class Source(BaseModel):
    """A cited source URL with title."""

    url: str
    title: str


class Finding(BaseModel):
    """Structured evidence for a sub-question."""

    sub_question: str
    evidence: str
    sources: list[Source] = Field(default_factory=list)


class ResearchContext(GoalContext):
    """Structured context accumulating across research iterations.

    This is the introspection boundary. The planner and evaluator only see
    these fields, never raw execution results. update_context() populates
    these from raw ExecutionResult after each iteration.
    """

    research_plan: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    queries_tried: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    outline: str | None = None
    draft_report: str | None = None
    sufficient: bool = False
    depth_score: float = 0.0
