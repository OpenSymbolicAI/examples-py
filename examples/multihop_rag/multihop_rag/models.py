"""Data models for the Multihop RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opensymbolicai.models import GoalContext
from pydantic import BaseModel, Field


@dataclass
class Document:
    """A retrieved document with content and metadata."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id!r}, score={self.score:.3f}, content={preview!r})"


class MultiHopContext(GoalContext):
    """Structured context accumulating across hops -- the introspection boundary.

    The planner and evaluator only see these fields, never raw execution results.
    update_context() populates these from raw ExecutionResult after each iteration.
    """

    evidence: list[str] = Field(default_factory=list)
    queries_tried: list[str] = Field(default_factory=list)
    current_answer: str | None = None
    sufficient: bool = False


class QueryItem(BaseModel):
    """A query from the MultiHop-RAG dataset."""

    query: str
    answer: str
    question_type: str
    evidence_list: list[dict[str, Any]] = Field(default_factory=list)
