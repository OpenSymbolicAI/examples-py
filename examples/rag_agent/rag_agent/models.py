"""Data models for the RAG agent."""

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class ValidationResult:
    """Result of answer validation against sources."""

    is_valid: bool
    confidence: float
    issues: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, confidence={self.confidence:.2f})"
