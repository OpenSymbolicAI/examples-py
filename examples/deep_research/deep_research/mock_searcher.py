"""Mock Tavily searcher that serves pre-downloaded fixture data.

Used with ``--mock`` to run demos without hitting the Tavily API.
Matches queries to fixtures using keyword similarity so the agent's
dynamically generated search queries still get reasonable results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deep_research.models import PageContent, SearchResult

log = logging.getLogger(__name__)

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


def _keyword_overlap(query: str, fixture_key: str) -> int:
    """Score how many words overlap between a query and a fixture key."""
    q_words = set(query.lower().split())
    f_words = set(fixture_key.lower().split())
    return len(q_words & f_words)


class MockSearcher:
    """Drop-in replacement for TavilySearcher using local fixture data.

    Matches incoming queries to the closest pre-downloaded fixture key
    by keyword overlap. If no reasonable match is found, returns the
    fixture with the most overlap (always returns something).
    """

    def __init__(self, fixtures_path: Path | str = FIXTURES_PATH) -> None:
        with open(fixtures_path) as f:
            raw = json.load(f)

        self._extracts: dict[str, str] = raw.pop("__extracts__", {})
        self._fixtures: dict[str, list[dict[str, object]]] = raw

        log.info(
            "[Mock] Loaded %d search fixtures, %d page extracts",
            len(self._fixtures),
            len(self._extracts),
        )

    def _best_match(self, query: str) -> list[dict[str, object]]:
        """Find the fixture key with the highest keyword overlap."""
        best_key = max(self._fixtures, key=lambda k: _keyword_overlap(query, k))
        score = _keyword_overlap(query, best_key)
        log.info("[Mock] query=%r -> fixture=%r (overlap=%d)", query, best_key, score)
        return self._fixtures[best_key]

    def search(
        self,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        search_depth: str = "advanced",
    ) -> list[SearchResult]:
        """Return fixture search results matched by keyword similarity."""
        items = self._best_match(query)
        results = []
        for item in items[:max_results]:
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                )
            )
        return results

    def extract(self, urls: list[str]) -> list[PageContent]:
        """Return fixture page content for known URLs."""
        if not urls:
            return []
        pages = []
        for url in urls:
            content = self._extracts.get(url)
            if content:
                pages.append(PageContent(url=url, content=content))
            else:
                # Try partial URL match
                for fixture_url, fixture_content in self._extracts.items():
                    if url in fixture_url or fixture_url in url:
                        pages.append(PageContent(url=url, content=fixture_content))
                        break
                else:
                    log.warning("[Mock] No extract for URL: %s", url)
                    pages.append(
                        PageContent(url=url, content="(mock) No content available.")
                    )
        return pages
