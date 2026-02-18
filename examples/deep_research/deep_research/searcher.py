"""Tavily web search wrapper for the Deep Research agent."""

from __future__ import annotations

import os

from deep_research.models import PageContent, SearchResult


class TavilySearcher:
    """Tavily-based web searcher.

    Wraps the Tavily Python SDK for web search and page extraction.
    Requires TAVILY_API_KEY environment variable.
    """

    def __init__(self) -> None:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            msg = (
                "TAVILY_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )
            raise RuntimeError(msg)

        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        search_depth: str = "advanced",
    ) -> list[SearchResult]:
        """Search the web using Tavily.

        Args:
            query: Search query string.
            max_results: Number of results (1-20).
            topic: Search topic ("general", "news", "finance").
            search_depth: "basic" (1 credit) or "advanced" (2 credits).

        Returns:
            List of SearchResult objects.
        """
        response = self.client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_raw_content=False,
        )

        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                )
            )
        return results

    def extract(self, urls: list[str]) -> list[PageContent]:
        """Extract full page content from URLs using Tavily.

        Args:
            urls: List of URLs to extract (1-20).

        Returns:
            List of PageContent objects for successfully extracted pages.
        """
        if not urls:
            return []

        response = self.client.extract(
            urls=urls[:20],
            format="markdown",
        )

        pages = []
        for item in response.get("results", []):
            raw = item.get("raw_content", "")
            if raw:
                pages.append(
                    PageContent(
                        url=item.get("url", ""),
                        content=raw,
                    )
                )
        return pages
