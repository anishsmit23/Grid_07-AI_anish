"""Tool functions used by phase 2."""

from __future__ import annotations

from data.mock_news import NEWS

try:
    from langchain.tools import tool
except Exception:
    # Fallback no-op decorator when langchain is unavailable.
    def tool(fn):  # type: ignore
        return fn


@tool
def mock_searxng_search(query: str) -> str:
    """Search mock web headlines for a topic query."""
    lowered = query.lower()
    for keyword, headline in NEWS.items():
        if keyword in lowered:
            return headline
    return "No relevant news found today."
