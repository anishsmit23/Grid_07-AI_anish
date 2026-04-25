"""LangGraph node functions for content generation."""

from __future__ import annotations

from phase2_content_engine.schemas import PostOutput
from phase2_content_engine.tools import mock_searxng_search


def _choose_topic(persona: dict) -> str:
    text = f"{persona.get('name', '')} {persona.get('description', '')}".lower()
    if "crypto" in text:
        return "crypto markets"
    if "privacy" in text or "civil liberties" in text:
        return "privacy regulation"
    if "policy" in text or "governance" in text:
        return "ai regulation"
    return "ai innovation"


def decide_search(state: dict) -> dict:
    persona = state["persona"]
    topic = _choose_topic(persona)
    state["topic"] = topic
    state["search_query"] = f"latest news about {topic}"
    return state


def web_search(state: dict) -> dict:
    state["search_results"] = mock_searxng_search.invoke(state["search_query"])
    return state


def draft_post(state: dict) -> dict:
    persona = state["persona"]
    headline = state.get("search_results", "")
    topic = state.get("topic", "technology")
    post_content = (
        f"{persona['name']}: {headline} This is exactly why we need sharper "
        f"thinking on {topic} right now."
    )
    post_content = post_content[:280]
    state["post_output"] = PostOutput(
        bot_id=persona["id"],
        topic=topic,
        post_content=post_content,
    )
    return state
