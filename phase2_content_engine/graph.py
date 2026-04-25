"""Graph wiring for phase 2 content engine."""

from __future__ import annotations

from phase2_content_engine.nodes import decide_search, draft_post, web_search
from phase2_content_engine.schemas import PostOutput


def _run_fallback(bot_persona: dict) -> PostOutput:
    state = {"persona": bot_persona}
    state = decide_search(state)
    state = web_search(state)
    state = draft_post(state)
    return state["post_output"]


def build_graph():
    """Build a LangGraph app if available."""
    try:
        from langgraph.graph import StateGraph
    except Exception:
        return None

    graph = StateGraph(dict)
    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)
    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.set_finish_point("draft_post")
    return graph.compile()


def run_content_engine(bot_persona: dict) -> PostOutput:
    app = build_graph()
    if app is None:
        return _run_fallback(bot_persona)
    result = app.invoke({"persona": bot_persona})
    return result["post_output"]
