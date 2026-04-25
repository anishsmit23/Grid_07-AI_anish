"""Utilities for formatting a thread into model-readable context."""

from __future__ import annotations


def format_comment(author: str, content: str) -> str:
    return f"[{author}] {content}"


def build_thread_context(parent_post: str, comment_history: list[dict], human_reply: str) -> str:
    lines = [f"[ORIGINAL POST] {parent_post}"]
    for comment in comment_history:
        author = comment.get("author", "UNKNOWN")
        content = comment.get("content", "")
        lines.append(format_comment(author, content))
    lines.append(f"[NEW HUMAN MESSAGE] {human_reply}")
    return "\n".join(lines)
