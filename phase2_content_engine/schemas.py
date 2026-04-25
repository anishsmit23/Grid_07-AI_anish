"""Pydantic schemas for content-engine outputs."""

from pydantic import BaseModel, Field


class PostOutput(BaseModel):
    bot_id: str = Field(..., description="Persona id, e.g. BotA")
    topic: str = Field(..., description="Chosen posting topic")
    post_content: str = Field(..., description="Generated short post content")
