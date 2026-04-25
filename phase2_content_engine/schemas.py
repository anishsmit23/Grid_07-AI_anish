"""Pydantic schemas for content-engine outputs."""

from pydantic import BaseModel, Field


class SearchDecision(BaseModel):
    topic: str = Field(..., description="Topic selected by the persona")
    search_query: str = Field(..., description="4-7 word search query")


class PostOutput(BaseModel):
    bot_id: str = Field(..., description="Persona id, e.g. BotA")
    topic: str = Field(..., description="Chosen posting topic")
    post_content: str = Field(..., description="Generated short post content")
