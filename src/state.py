from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for a single paper fetched from arXiv."""

    arxiv_id: str = Field(..., description="arXiv identifier, e.g. 2301.12345")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    url: str = Field(default="", description="arXiv abstract URL")
    pdf_url: str = Field(default="", description="Direct PDF download URL")
    published: Optional[datetime] = Field(default=None, description="Publication date")
    summary: str = Field(default="", description="arXiv abstract / summary")


class PaperContent(BaseModel):
    """Raw text extracted from a downloaded paper PDF."""

    arxiv_id: str
    full_text: str = Field(default="", description="Full plain-text content of the paper")


class Critique(BaseModel):
    """Methodological critique of a single paper."""

    arxiv_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    methodology_notes: str = Field(default="")
    novelty_assessment: str = Field(default="")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=10.0, description="Overall quality score (0-10)"
    )


class AgentState(BaseModel):
    """Top-level state carried through the LangGraph workflow."""

    # ---- Input ----
    original_query: str = Field(
        default="", description="The user's research question or topic"
    )

    # ---- Search phase ----
    papers: list[PaperMetadata] = Field(
        default_factory=list, description="Papers discovered by the search agent"
    )

    # ---- Reading phase ----
    paper_texts: list[PaperContent] = Field(
        default_factory=list, description="Raw text extracted from each downloaded PDF"
    )

    # ---- Critique phase ----
    critiques: list[Critique] = Field(
        default_factory=list, description="Methodological critiques of each paper"
    )

    # ---- Synthesis phase ----
    final_draft: str = Field(
        default="", description="The final synthesized literature review draft"
    )
