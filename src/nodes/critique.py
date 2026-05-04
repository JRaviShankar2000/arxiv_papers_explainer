from __future__ import annotations

import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from src.state import AgentState, Critique
from src.utils import llm

logger = logging.getLogger(__name__)

_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior research methodologist reviewing an academic paper. "
            "Evaluate the paper honestly and concisely. Identify what the paper "
            "does well, where it falls short, and assess its novelty. "
            "Return your evaluation as a JSON object with exactly these keys: "
            "strengths (array of strings), weaknesses (array of strings), "
            "methodology_notes (string), novelty_assessment (string), "
            "score (float between 0 and 10). "
            "Keep methodology_notes under 200 words and novelty_assessment under 100 words. "
            "Return ONLY the JSON object, no other text.",
        ),
        (
            "human",
            "Paper title: {title}\n"
            "Authors: {authors}\n"
            "Abstract: {abstract}\n\n"
            "Full text (first 6000 chars):\n{text}",
        ),
    ]
)


def critique(state: AgentState) -> AgentState:
    """Critique the methodology of each downloaded paper using an LLM.

    Reads from ``state.papers`` and ``state.paper_texts``, populates
    ``state.critiques``. Papers without extracted text are skipped.
    """
    if not state.paper_texts:
        logger.warning("No paper texts available — nothing to critique.")
        return state

    for text_entry in state.paper_texts:
        if any(c.arxiv_id == text_entry.arxiv_id for c in state.critiques):
            continue

        paper = next(
            (p for p in state.papers if p.arxiv_id == text_entry.arxiv_id), None
        )
        title = paper.title if paper else text_entry.arxiv_id
        authors = ", ".join(paper.authors) if paper else "Unknown"
        abstract = paper.summary if paper else ""

        logger.info("Critiquing %s ...", title[:80])

        sample = text_entry.full_text[:6000]

        try:
            chain = _CRITIQUE_PROMPT | llm
            response = chain.invoke(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "text": sample,
                }
            )
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("LLM critique failed for %s: %s", text_entry.arxiv_id, exc)
            continue

        # Parse the JSON response.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse critique JSON for %s.", text_entry.arxiv_id)
            continue

        critique_entry = Critique(
            arxiv_id=text_entry.arxiv_id,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            methodology_notes=data.get("methodology_notes", ""),
            novelty_assessment=data.get("novelty_assessment", ""),
            score=data.get("score"),
        )
        state.critiques.append(critique_entry)

        # Print a quick summary.
        print(f"\n  Critique for: {title[:100]}")
        print(f"  Score: {critique_entry.score}/10")
        print(f"  Strengths: {', '.join(critique_entry.strengths[:3])}")
        print(f"  Weaknesses: {', '.join(critique_entry.weaknesses[:3])}")

    logger.info("Completed %d critique(s).", len(state.critiques))
    return state
