from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.state import AgentState
from src.utils import llm

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert academic writer producing a structured literature "
            "review. Write a well-organized synthesis that ties together the "
            "provided papers into a coherent narrative. "

            "Use the following structure:\n\n"
            "## 1. Introduction\n"
            "  - Describe the research area and why it matters.\n"
            "  - State the scope of this review.\n\n"
            "## 2. Key Themes and Findings\n"
            "  - Group papers by common themes or approaches.\n"
            "  - Summarize each paper's contribution in context.\n\n"
            "## 3. Methodological Comparison\n"
            "  - Compare the approaches, datasets, and evaluation methods.\n"
            "  - Highlight strengths and limitations across studies.\n\n"
            "## 4. Gaps and Open Problems\n"
            "  - Identify what the literature is missing.\n"
            "  - Suggest directions for future work.\n\n"
            "## 5. Conclusion\n"
            "  - Recap the state of the field and the most promising directions.\n\n"
            "Write in an academic but accessible style. Cite papers by their "
            "arXiv IDs when referencing them. Aim for 800-1500 words.",
        ),
        (
            "human",
            "Topic: {topic}\n\n"
            "Papers to review:\n\n{papers}\n\n"
            "Methodological critiques:\n\n{critiques}\n\n"
            "Write the literature review draft.",
        ),
    ]
)


def synthesis(state: AgentState) -> AgentState:
    """Synthesize all papers and critiques into a structured literature review.

    Reads from ``state.papers``, ``state.paper_texts``, and ``state.critiques``.
    Writes the result to ``state.final_draft``.
    """
    if not state.papers:
        logger.warning("No papers to synthesize.")
        return state

    # Build a compact summary of each paper.
    paper_blocks = []
    for paper in state.papers:
        text_entry = next(
            (t for t in state.paper_texts if t.arxiv_id == paper.arxiv_id), None
        )
        critique_entry = next(
            (c for c in state.critiques if c.arxiv_id == paper.arxiv_id), None
        )

        authors = ", ".join(paper.authors)
        block = (
            f"  [{paper.arxiv_id}] {paper.title}\n"
            f"    Authors: {authors}\n"
            f"    Abstract: {paper.summary[:600]}\n"
        )
        if text_entry:
            block += f"    Full text excerpt: {text_entry.full_text[:1500]}\n"
        if critique_entry:
            block += (
                f"    Score: {critique_entry.score}/10\n"
                f"    Strengths: {'; '.join(critique_entry.strengths)}\n"
                f"    Weaknesses: {'; '.join(critique_entry.weaknesses)}\n"
                f"    Novelty: {critique_entry.novelty_assessment}\n"
            )
        paper_blocks.append(block)

    logger.info("Synthesizing literature review from %d paper(s) ...", len(state.papers))

    try:
        chain = _SYNTHESIS_PROMPT | llm
        response = chain.invoke(
            {
                "topic": state.original_query,
                "papers": "\n".join(paper_blocks),
                "critiques": "\n".join(
                    f"[{c.arxiv_id}] score={c.score}, "
                    f"novelty={c.novelty_assessment}"
                    for c in state.critiques
                ),
            }
        )
        draft = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("Synthesis LLM call failed: %s", exc)
        draft = "Could not generate literature review due to an error."

    state.final_draft = draft

    print(f"\n{'='*60}")
    print(f"  Literature Review Draft")
    print(f"{'='*60}")
    print(draft)
    print(f"{'='*60}")

    return state
