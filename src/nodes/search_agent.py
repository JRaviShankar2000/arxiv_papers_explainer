from __future__ import annotations

import logging
import re

from arxiv import Client, Search, SortCriterion

from src.state import AgentState, PaperMetadata

logger = logging.getLogger(__name__)

# Shared client instance (respects arXiv's 3-second rate limit by default)
_client = Client(page_size=100, delay_seconds=3.0, num_retries=5)


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract the bare arXiv ID from a full entry-id URL."""
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id


def search_agent(state: AgentState) -> AgentState:
    """Search arXiv for the top 5 most relevant papers matching the query.

    Appends ``PaperMetadata`` entries to ``state.papers``.
    On failure (network error, empty results, etc.) the state is returned
    unchanged and a warning is logged.
    """
    query = state.original_query.strip()
    if not query:
        logger.warning("Empty query — nothing to search for.")
        return state

    search = Search(query=query, max_results=5, sort_by=SortCriterion.Relevance)

    try:
        results = list(_client.results(search))
    except Exception as exc:
        logger.warning("arXiv search failed for query %r: %s", query, exc)
        return state

    if not results:
        logger.warning("No arXiv papers found for query %r.", query)
        return state

    papers: list[PaperMetadata] = []
    for result in results:
        arxiv_id = _extract_arxiv_id(result.entry_id)
        papers.append(
            PaperMetadata(
                arxiv_id=arxiv_id,
                title=(result.title or "").strip(),
                authors=[a.name for a in result.authors] if result.authors else [],
                url=result.entry_id,
                pdf_url=result.pdf_url or "",
                published=result.published,
                summary=(result.summary or "").strip(),
            )
        )

    logger.info("Found %d paper(s) for query %r.", len(papers), query)

    # Print a quick summary so you can decide which papers to explore deeper.
    print(f"\n{'='*60}")
    print(f"📚 Top {len(papers)} papers for: {query}")
    print(f"{'='*60}")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors += " et al."
        short_summary = p.summary[:300]
        if len(p.summary) > 300:
            short_summary += "…"
        print(f"\n  [{i}] {p.title}")
        print(f"      Authors: {authors}")
        print(f"      arXiv: {p.url}")
        print(f"      Summary: {short_summary}")
        print(f"      {'─'*55}")

    state.papers = papers
    return state
