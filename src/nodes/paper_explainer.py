from __future__ import annotations

import logging
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import PyPDF2
from arxiv import Client, Search
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state import AgentState, PaperContent, PaperMetadata

logger = logging.getLogger(__name__)

_client = Client(page_size=10, delay_seconds=3.0, num_retries=5)

# DeepSeek-compatible LLM configuration
_DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-716b53156cb54775a5c11649b99990a6",  # fallback for local dev
)
_LLM = ChatOpenAI(
    model="deepseek-chat",
    api_key=_DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant that explains academic papers so a high "
            "school graduate can understand them. Avoid jargon. Use analogies and "
            "everyday examples. Break down complex ideas step by step. Your goal "
            "is to help someone with no background in the field understand what "
            "the paper is about, how it works, and why it matters.",
        ),
        (
            "human",
            "Explain this research paper in simple words:\n\n{text}",
        ),
    ]
)


def _download_pdf(pdf_url: str, arxiv_id: str) -> Path | None:
    """Download a PDF to a temp file and return the path, or None on failure."""
    try:
        tmp = Path(tempfile.mktemp(suffix=".pdf"))
        urllib.request.urlretrieve(pdf_url, tmp)
        return tmp if tmp.exists() else None
    except Exception as exc:
        logger.warning("Failed to download PDF for %s: %s", arxiv_id, exc)
        return None


def _extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
        return ""

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                pages.append(text)
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", i, exc)

    return "\n\n".join(pages)


def _build_summary_text(paper: PaperMetadata, feedback: str) -> str:
    """Print a formatted simple-language explanation of the paper."""
    lines = [
        f"\n{'='*60}",
        f"📖 Simple Explanation",
        f"{'='*60}",
        f"  Title:  {paper.title}",
        f"  Authors: {', '.join(paper.authors[:4])}{' et al.' if len(paper.authors) > 4 else ''}",
        f"  arXiv:  {paper.url}",
        f"{'='*60}",
        "",
        feedback.strip(),
    ]
    return "\n".join(lines)


def paper_explainer(state: AgentState) -> AgentState:
    """Deep-dive: download a paper, extract its text, and explain it in simple words.

    Looks for ``state.target_paper_id`` in the existing ``state.papers`` list.
    Downloads the PDF, extracts text, and uses an LLM to produce a plain-language
    explanation. Stores the full text in ``state.paper_texts``.
    """
    target_id = state.target_paper_id.strip()
    if not target_id:
        logger.warning("No target_paper_id set — nothing to explain.")
        return state

    # Find paper metadata (if it came from a previous search).
    paper: PaperMetadata | None = next(
        (p for p in state.papers if p.arxiv_id == target_id), None
    )

    if not paper:
        # Fetch metadata from arXiv if not already in state.
        try:
            search = Search(id_list=[target_id])
            result = next(_client.results(search))
            paper = PaperMetadata(
                arxiv_id=_extract_arxiv_id(result.entry_id),
                title=(result.title or "").strip(),
                authors=[a.name for a in result.authors] if result.authors else [],
                url=result.entry_id,
                pdf_url=result.pdf_url or "",
                published=result.published,
                summary=(result.summary or "").strip(),
            )
        except Exception as exc:
            logger.warning("Could not fetch metadata for %s: %s", target_id, exc)
            return state

    # Download the PDF and extract text.
    pdf_url = paper.pdf_url or f"https://arxiv.org/pdf/{target_id}.pdf"
    pdf_path = _download_pdf(pdf_url, target_id)
    if pdf_path is None:
        logger.warning("Skipping explanation — PDF download failed for %s.", target_id)
        return state

    full_text = _extract_text(pdf_path)

    # Clean up the temp file.
    try:
        pdf_path.unlink(missing_ok=True)
    except Exception:
        pass

    if not full_text.strip():
        logger.warning("No text extracted from %s — nothing to explain.", target_id)
        return state

    # Store the extracted text in state.
    state.paper_texts.append(
        PaperContent(arxiv_id=target_id, full_text=full_text)
    )

    # Generate a simple explanation using the LLM.
    # Use only the first ~8000 chars to keep prompt affordable.
    sample = full_text[:8000]
    try:
        chain = _SYSTEM_PROMPT | _LLM
        response = chain.invoke({"text": sample})
        explanation = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("LLM explanation failed for %s: %s", target_id, exc)
        explanation = (
            "Could not generate an automated explanation. "
            f"See the abstract at {paper.url} for a summary."
        )

    # Print the explanation.
    print(_build_summary_text(paper, explanation))

    return state


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract the bare arXiv ID from a full entry-id URL."""
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id
