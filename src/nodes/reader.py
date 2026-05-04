from __future__ import annotations

import logging
import tempfile
import urllib.request
from pathlib import Path

import PyPDF2

from src.state import AgentState, PaperContent

logger = logging.getLogger(__name__)


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


def reader(state: AgentState) -> AgentState:
    """Download PDFs for all found papers and extract raw text.

    Populates ``state.paper_texts``. Skips papers that fail to download
    or extract — they simply won't appear in the text list.
    """
    if not state.papers:
        logger.warning("No papers in state — nothing to read.")
        return state

    for paper in state.papers:
        # Skip if we already have text for this paper.
        if any(t.arxiv_id == paper.arxiv_id for t in state.paper_texts):
            continue

        pdf_url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        logger.info("Downloading %s ...", paper.arxiv_id)

        pdf_path = _download_pdf(pdf_url, paper.arxiv_id)
        if pdf_path is None:
            continue

        full_text = _extract_text(pdf_path)

        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

        if not full_text.strip():
            logger.warning("No text extracted from %s.", paper.arxiv_id)
            continue

        state.paper_texts.append(
            PaperContent(arxiv_id=paper.arxiv_id, full_text=full_text)
        )
        logger.info("Extracted %d chars from %s.", len(full_text), paper.arxiv_id)

    print(f"\n  Downloaded and extracted text from {len(state.paper_texts)} paper(s).")
    return state
