from __future__ import annotations

import logging
import os
import re

from arxiv import Client
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Shared arXiv client (respects rate limits by default)
arxiv_client = Client(page_size=100, delay_seconds=3.0, num_retries=5)

# DeepSeek-compatible LLM — API key MUST be set via environment variable
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not _DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY environment variable is not set. "
        "Get a key at https://platform.deepseek.com and set it:\n"
        "  export DEEPSEEK_API_KEY='your-key-here'"
    )
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=_DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)


def extract_arxiv_id(entry_id: str) -> str:
    """Extract the bare arXiv ID from a full entry-id URL."""
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id
