from __future__ import annotations

import logging
import os
import re

from arxiv import Client
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Shared arXiv client (respects rate limits by default)
arxiv_client = Client(page_size=100, delay_seconds=3.0, num_retries=5)

# DeepSeek-compatible LLM
_LLM_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "",
)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=_LLM_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)


def extract_arxiv_id(entry_id: str) -> str:
    """Extract the bare arXiv ID from a full entry-id URL."""
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id
