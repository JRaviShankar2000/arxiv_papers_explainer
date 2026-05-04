#!/usr/bin/env python3
"""arXiv Papers Explainer — search, explain, critique, and synthesize arXiv papers.

Usage:
    python main.py "your research query"                    # Full pipeline
    python main.py "query" --paper 2301.12345               # Deep-dive on a paper
    python main.py "query" --skip-critique                  # Skip methodological review
    python main.py "query" --output review.md               # Save output to a file

Environment variables:
    DEEPSEEK_API_KEY   API key for DeepSeek (used for LLM calls).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.graph.graph import build_graph
from src.state import AgentState


def main():
    parser = argparse.ArgumentParser(
        description="arXiv Papers Explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Research question or topic to search for",
    )
    parser.add_argument(
        "--paper",
        default="",
        help="arXiv ID of a specific paper to deep-dive explain",
    )
    parser.add_argument(
        "--skip-critique",
        action="store_true",
        help="Skip the methodological critique step",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Save the final literature review to a file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Set up logging.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # If no query provided, prompt interactively.
    query = args.query.strip()
    if not query:
        query = input("Research topic or question: ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(1)

    # Build initial state.
    state = AgentState(
        original_query=query,
        target_paper_id=args.paper.strip(),
    )

    # Build and run the graph.
    print(f"\n  Starting literature review for: {query}\n")
    graph = build_graph()

    try:
        result = graph.invoke(state)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  Error: {exc}")
        if args.verbose:
            raise
        sys.exit(1)

    # Result is a dict-like state; extract final_draft.
    final_draft = result.get("final_draft", "") if isinstance(result, dict) else getattr(result, "final_draft", "")

    if args.output and final_draft:
        out_path = Path(args.output)
        out_path.write_text(final_draft, encoding="utf-8")
        print(f"\n  Literature review saved to: {out_path.resolve()}")

    if not final_draft:
        print("\n  No literature review was generated. Try a different query.")
        sys.exit(1)


if __name__ == "__main__":
    main()
