from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.nodes.critique import critique
from src.nodes.paper_explainer import paper_explainer
from src.nodes.reader import reader
from src.nodes.search_agent import search_agent
from src.nodes.synthesis import synthesis
from src.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_search(state: AgentState) -> str:
    """Decide where to go after the search step."""
    if not state.papers:
        return "end"
    if state.target_paper_id.strip():
        return "explainer"
    return "reader"


def _route_after_reader(state: AgentState) -> str:
    """Decide where to go after reading papers."""
    return "critique" if state.paper_texts else "end"


def _route_after_critique(state: AgentState) -> str:
    """Decide where to go after critique."""
    return "synthesis" if state.critiques else "end"


def build_graph() -> StateGraph:
    """Build and compile the arXiv Papers Explainer workflow.

    Pipeline:
        search_agent -> paper_explainer (optional) -> reader -> critique -> synthesis
    """
    graph = StateGraph(AgentState)

    graph.add_node("search_agent", search_agent)
    graph.add_node("paper_explainer", paper_explainer)
    graph.add_node("reader", reader)
    graph.add_node("critique", critique)
    graph.add_node("synthesis", synthesis)

    graph.set_entry_point("search_agent")

    graph.add_conditional_edges(
        "search_agent",
        _route_after_search,
        {
            "explainer": "paper_explainer",
            "reader": "reader",
            "end": END,
        },
    )

    graph.add_edge("paper_explainer", "reader")

    graph.add_conditional_edges(
        "reader",
        _route_after_reader,
        {"critique": "critique", "end": END},
    )

    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"synthesis": "synthesis", "end": END},
    )

    graph.add_edge("synthesis", END)

    return graph.compile()
