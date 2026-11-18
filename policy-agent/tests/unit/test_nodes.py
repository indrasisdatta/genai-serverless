"""Unit tests for graph nodes."""

import pytest
from unittest.mock import Mock, MagicMock

from app.graph.nodes.router import router
from app.graph.nodes.decompose import decompose
from app.graph.nodes.grade_docs import grade_docs
from app.graph.nodes.rewrite_query import rewrite_query
from app.graph.nodes.generate import generate
from app.graph.nodes.groundedness import groundedness
from app.graph.nodes.abstain import abstain
from app.graph.nodes.refuse import refuse


def test_router_returns_route():
    """Test that router returns a valid route."""
    state = {"question": "What is the late fee?"}
    # Note: requires LLM mocking in real tests
    result = router(state)
    assert "route" in result
    assert result["route"] in ["simple", "multi_hop", "out_of_scope"]


def test_abstain_sets_status():
    """Test that abstain node sets status to abstained."""
    state = {"docs": []}
    result = abstain(state)
    assert result["status"] == "abstained"
    assert "answer" in result


def test_refuse_sets_status():
    """Test that refuse node sets status to refused."""
    state = {}
    result = refuse(state)
    assert result["status"] == "refused"
    assert "answer" in result