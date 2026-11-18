"""Integration tests for the full graph."""

import pytest

from app.graph.builder import build_graph


def test_graph_compiles():
    """Test that the graph compiles without errors."""
    graph = build_graph()
    assert graph is not None


def test_graph_invocation():
    """Test basic graph invocation."""
    graph = build_graph()
    result = graph.invoke(
        {"question": "What is the late fee?", "retries": 0},
        config={"recursion_limit": 15},
    )
    assert "answer" in result
    assert "status" in result