"""Test for retrieval module."""

import pytest

from app.retrieval.fusion import rrf


def test_rrf_basic():
    """Test reciprocal rank fusion with simple inputs."""
    results = [
        [{"id": 1}, {"id": 2}, {"id": 3}],
        [{"id": 2}, {"id": 4}, {"id": 1}],
    ]
    fused = rrf(results, top_n=3)
    assert len(fused) == 3
    # Doc 2 appears in both lists at top, should rank highest
    assert fused[0]["id"] == 2


def test_rrf_empty():
    """Test RRF with empty lists."""
    results = [[], []]
    fused = rrf(results)
    assert fused == []


def test_rrf_single_list():
    """Test RRF with single list."""
    results = [[{"id": 1}, {"id": 2}]]
    fused = rrf(results)
    assert len(fused) == 2