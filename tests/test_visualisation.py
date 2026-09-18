"""Tests for network congestion visualisation."""

from __future__ import annotations

import copy
from pathlib import Path

import networkx as nx
import pytest

from src.network import create_default_network
from src.visualisation import DEFAULT_NODE_POSITIONS, plot_network_congestion


# ---------------------------------------------------------------------------
# plot_network_congestion — primary seam
# ---------------------------------------------------------------------------


def _snapshot_edge_attrs(graph: nx.DiGraph) -> dict:
    return {
        (u, v): copy.deepcopy(dict(attrs))
        for u, v, attrs in graph.edges(data=True)
    }


def test_plot_network_congestion_writes_nonempty_png(tmp_path: Path):
    graph = create_default_network()
    graph[0][1]["occupancy"] = 8
    graph[1][0]["occupancy"] = 2

    out = tmp_path / "congestion.png"
    plot_network_congestion(graph, out)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_network_congestion_does_not_mutate_edge_attributes(tmp_path: Path):
    graph = create_default_network()
    graph[0][1]["occupancy"] = 5
    before = _snapshot_edge_attrs(graph)

    plot_network_congestion(graph, tmp_path / "out.png")

    assert _snapshot_edge_attrs(graph) == before


def test_plot_network_congestion_zero_occupancy_writes_file(tmp_path: Path):
    """Fresh default network (all occupancy 0) is a valid uncongested input."""
    graph = create_default_network()
    out = tmp_path / "empty.png"

    plot_network_congestion(graph, out)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_network_congestion_accepts_custom_pos(tmp_path: Path):
    graph = nx.DiGraph()
    graph.add_edge("a", "b", capacity=4, occupancy=2, free_flow_time=1.0)
    graph.add_edge("b", "a", capacity=4, occupancy=0, free_flow_time=1.0)
    pos = {"a": (0.0, 0.0), "b": (1.0, 0.0)}
    out = tmp_path / "custom.png"

    plot_network_congestion(graph, out, pos=pos)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_network_congestion_rejects_incomplete_pos(tmp_path: Path):
    graph = create_default_network()
    incomplete = {0: (0.0, 0.0), 1: (1.0, 0.0)}

    with pytest.raises(ValueError, match="pos is missing"):
        plot_network_congestion(graph, tmp_path / "bad.png", pos=incomplete)


def test_default_node_positions_match_2x3_grid():
    assert set(DEFAULT_NODE_POSITIONS) == set(range(6))
    assert DEFAULT_NODE_POSITIONS[0][1] == DEFAULT_NODE_POSITIONS[1][1]
    assert DEFAULT_NODE_POSITIONS[0][1] == DEFAULT_NODE_POSITIONS[2][1]
    assert DEFAULT_NODE_POSITIONS[3][1] == DEFAULT_NODE_POSITIONS[4][1]
    assert DEFAULT_NODE_POSITIONS[3][1] == DEFAULT_NODE_POSITIONS[5][1]
    assert DEFAULT_NODE_POSITIONS[0][1] > DEFAULT_NODE_POSITIONS[3][1]
