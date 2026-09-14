"""Tests for the default synthetic road network."""

import networkx as nx

from src.network import create_default_network, summarize_network


REQUIRED_EDGE_ATTRIBUTES = {
    "free_flow_time",
    "capacity",
    "occupancy",
    "current_travel_time",
}

TWO_WAY_ROADS = {
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (0, 3),
    (2, 5),
}


def test_default_network_is_directed_graph_with_expected_size():
    graph = create_default_network()

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 12


def test_default_network_edges_have_required_attributes():
    graph = create_default_network()

    for _, _, attrs in graph.edges(data=True):
        assert REQUIRED_EDGE_ATTRIBUTES <= attrs.keys()
        assert attrs["occupancy"] == 0
        assert attrs["current_travel_time"] == attrs["free_flow_time"]


def test_two_way_roads_are_explicit_opposite_directed_edges():
    graph = create_default_network()

    for u, v in TWO_WAY_ROADS:
        assert graph.has_edge(u, v)
        assert graph.has_edge(v, u)


def test_default_network_is_reproducible():
    first = create_default_network()
    second = create_default_network()

    assert set(first.nodes) == set(second.nodes)
    assert set(first.edges) == set(second.edges)

    for edge in first.edges:
        assert first.edges[edge] == second.edges[edge]


def test_summarize_network_includes_counts_and_edge_attributes():
    summary = summarize_network(create_default_network())

    assert "nodes: 6" in summary
    assert "edges: 12" in summary
    assert "0->1:" in summary
    assert "free_flow_time=1.0" in summary
