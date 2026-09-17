"""Tests for static shortest-path routing."""

import networkx as nx
import pytest

from src.network import create_default_network
from src.routing import find_shortest_route


def _path_cost(graph: nx.DiGraph, route: list, weight: str = "free_flow_time") -> float:
    return sum(graph[u][v][weight] for u, v in zip(route, route[1:]))


# ---------------------------------------------------------------------------
# Happy path on the default network — Primary Seam
# ---------------------------------------------------------------------------


def test_default_network_route_endpoints_and_connectivity():
    graph = create_default_network()
    origin, destination = 0, 5

    route = find_shortest_route(graph, origin, destination)

    assert route[0] == origin
    assert route[-1] == destination
    for u, v in zip(route, route[1:]):
        assert graph.has_edge(u, v)


def test_default_network_route_cost_matches_networkx_shortest_path_length():
    graph = create_default_network()
    origin, destination = 0, 5

    route = find_shortest_route(graph, origin, destination)
    expected_length = nx.shortest_path_length(
        graph, source=origin, target=destination, weight="free_flow_time"
    )

    assert _path_cost(graph, route) == expected_length


def test_route_is_pure_and_does_not_mutate_graph():
    graph = create_default_network()
    edge_snapshot = {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)}
    node_snapshot = set(graph.nodes)

    find_shortest_route(graph, 0, 5)

    assert set(graph.nodes) == node_snapshot
    assert {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)} == edge_snapshot


# ---------------------------------------------------------------------------
# Custom weighted graph — Secondary Seam
# ---------------------------------------------------------------------------


def test_weighted_graph_prefers_free_flow_cheapest_over_fewest_hops():
    """Fewest hops A→C costs 10; A→B→C costs 2 and must be chosen."""
    graph = nx.DiGraph()
    graph.add_edge("A", "C", free_flow_time=10.0)
    graph.add_edge("A", "B", free_flow_time=1.0)
    graph.add_edge("B", "C", free_flow_time=1.0)

    route = find_shortest_route(graph, "A", "C")

    assert route == ["A", "B", "C"]
    assert _path_cost(graph, route) == 2.0
    assert _path_cost(graph, ["A", "C"]) == 10.0


def test_custom_weight_parameter_uses_supplied_attribute():
    graph = nx.DiGraph()
    # Free-flow prefers A→B→C; current travel time prefers direct A→C.
    graph.add_edge("A", "C", free_flow_time=10.0, current_travel_time=1.0)
    graph.add_edge("A", "B", free_flow_time=1.0, current_travel_time=5.0)
    graph.add_edge("B", "C", free_flow_time=1.0, current_travel_time=5.0)

    route = find_shortest_route(graph, "A", "C", weight="current_travel_time")

    assert route == ["A", "C"]
    assert _path_cost(graph, route, weight="current_travel_time") == 1.0


# ---------------------------------------------------------------------------
# Error cases — Tertiary Seam
# ---------------------------------------------------------------------------


def test_missing_origin_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="origin"):
        find_shortest_route(graph, origin=99, destination=5)


def test_missing_destination_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="destination"):
        find_shortest_route(graph, origin=0, destination=99)


def test_origin_equals_destination_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="cannot be the same"):
        find_shortest_route(graph, origin=0, destination=0)


def test_unreachable_pair_raises_value_error():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, free_flow_time=1.0)
    graph.add_node(2)  # disconnected

    with pytest.raises(ValueError, match="no path exists"):
        find_shortest_route(graph, origin=0, destination=2)
