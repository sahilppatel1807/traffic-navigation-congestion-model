"""Synthetic road-network topology for the traffic congestion model.

The default network is a deterministic six-intersection directed graph:

    0 <----> 1 <----> 2
    ^                   ^
    |                   |
    v                   v
    3 <----> 4 <----> 5

Each physical two-way road is represented as two opposite directed edges in a
``networkx.DiGraph``. Travel times are measured in simulation time steps, while
capacity and occupancy are vehicle counts.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

DEFAULT_NODE_COUNT = 6

DEFAULT_EDGE_ATTRIBUTES: dict[str, float | int] = {
    "free_flow_time": 1.0,
    "capacity": 10,
    "occupancy": 0,
}

DEFAULT_ROADS: tuple[dict[str, Any], ...] = (
    {"u": 0, "v": 1, "free_flow_time": 1.0, "capacity": 10, "bidirectional": True},
    {"u": 1, "v": 2, "free_flow_time": 1.0, "capacity": 10, "bidirectional": True},
    {"u": 3, "v": 4, "free_flow_time": 1.0, "capacity": 10, "bidirectional": True},
    {"u": 4, "v": 5, "free_flow_time": 1.0, "capacity": 10, "bidirectional": True},
    {"u": 0, "v": 3, "free_flow_time": 1.5, "capacity": 8, "bidirectional": True},
    {"u": 2, "v": 5, "free_flow_time": 1.5, "capacity": 8, "bidirectional": True},
)


def _edge_attributes(free_flow_time: float, capacity: int | float) -> dict[str, int | float]:
    return {
        **DEFAULT_EDGE_ATTRIBUTES,
        "free_flow_time": float(free_flow_time),
        "capacity": capacity,
        "current_travel_time": float(free_flow_time),
    }


def _add_road(
    graph: nx.DiGraph,
    u: int,
    v: int,
    *,
    free_flow_time: float,
    capacity: int | float,
    bidirectional: bool = True,
) -> None:
    graph.add_edge(u, v, **_edge_attributes(free_flow_time, capacity))
    if bidirectional:
        graph.add_edge(v, u, **_edge_attributes(free_flow_time, capacity))


def create_default_network() -> nx.DiGraph:
    """Create the deterministic default synthetic road network."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(DEFAULT_NODE_COUNT))

    for road in DEFAULT_ROADS:
        _add_road(graph, **road)

    return graph


def summarize_network(graph: nx.DiGraph) -> str:
    """Return a compact text summary of nodes, edges, and road attributes."""

    lines = [
        f"nodes: {graph.number_of_nodes()}",
        f"edges: {graph.number_of_edges()}",
    ]

    for u, v, attrs in sorted(graph.edges(data=True)):
        attr_summary = ", ".join(f"{key}={attrs[key]}" for key in sorted(attrs))
        lines.append(f"{u}->{v}: {attr_summary}")

    return "\n".join(lines)
