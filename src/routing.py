"""Static shortest-path routing and route-cost estimation on directed road graphs.

Uninformed / static routing chooses the path that minimises total free-flow
travel time (``free_flow_time`` edge weights). The same helper can later be
reused for selfish routing by passing ``weight="current_travel_time"``.

``estimate_route_cost`` sums a chosen edge-weight attribute along a planned
node-list route. By default it uses live congested travel times
(``current_travel_time``). Callers must refresh BPR travel times before asking
for a live cost; this helper never updates congestion itself.

Every edge must carry the chosen weight attribute; missing weights are not
wrapped — NetworkX / KeyError errors propagate so graph data problems stay
visible.

Public API
----------
find_shortest_route(graph, origin, destination, weight) -> list
    Pure Dijkstra shortest path; does not mutate the graph or assign routes.
estimate_route_cost(graph, route, weight) -> float
    Pure sum of edge weights along a route; does not mutate the graph.
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx


def find_shortest_route(
    graph: nx.DiGraph,
    origin: Any,
    destination: Any,
    weight: str = "free_flow_time",
) -> list[Any]:
    """Return a shortest path of nodes from ``origin`` to ``destination``.

    Uses NetworkX Dijkstra with the given edge-attribute ``weight``. The default
    weight ``"free_flow_time"`` implements uninformed / static routing.

    Parameters
    ----------
    graph:
        Directed road network. Each edge must define the attribute named by
        ``weight``.
    origin:
        Start node. Must be present in ``graph``.
    destination:
        End node. Must be present in ``graph`` and distinct from ``origin``.
    weight:
        Edge attribute used as Dijkstra cost. Defaults to ``"free_flow_time"``.

    Returns
    -------
    list
        Ordered node list from ``origin`` to ``destination`` inclusive. Suitable
        for ``Vehicle.set_route``.

    Raises
    ------
    ValueError
        If ``origin`` or ``destination`` is missing from the graph, if they are
        equal, or if no directed path exists between them.
    """
    if origin not in graph:
        raise ValueError(f"origin {origin!r} is not a node in the graph")
    if destination not in graph:
        raise ValueError(f"destination {destination!r} is not a node in the graph")
    if origin == destination:
        raise ValueError(
            f"origin and destination cannot be the same: {origin!r}"
        )

    try:
        return nx.shortest_path(
            graph, source=origin, target=destination, weight=weight
        )
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"no path exists between origin {origin!r} and destination {destination!r}"
        ) from exc


def estimate_route_cost(
    graph: nx.DiGraph,
    route: Sequence[Any],
    weight: str = "current_travel_time",
) -> float:
    """Return the sum of ``weight`` along consecutive edges of ``route``.

    Pure read/sum: does not mutate ``graph`` and does not invoke BPR congestion
    updates. The default ``weight="current_travel_time"`` scores a path under
    live congested travel times already stored on each directed edge.

    Parameters
    ----------
    graph:
        Directed road network.
    route:
        Ordered node list with at least two nodes. Consecutive pairs must be
        directed edges in ``graph``.
    weight:
        Edge attribute summed along the route. Defaults to
        ``"current_travel_time"``.

    Returns
    -------
    float
        Sum of the named attribute over each consecutive pair
        ``(route[i], route[i + 1])``.

    Raises
    ------
    ValueError
        If ``route`` is empty, has a single node, or is missing a directed edge
        between consecutive nodes.
    KeyError
        If an existing edge lacks the ``weight`` attribute (not wrapped).
    """
    if len(route) == 0:
        raise ValueError("route must not be empty")
    if len(route) == 1:
        raise ValueError(
            f"route must contain at least two nodes; got a single node {route[0]!r}"
        )

    total = 0.0
    for u, v in zip(route, route[1:]):
        if not graph.has_edge(u, v):
            raise ValueError(
                f"no directed edge between consecutive route nodes {u!r} and {v!r}"
            )
        total += graph[u][v][weight]
    return float(total)
