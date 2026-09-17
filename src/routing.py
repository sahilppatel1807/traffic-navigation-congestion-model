"""Static shortest-path routing on directed road graphs.

Uninformed / static routing chooses the path that minimises total free-flow
travel time (``free_flow_time`` edge weights). The same helper can later be
reused for selfish routing by passing ``weight="current_travel_time"``.

Every edge must carry the chosen weight attribute; missing weights are not
wrapped — NetworkX errors propagate so graph data problems stay visible.

Public API
----------
find_shortest_route(graph, origin, destination, weight) -> list
    Pure Dijkstra shortest path; does not mutate the graph or assign routes.
"""

from __future__ import annotations

from typing import Any

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
