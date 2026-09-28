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
compare_selfish_and_coordinated(graph, vehicles, until) -> dict
    Run selfish vs coordinated and return a comparison summary dict.
"""

from __future__ import annotations

import copy
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


def compare_selfish_and_coordinated(
    graph: nx.DiGraph,
    vehicles: list,
    until: int,
) -> dict[str, float]:
    """Compare selfish and coordinated routing on identical scenarios.

    Both runs operate on independent deep copies of ``graph`` and
    ``vehicles``. In the selfish run every vehicle has
    ``uses_navigation_app=True`` (ignoring original adoption flags); any
    pre-set routes are preserved. In the coordinated run
    ``routing_policy="coordinated"`` is used.

    Parameters
    ----------
    graph:
        Directed road network. Deep-copied before each run.
    vehicles:
        Pre-built vehicle list. Deep-copied before each run.
    until:
        Maximum simulation steps for each run. Passed straight to
        :meth:`~src.simulation.Simulation.run`.

    Returns
    -------
    dict
        Keys: ``selfish_total_journey_time``, ``selfish_mean_journey_time``,
        ``coordinated_total_journey_time``, ``coordinated_mean_journey_time``.

    Raises
    ------
    ValueError
        If any vehicle does not complete within ``until`` steps in either run.
    """
    # Import here to avoid a circular import at module level.
    from src.simulation import Simulation

    def _run(routing_policy: str, force_nav: bool) -> list:
        g = copy.deepcopy(graph)
        vs = copy.deepcopy(vehicles)
        if force_nav:
            for v in vs:
                v.uses_navigation_app = True
        sim = Simulation(g, vs, routing_policy=routing_policy)
        sim.run(until=until)
        return vs

    selfish_vs = _run("decentralized", force_nav=True)
    coordinated_vs = _run("coordinated", force_nav=False)

    def _check_and_sum(vs: list, label: str) -> tuple[float, float]:
        times = []
        for v in vs:
            if v.completion_time is None:
                raise ValueError(
                    f"compare_selfish_and_coordinated: vehicle {v.vehicle_id!r} "
                    f"did not complete within {until} steps under {label} routing"
                )
            times.append(v.completion_time - v.start_time)
        total = float(sum(times))
        mean = total / len(times) if times else 0.0
        return total, mean

    s_total, s_mean = _check_and_sum(selfish_vs, "selfish")
    c_total, c_mean = _check_and_sum(coordinated_vs, "coordinated")

    return {
        "selfish_total_journey_time": s_total,
        "selfish_mean_journey_time": s_mean,
        "coordinated_total_journey_time": c_total,
        "coordinated_mean_journey_time": c_mean,
    }
