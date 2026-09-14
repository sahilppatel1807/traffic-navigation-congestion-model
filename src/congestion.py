"""Congestion travel-time calculation using the Bureau of Public Roads (BPR) formula.

The BPR formula estimates travel time on a road given its free-flow speed, capacity,
and current occupancy:

    t = t0 * (1 + alpha * (occupancy / capacity) ** beta)

where:
    t0          free-flow travel time (simulation steps)
    occupancy   number of vehicles currently on the road
    capacity    maximum number of vehicles the road can handle efficiently
    alpha       sensitivity coefficient (default 0.15)
    beta        shape exponent (default 4.0)

Public API
----------
calculate_travel_time(free_flow_time, capacity, occupancy, alpha, beta) -> float
    Pure function; raises ValueError for invalid inputs.

update_road_travel_time(road, alpha, beta) -> None
    Mutates a NetworkX edge-attribute dict in place.

update_all_travel_times(graph, alpha, beta) -> None
    Refreshes every edge in a networkx.DiGraph.
"""

from __future__ import annotations

import networkx as nx

DEFAULT_ALPHA: float = 0.15
DEFAULT_BETA: float = 4.0


def calculate_travel_time(
    free_flow_time: float,
    capacity: float,
    occupancy: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """Return the BPR travel time for a road given its current occupancy.

    Parameters
    ----------
    free_flow_time:
        Travel time when the road is empty (simulation steps). Must be >= 0.
    capacity:
        Vehicle count at which the road operates at designed capacity. Must be > 0.
    occupancy:
        Current number of vehicles on the road. Must be >= 0.
    alpha:
        BPR sensitivity coefficient. Defaults to ``DEFAULT_ALPHA`` (0.15).
    beta:
        BPR shape exponent. Defaults to ``DEFAULT_BETA`` (4.0).

    Returns
    -------
    float
        Congested travel time in simulation steps.

    Raises
    ------
    ValueError
        If ``capacity <= 0``, ``occupancy < 0``, or ``free_flow_time < 0``.
    """
    if free_flow_time < 0:
        raise ValueError(
            f"free_flow_time must be >= 0, got {free_flow_time!r}"
        )
    if capacity <= 0:
        raise ValueError(
            f"capacity must be > 0, got {capacity!r}"
        )
    if occupancy < 0:
        raise ValueError(
            f"occupancy must be >= 0, got {occupancy!r}"
        )

    return free_flow_time * (1 + alpha * (occupancy / capacity) ** beta)


def update_road_travel_time(
    road: dict,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> None:
    """Update ``road["current_travel_time"]`` using the BPR formula.

    Reads ``free_flow_time``, ``capacity``, and ``occupancy`` from the edge-attribute
    dict ``road`` (as produced by ``src/network.py``) and writes the result back to
    ``road["current_travel_time"]``.

    Parameters
    ----------
    road:
        A NetworkX edge-attribute dictionary. Must contain ``free_flow_time``,
        ``capacity``, and ``occupancy``.
    alpha:
        BPR sensitivity coefficient. Defaults to ``DEFAULT_ALPHA``.
    beta:
        BPR shape exponent. Defaults to ``DEFAULT_BETA``.
    """
    road["current_travel_time"] = calculate_travel_time(
        free_flow_time=road["free_flow_time"],
        capacity=road["capacity"],
        occupancy=road["occupancy"],
        alpha=alpha,
        beta=beta,
    )


def update_all_travel_times(
    graph: nx.DiGraph,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> None:
    """Refresh ``current_travel_time`` on every edge of *graph*.

    Iterates over all edges in a ``networkx.DiGraph`` and calls
    ``update_road_travel_time`` on each edge's attribute dictionary.

    Parameters
    ----------
    graph:
        A directed road network. Each edge must have ``free_flow_time``,
        ``capacity``, and ``occupancy`` attributes.
    alpha:
        BPR sensitivity coefficient passed through to each edge update.
    beta:
        BPR shape exponent passed through to each edge update.
    """
    for _u, _v, road in graph.edges(data=True):
        update_road_travel_time(road, alpha=alpha, beta=beta)
