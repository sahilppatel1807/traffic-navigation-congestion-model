"""Road-disruption helpers for experiment scenarios.

This module applies physical-road capacity cuts and full closures without
owning the simulation clock or experiment harness. Callers may invoke helpers
before a run or between steps. Capacity cuts refresh travel times on affected
edges via the congestion updater; closures remove edges and do not refresh.
"""

from __future__ import annotations

import networkx as nx

from src.congestion import update_road_travel_time


def close_road(graph: nx.DiGraph, u, v) -> nx.DiGraph:
    """Fully close a physical road by removing both directed edges.

    Requires directed edges ``(u, v)`` and ``(v, u)``. Removes both edges
    (discarding their attributes), leaves nodes in place, mutates ``graph``
    in place, and returns the same graph object. Does not refresh travel
    times on remaining edges, check connectivity, or inspect occupancy.

    Applying closure mid-run while vehicles may still reference removed
    edges is unsupported for this milestone.

    Parameters
    ----------
    graph:
        Directed road network.
    u, v:
        Ordered pair naming a physical two-way road (both directions required).

    Returns
    -------
    networkx.DiGraph
        The same ``graph`` object after both edges are removed.

    Raises
    ------
    ValueError
        If either directed edge ``(u, v)`` or ``(v, u)`` is missing.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"missing directed edge ({u!r}, {v!r})")
    if not graph.has_edge(v, u):
        raise ValueError(f"missing directed edge ({v!r}, {u!r})")

    graph.remove_edge(u, v)
    graph.remove_edge(v, u)
    return graph


def reduce_road_capacity(
    graph: nx.DiGraph,
    u,
    v,
    *,
    factor,
) -> nx.DiGraph:
    """Reduce capacity on both directions of a physical road by ``factor``.

    Requires directed edges ``(u, v)`` and ``(v, u)``. Multiplies each
    direction's current capacity by ``factor`` (float product, no rounding),
    leaves occupancy unchanged, refreshes ``current_travel_time`` on both
    edges via :func:`~src.congestion.update_road_travel_time`, mutates
    ``graph`` in place, and returns the same graph object.

    Parameters
    ----------
    graph:
        Directed road network.
    u, v:
        Ordered pair naming a physical two-way road (both directions required).
    factor:
        Remaining-capacity fraction in ``(0.0, 1.0]``. Accepts ``int`` or
        ``float`` (including integer ``1``); rejects ``bool``.

    Returns
    -------
    networkx.DiGraph
        The same ``graph`` object after the cut.

    Raises
    ------
    TypeError
        If ``factor`` is a ``bool`` or not an ``int``/``float``, or if either
        direction's existing ``capacity`` is non-numeric.
    ValueError
        If ``factor`` is outside ``(0.0, 1.0]``, either directed edge is
        missing, or either direction's existing ``capacity`` is ``<= 0``.
    """
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise TypeError(
            f"factor must be an int or float in (0.0, 1.0], got {type(factor).__name__}"
        )
    if factor <= 0.0 or factor > 1.0:
        raise ValueError(f"factor must be in (0.0, 1.0], got {factor}")

    if not graph.has_edge(u, v):
        raise ValueError(f"missing directed edge ({u!r}, {v!r})")
    if not graph.has_edge(v, u):
        raise ValueError(f"missing directed edge ({v!r}, {u!r})")

    forward = graph[u][v]
    reverse = graph[v][u]
    _require_positive_numeric_capacity(forward, u, v)
    _require_positive_numeric_capacity(reverse, v, u)

    forward["capacity"] = float(forward["capacity"]) * float(factor)
    reverse["capacity"] = float(reverse["capacity"]) * float(factor)

    update_road_travel_time(forward)
    update_road_travel_time(reverse)

    return graph


def _require_positive_numeric_capacity(road: dict, u, v) -> None:
    capacity = road["capacity"]
    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
        raise TypeError(
            f"capacity on edge ({u!r}, {v!r}) must be numeric, "
            f"got {type(capacity).__name__}"
        )
    if capacity <= 0:
        raise ValueError(
            f"capacity on edge ({u!r}, {v!r}) must be > 0, got {capacity!r}"
        )
