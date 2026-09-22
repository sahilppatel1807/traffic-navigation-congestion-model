"""Named demand presets and corridor batch builders for baseline scenarios.

Demand intensity is expressed as a simultaneous batch size ``n`` against the
default corridor capacity of 10. Vehicles are returned bare (no route); the
simulation auto-routes on entry using free-flow travel times.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.vehicle import Vehicle

# Locked presets against top-corridor capacity 10.
DEMAND_LOW = 1
DEMAND_MEDIUM = 5
DEMAND_HIGH = 15

DEFAULT_ORIGIN = 0
DEFAULT_DESTINATION = 2

DEMAND_LEVELS: dict[str, int] = {
    "low": DEMAND_LOW,
    "medium": DEMAND_MEDIUM,
    "high": DEMAND_HIGH,
}


def build_corridor_demand(
    graph: nx.DiGraph,
    n: int,
    *,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
) -> list[Vehicle]:
    """Build ``n`` bare corridor vehicles released at ``start_time=0``.

    Parameters
    ----------
    graph:
        Directed road network used to validate origin and destination nodes.
    n:
        Batch size (number of vehicles). Must be >= 1.
    origin:
        Trip origin node. Defaults to ``0`` (top corridor start).
    destination:
        Trip destination node. Defaults to ``2`` (top corridor end).

    Returns
    -------
    list[Vehicle]
        Vehicles with no pre-assigned route and ``start_time=0``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``, or if origin/destination are missing from ``graph``.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    vehicles: list[Vehicle] = []
    for i in range(n):
        vehicle = Vehicle.create_for_network(
            graph,
            vehicle_id=f"demand-{i}",
            origin=origin,
            destination=destination,
            start_time=0,
        )
        vehicles.append(vehicle)
    return vehicles


def build_low_demand(
    graph: nx.DiGraph,
    *,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
) -> list[Vehicle]:
    """Build the low-demand preset batch (``n=1``)."""
    return build_corridor_demand(
        graph, DEMAND_LOW, origin=origin, destination=destination
    )


def build_medium_demand(
    graph: nx.DiGraph,
    *,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
) -> list[Vehicle]:
    """Build the medium-demand preset batch (``n=5``)."""
    return build_corridor_demand(
        graph, DEMAND_MEDIUM, origin=origin, destination=destination
    )


def build_high_demand(
    graph: nx.DiGraph,
    *,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
) -> list[Vehicle]:
    """Build the high-demand preset batch (``n=15``)."""
    return build_corridor_demand(
        graph, DEMAND_HIGH, origin=origin, destination=destination
    )
