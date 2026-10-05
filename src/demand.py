"""Named demand presets and corridor batch builders for baseline scenarios.

Demand intensity is expressed as a simultaneous batch size ``n`` against the
default corridor capacity of 10. Vehicles are returned bare (no route); the
simulation auto-routes on entry using free-flow travel times.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def build_scheduled_corridor_demand(
    graph: nx.DiGraph,
    schedule: Mapping[int, int],
    *,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
    vehicle_id_prefix: str = "scheduled",
) -> list[Vehicle]:
    """Build corridor vehicles from a reproducible release schedule.

    ``schedule`` maps a non-negative simulation timestep to a positive number
    of vehicles.  The returned list is ordered by release time and then by
    vehicle number.  Existing batch builders intentionally remain unchanged;
    this helper is for finite, staggered presentation and experiment demand.
    """
    if not isinstance(schedule, Mapping):
        raise TypeError(
            f"schedule must be a mapping of timestep to count, got "
            f"{type(schedule).__name__}"
        )
    if not isinstance(vehicle_id_prefix, str):
        raise TypeError("vehicle_id_prefix must be a string")
    if not schedule:
        raise ValueError("schedule must contain at least one release")

    vehicles: list[Vehicle] = []
    vehicle_number = 0
    for start_time, count in sorted(schedule.items()):
        if not isinstance(start_time, int) or isinstance(start_time, bool):
            raise TypeError(
                f"schedule timesteps must be integers, got {type(start_time).__name__}"
            )
        if start_time < 0:
            raise ValueError(f"schedule timesteps must be >= 0, got {start_time}")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(
                f"schedule counts must be integers, got {type(count).__name__}"
            )
        if count < 1:
            raise ValueError(f"schedule counts must be >= 1, got {count}")

        for _ in range(count):
            vehicles.append(
                Vehicle.create_for_network(
                    graph,
                    vehicle_id=f"{vehicle_id_prefix}-{vehicle_number}",
                    origin=origin,
                    destination=destination,
                    start_time=start_time,
                )
            )
            vehicle_number += 1
    return vehicles


def build_staggered_corridor_demand(
    graph: nx.DiGraph,
    *,
    first_wave: int = 10,
    second_wave: int = 10,
    first_release: int = 0,
    second_release: int = 3,
    origin: Any = DEFAULT_ORIGIN,
    destination: Any = DEFAULT_DESTINATION,
) -> list[Vehicle]:
    """Build the focused presentation schedule: ten vehicles at step 0, then ten at step 3.

    The gap of 3 steps between waves is large enough that navigation users in
    wave 2 see a congested primary corridor (0→1→2) and herd onto the
    alternative (0→3→4→5→2).  With both corridors capacity-constrained at 2,
    herding at high adoption saturates whichever route is recommended.
    The optimal adoption sits near 50%, where the nav-user cohort is small
    enough to relieve primary congestion without flooding the alternative.
    """
    return build_scheduled_corridor_demand(
        graph,
        {first_release: first_wave, second_release: second_wave},
        origin=origin,
        destination=destination,
        vehicle_id_prefix="staggered",
    )


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
