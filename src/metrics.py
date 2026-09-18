"""Journey-level and road-level metrics for the traffic simulation.

Public API
----------
journey_time(vehicle) -> int | None
    Duration in discrete simulation steps, or ``None`` if incomplete.

completed_count(vehicles) -> int
    Number of vehicles with a recorded ``completion_time``.

mean_journey_time(vehicles) -> float | None
    Mean of completed journey times; ``None`` when none are complete.

road_congestion(graph) -> dict[tuple, float]
    Directed-edge occupancy/capacity ratios, each capped at ``1.0``.

simulation_summary(simulation) -> dict
    Plain dictionary with ``completed_count``, ``mean_journey_time``, and
    ``road_congestion``. Does not mutate simulation state.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx

from src.simulation import Simulation
from src.vehicle import Vehicle


def journey_time(vehicle: Vehicle) -> int | None:
    """Return ``completion_time - start_time`` in simulation steps.

    Incomplete vehicles (``completion_time is None``) return ``None`` rather
    than a zero-duration journey.

    Parameters
    ----------
    vehicle:
        A vehicle agent with ``start_time`` and ``completion_time``.

    Returns
    -------
    int | None
        Journey duration in discrete steps, or ``None`` if incomplete.

    Raises
    ------
    TypeError
        If ``start_time`` is not an integer, or if ``completion_time`` is set
        but is not an integer.
    ValueError
        If ``completion_time`` is less than ``start_time``.
    """
    start = vehicle.start_time
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError(
            f"start_time must be an integer, got {type(start).__name__}"
        )

    completion = vehicle.completion_time
    if completion is None:
        return None

    if not isinstance(completion, int) or isinstance(completion, bool):
        raise TypeError(
            f"completion_time must be an integer, got {type(completion).__name__}"
        )
    if completion < start:
        raise ValueError(
            f"completion_time {completion} cannot be less than start_time {start}"
        )

    return completion - start


def completed_count(vehicles: Iterable[Vehicle]) -> int:
    """Return the number of vehicles that have finished their journeys."""
    return sum(1 for vehicle in vehicles if vehicle.completion_time is not None)


def mean_journey_time(vehicles: Iterable[Vehicle]) -> float | None:
    """Return the mean journey time over completed vehicles only.

    Incomplete vehicles are excluded. If no vehicle is complete, returns
    ``None``.
    """
    durations: list[int] = []
    for vehicle in vehicles:
        duration = journey_time(vehicle)
        if duration is not None:
            durations.append(duration)

    if not durations:
        return None

    return sum(durations) / len(durations)


def road_congestion(graph: nx.DiGraph) -> dict[tuple[Any, Any], float]:
    """Return each directed edge's occupancy/capacity ratio, capped at ``1.0``.

    Parameters
    ----------
    graph:
        Directed road network. Every edge must expose ``capacity`` and
        ``occupancy`` attributes.

    Returns
    -------
    dict[tuple, float]
        Mapping from ``(u, v)`` edge keys to congestion ratios in ``[0.0, 1.0]``.

    Raises
    ------
    KeyError
        If an edge is missing ``capacity`` or ``occupancy``.
    TypeError
        If ``capacity`` or ``occupancy`` is not a real number.
    ValueError
        If ``capacity <= 0`` or ``occupancy < 0``.
    """
    ratios: dict[tuple[Any, Any], float] = {}
    for u, v, attrs in graph.edges(data=True):
        ratios[(u, v)] = _edge_congestion_ratio(attrs)
    return ratios


def simulation_summary(simulation: Simulation) -> dict[str, Any]:
    """Return a plain-dictionary snapshot of journey and road metrics.

    Consumes ``simulation.vehicles`` and ``simulation.graph`` without mutating
    either. Keys: ``completed_count``, ``mean_journey_time``, ``road_congestion``.
    """
    vehicles = simulation.vehicles
    return {
        "completed_count": completed_count(vehicles),
        "mean_journey_time": mean_journey_time(vehicles),
        "road_congestion": road_congestion(simulation.graph),
    }


def _edge_congestion_ratio(attrs: Mapping[str, Any]) -> float:
    """Validate edge attributes and return ``min(occupancy / capacity, 1.0)``."""
    capacity = attrs["capacity"]
    occupancy = attrs["occupancy"]

    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
        raise TypeError(
            f"capacity must be a number, got {type(capacity).__name__}"
        )
    if isinstance(occupancy, bool) or not isinstance(occupancy, (int, float)):
        raise TypeError(
            f"occupancy must be a number, got {type(occupancy).__name__}"
        )
    if capacity <= 0:
        raise ValueError(f"capacity must be > 0, got {capacity!r}")
    if occupancy < 0:
        raise ValueError(f"occupancy must be >= 0, got {occupancy!r}")

    return min(occupancy / capacity, 1.0)
