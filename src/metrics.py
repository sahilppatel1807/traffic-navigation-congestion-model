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

network_congestion_score(graph) -> float
    Uncapped capacity-weighted congestion over open edges; ``math.nan`` when
    no open edges remain.

pre_disruption_congestion(C, t_star, W) -> float
    Mean of dense series ``C`` over the pre-disruption window
    ``[t_star - W, t_star)``.

congestion_recovery(C, t_star, t_r, W, K, horizon) -> dict
    Threshold-based recovery analysis on a dense congestion series after a
    disruption/restore pair. Does not schedule disruptions or claim integrated
    experiment validity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Iterable, Mapping

import networkx as nx

from src.simulation import Simulation
from src.vehicle import Vehicle

# Fixed recovery-threshold factor: τ = RECOVERY_THRESHOLD_FACTOR * C_pre.
RECOVERY_THRESHOLD_FACTOR = 1.05


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


def network_congestion_score(graph: nx.DiGraph) -> float:
    """Return uncapped capacity-weighted congestion over open edges.

    Computes

    .. math::

        C = \\frac{\\sum_{e \\in \\mathrm{open}} occupancy_e}
                  {\\sum_{e \\in \\mathrm{open}} capacity_e}

    Closed or missing edges are excluded by absence from the graph (closures
    remove edges). Ratios are not capped, so overload above capacity remains
    visible.

    Parameters
    ----------
    graph:
        Directed road network. Every open edge must expose numeric
        ``capacity`` and ``occupancy`` attributes.

    Returns
    -------
    float
        Capacity-weighted congestion score, or ``math.nan`` when there are no
        open edges.

    Raises
    ------
    KeyError
        If an edge is missing ``capacity`` or ``occupancy``.
    TypeError
        If ``capacity`` or ``occupancy`` is not a real number.
    ValueError
        If ``capacity <= 0`` or ``occupancy < 0``.
    """
    total_occupancy = 0.0
    total_capacity = 0.0
    open_edges = 0

    for _u, _v, attrs in graph.edges(data=True):
        capacity, occupancy = _validate_edge_occupancy_capacity(attrs)
        total_occupancy += occupancy
        total_capacity += capacity
        open_edges += 1

    if open_edges == 0:
        return math.nan

    return total_occupancy / total_capacity


def pre_disruption_congestion(C: Sequence[float], t_star: int, W: int) -> float:
    """Return the mean of ``C`` over the pre-disruption window ``[t_star-W, t_star)``.

    Parameters
    ----------
    C:
        Dense congestion series indexed by absolute timestep ``0..H``.
    t_star:
        Disruption onset timestep (exclusive end of the pre-window).
    W:
        Pre-disruption window length in steps (``W >= 1``).

    Returns
    -------
    float
        Arithmetic mean of ``C[t_star - W]`` … ``C[t_star - 1]``.

    Raises
    ------
    TypeError
        If ``C`` is not a non-string sequence.
    ValueError
        If ``W < 1``, ``t_star < 0``, ``t_star < W``, or ``len(C) < t_star``.
    """
    _require_dense_series(C)
    _require_positive_int(W, "W")
    if not isinstance(t_star, int) or isinstance(t_star, bool):
        raise TypeError(f"t_star must be an integer, got {type(t_star).__name__}")
    if t_star < 0:
        raise ValueError(f"t_star must be >= 0, got {t_star}")
    if t_star < W:
        raise ValueError(
            f"t_star must be >= W for a full pre-disruption window, "
            f"got t_star={t_star}, W={W}"
        )
    if len(C) < t_star:
        raise ValueError(
            f"C must cover the pre-disruption window "
            f"(len(C) >= t_star={t_star}), got len(C)={len(C)}"
        )

    window = C[t_star - W : t_star]
    return sum(window) / W


def congestion_recovery(
    C: Sequence[float],
    t_star: int,
    t_r: int,
    W: int,
    K: int,
    horizon: int,
) -> dict[str, Any]:
    """Analyse threshold recovery of network congestion after a disruption.

    Computes ``C_pre`` via :func:`pre_disruption_congestion` and sets
    ``τ = 1.05 * C_pre`` internally. Recovery requires an excursion above
    ``τ`` during ``[t_star, t_r)`` and a post-restore ``K``-step streak with
    ``C(t) ≤ τ`` that confirms on or before the inclusive ``horizon``.

    Recommended caller defaults (not hard-coded here): ``W = 4 * T_corr`` and
    ``K = 2 * T_corr`` for a configuration-derived free-flow OD time
    ``T_corr``.

    Parameters
    ----------
    C:
        Dense congestion series indexed by absolute timestep ``0..H``.
        Requires ``len(C) >= horizon + 1``. Dictionaries are unsupported.
    t_star:
        Disruption onset; recovery clock starts here.
    t_r:
        Restore time. Valid schedules satisfy ``t_star ≤ t_r ≤ horizon``.
    W:
        Pre-disruption window length for ``C_pre``.
    K:
        Confirmation streak length (``K >= 1``).
    horizon:
        Inclusive absolute timestep index ``H`` for the analysis window.

    Returns
    -------
    dict
        Structured recovery result. Programming errors raise; invalid
        schedules / non-positive ``C_pre`` return ``valid=False`` with
        ``invalid_reason`` and null derived fields.

    Raises
    ------
    TypeError
        If ``C`` is not a non-string sequence.
    ValueError
        If ``K < 1``, ``t_star < 0``, ``W < 1``, ``t_star < W``, or
        ``len(C) <= horizon``.
    """
    _require_dense_series(C)
    _require_positive_int(K, "K")
    _require_positive_int(W, "W")
    if not isinstance(t_star, int) or isinstance(t_star, bool):
        raise TypeError(f"t_star must be an integer, got {type(t_star).__name__}")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError(f"horizon must be an integer, got {type(horizon).__name__}")
    if t_star < 0:
        raise ValueError(f"t_star must be >= 0, got {t_star}")
    if t_star < W:
        raise ValueError(
            f"t_star must be >= W for a full pre-disruption window, "
            f"got t_star={t_star}, W={W}"
        )
    if len(C) <= horizon:
        raise ValueError(
            f"C must satisfy len(C) >= horizon + 1 "
            f"(got len(C)={len(C)}, horizon={horizon})"
        )

    base = {
        "valid": True,
        "invalid_reason": None,
        "recovered": False,
        "recovery_time": None,
        "recovery_start": None,
        "confirmation_time": None,
        "censored": False,
        "censor_time": None,
        "no_excursion": False,
        "excursion_start": None,
        "tau": None,
        "C_pre": None,
        "t_star": t_star,
        "t_r": t_r,
        "K": K,
        "W": W,
        "horizon": horizon,
    }

    if (
        not isinstance(t_r, int)
        or isinstance(t_r, bool)
        or t_r < t_star
        or t_r > horizon
    ):
        return {
            **base,
            "valid": False,
            "invalid_reason": "invalid_restore_time",
        }

    c_pre = pre_disruption_congestion(C, t_star, W)
    base["C_pre"] = c_pre
    if c_pre <= 0:
        return {
            **base,
            "valid": False,
            "invalid_reason": "non_positive_pre_disruption_congestion",
        }

    tau = RECOVERY_THRESHOLD_FACTOR * c_pre
    base["tau"] = tau

    excursion_start: int | None = None
    for t in range(t_star, t_r):
        if C[t] > tau:
            excursion_start = t
            break

    if excursion_start is None:
        return {
            **base,
            "no_excursion": True,
            "excursion_start": None,
        }

    base["excursion_start"] = excursion_start

    # Inclusive horizon: streak confirming at start+K-1 must satisfy
    # start + K - 1 <= horizon, i.e. start <= horizon - K + 1.
    latest_start = horizon - K + 1
    for start in range(t_r, latest_start + 1):
        if all(C[t] <= tau for t in range(start, start + K)):
            confirmation = start + K - 1
            return {
                **base,
                "recovered": True,
                "recovery_time": start - t_star,
                "recovery_start": start,
                "confirmation_time": confirmation,
                "censored": False,
                "censor_time": None,
                "no_excursion": False,
            }

    return {
        **base,
        "recovered": False,
        "recovery_time": None,
        "recovery_start": None,
        "confirmation_time": None,
        "censored": True,
        "censor_time": horizon - t_star,
        "no_excursion": False,
    }


def _edge_congestion_ratio(attrs: Mapping[str, Any]) -> float:
    """Validate edge attributes and return ``min(occupancy / capacity, 1.0)``."""
    capacity, occupancy = _validate_edge_occupancy_capacity(attrs)
    return min(occupancy / capacity, 1.0)


def _validate_edge_occupancy_capacity(
    attrs: Mapping[str, Any],
) -> tuple[float, float]:
    """Validate and return ``(capacity, occupancy)`` as floats."""
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

    return float(capacity), float(occupancy)


def _require_dense_series(C: object) -> None:
    """Reject non-sequences and string/bytes stand-ins for a float series."""
    if isinstance(C, (str, bytes)) or not isinstance(C, Sequence):
        raise TypeError(
            f"C must be a dense Sequence[float], got {type(C).__name__}"
        )


def _require_positive_int(value: object, name: str) -> None:
    """Require a non-bool integer ``>= 1``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
