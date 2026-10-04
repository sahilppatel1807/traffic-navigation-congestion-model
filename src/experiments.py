"""Reproducible demand, adoption, and routing experiment grids.

Crosses the named demand presets with the planned adoption fractions, runs
each cell on a fresh default network, and returns journey-metric rows. File
output lives in a thin script; this module does not write results.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from src.adoption import ADOPTION_RATES, assign_navigation_adoption
from src.demand import (
    DEMAND_LEVELS,
    build_corridor_demand,
    build_staggered_corridor_demand,
)
from src.disruption import close_road, reduce_road_capacity
from src.metrics import (
    completed_count,
    congestion_recovery,
    journey_time,
    final_completion_timestep,
    maximum_occupancy,
    makespan,
    mean_journey_time,
    most_common_route,
    network_congestion_score,
    route_diversity,
)
from src.network import create_default_network
from src.simulation import Simulation
from src.vehicle import Vehicle

# Same horizon as the uninformed baseline validation script.
GRID_HORIZON = 100

ROUTING_POLICIES: tuple[str, ...] = (
    "static", "selfish", "shared_navigation", "coordinated"
)
DISRUPTION_TYPES: tuple[str, ...] = ("none", "capacity_reduction", "closure")
COMPARISON_HORIZON = 120
# The baseline presets remain 1/5/15 vehicles, but the comparison experiment
# uses a sustained finite demand period so vehicles overlap on the network.
EXPERIMENT_DEMAND_LEVELS = {"low": 1, "medium": 5, "high": 10}
DEMAND_RELEASE_END = 3
WAVE_INTERVAL = 3
DISRUPTION_STEP = 15
RESTORE_STEP = 45
EXPERIMENT_PRIMARY_CAPACITY = 2
DISRUPTED_ROAD = (1, 2)
REDUCED_CAPACITY_FACTOR = 0.5
COMPARISON_SEEDS = (0, 1, 2, 3, 4)

# Focused presentation experiment: a finite 6 + 4 staggered demand schedule.
FOCUSED_POLICIES: tuple[str, ...] = (
    "uninformed", "selfish", "shared_navigation", "coordinated"
)
FOCUSED_ADOPTION_RATES: tuple[float, ...] = (0.0, 0.5, 1.0)
FOCUSED_DISRUPTIONS: tuple[str, ...] = ("none", "capacity_reduction")
FOCUSED_SEEDS = (0, 1, 2, 3, 4)
FOCUSED_HORIZON = 100
FOCUSED_ALTERNATIVE_TRAVEL_TIME = 12.0


def configure_focused_network(graph) -> Any:
    """Apply the focused scenario's capacity and live-choice calibration.

    The topology and node layout stay unchanged.  The lower entrance is made
    deliberately unattractive at free flow so sequential selfish entry can
    divert vehicle-by-vehicle, while a shared recommendation can keep one
    synchronized cohort on the primary corridor.  This isolates the herding
    contrast without changing the general network constructor.
    """
    for edge in ((0, 1), (1, 0), (1, 2), (2, 1)):
        graph[edge[0]][edge[1]]["capacity"] = EXPERIMENT_PRIMARY_CAPACITY
    for edge in ((0, 3), (3, 0)):
        graph[edge[0]][edge[1]]["free_flow_time"] = FOCUSED_ALTERNATIVE_TRAVEL_TIME
        graph[edge[0]][edge[1]]["current_travel_time"] = FOCUSED_ALTERNATIVE_TRAVEL_TIME
    return graph


def run_demand_adoption_grid(*, seed: int = 0) -> list[dict]:
    """Run every named demand level against every planned adoption fraction.

    Demand is the outer axis (low, medium, high) and adoption is the inner
    axis (``ADOPTION_RATES``). Each of the fifteen cells starts from a fresh
    default network, builds a bare corridor batch at the default origin and
    destination, assigns adoption with ``seed``, then runs until
    ``GRID_HORIZON``. An invalid seed fails inside
    :func:`~src.adoption.assign_navigation_adoption`.

    Parameters
    ----------
    seed:
        Shared integer seed for every cell. Defaults to ``0``. Stored on
        each returned row. Not validated here.

    Returns
    -------
    list[dict]
        One row per cell with keys ``demand_level``, ``n``,
        ``adoption_rate``, ``nav_count``, ``seed``, ``completed_count``, and
        ``mean_journey_time``. Unfinished vehicles are not treated as errors;
        they lower ``completed_count`` and are excluded from the mean.
    """
    rows: list[dict] = []
    for demand_level, n in DEMAND_LEVELS.items():
        for rate in ADOPTION_RATES:
            rows.append(_run_cell(demand_level, n, rate, seed))
    return rows


def run_focused_presentation_experiment(
    *,
    policies: Iterable[str] = FOCUSED_POLICIES,
    adoption_rates: Iterable[float] = FOCUSED_ADOPTION_RATES,
    disruptions: Iterable[str] = FOCUSED_DISRUPTIONS,
    seeds: Iterable[int] = FOCUSED_SEEDS,
    horizon: int = FOCUSED_HORIZON,
) -> list[dict[str, Any]]:
    """Run the reproducible 6+4 presentation scenario matrix.

    The focused matrix is deliberately separate from the broader sustained
    routing-comparison grid. It crosses four routing policies, 0/50/100%
    adoption, no disruption or a halved primary capacity, and seeds 0--4.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    rows: list[dict[str, Any]] = []
    for policy in policies:
        if policy not in FOCUSED_POLICIES:
            raise ValueError(f"unknown focused routing policy: {policy!r}")
        for adoption_rate in adoption_rates:
            if adoption_rate not in FOCUSED_ADOPTION_RATES:
                raise ValueError(
                    f"focused adoption must be one of {FOCUSED_ADOPTION_RATES}, "
                    f"got {adoption_rate!r}"
                )
            for disruption in disruptions:
                if disruption not in FOCUSED_DISRUPTIONS:
                    raise ValueError(
                        f"unknown focused disruption: {disruption!r}"
                    )
                for seed in seeds:
                    rows.append(
                        _run_focused_scenario(
                            policy=policy,
                            adoption_rate=float(adoption_rate),
                            disruption=disruption,
                            seed=seed,
                            horizon=horizon,
                        )
                    )
    return rows


def _run_focused_scenario(
    *,
    policy: str,
    adoption_rate: float,
    disruption: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    graph = create_default_network()
    configure_focused_network(graph)

    vehicles = build_staggered_corridor_demand(graph)
    assign_navigation_adoption(vehicles, adoption_rate, seed=seed)
    if policy == "uninformed":
        for vehicle in vehicles:
            vehicle.uses_navigation_app = False
    if disruption == "capacity_reduction":
        reduce_road_capacity(
            graph, *DISRUPTED_ROAD, factor=REDUCED_CAPACITY_FACTOR
        )

    simulation = Simulation(
        graph,
        vehicles,
        routing_policy=(
            "coordinated"
            if policy == "coordinated"
            else "shared_navigation"
            if policy == "shared_navigation"
            else "decentralized"
        ),
    )
    peak_occupancy = maximum_occupancy(graph)
    peak_congestion = network_congestion_score(graph)
    steps = 0
    while steps < horizon and (
        simulation.active or simulation._has_pending_or_future()
    ):
        simulation.step()
        peak_occupancy = max(peak_occupancy, maximum_occupancy(graph))
        peak_congestion = max(peak_congestion, network_congestion_score(graph))
        steps += 1

    route, common_count, common_share = most_common_route(vehicles)
    primary_route = (0, 1, 2)
    assigned = sum(1 for vehicle in vehicles if vehicle.route is not None)
    primary_count = sum(tuple(vehicle.route or ()) == primary_route for vehicle in vehicles)
    completion_count = completed_count(vehicles)
    return {
        "scenario_id": (
            f"policy-{policy}__adoption-{adoption_rate:g}"
            f"__disruption-{disruption}__seed-{seed}"
        ),
        "policy": policy,
        "adoption_rate": adoption_rate,
        "nav_count": sum(vehicle.uses_navigation_app for vehicle in vehicles),
        "disruption": disruption,
        "disruption_factor": (
            REDUCED_CAPACITY_FACTOR if disruption == "capacity_reduction" else None
        ),
        "seed": seed,
        "horizon": horizon,
        "vehicle_count": len(vehicles),
        "completed_count": completion_count,
        "completion_rate": completion_count / len(vehicles),
        "mean_journey_time": mean_journey_time(vehicles),
        "final_completion_timestep": final_completion_timestep(vehicles),
        "makespan": makespan(vehicles),
        "maximum_occupancy": peak_occupancy,
        "route_diversity": route_diversity(vehicles),
        "most_common_route": route,
        "most_common_route_count": common_count,
        "most_common_route_share": common_share,
        "primary_route_count": primary_count,
        "primary_route_share": primary_count / assigned if assigned else 0.0,
        "peak_network_congestion": peak_congestion,
    }


def _run_cell(demand_level: str, n: int, rate: float, seed: int) -> dict:
    """Run one demand × adoption cell and return its metric row."""
    graph = create_default_network()
    vehicles = build_corridor_demand(graph, n)
    assign_navigation_adoption(vehicles, rate, seed=seed)
    nav_count = sum(1 for vehicle in vehicles if vehicle.uses_navigation_app)
    sim = Simulation(graph, vehicles)
    sim.run(until=GRID_HORIZON)
    return {
        "demand_level": demand_level,
        "n": n,
        "adoption_rate": rate,
        "nav_count": nav_count,
        "seed": seed,
        "completed_count": completed_count(sim.vehicles),
        "mean_journey_time": mean_journey_time(sim.vehicles),
    }


def run_routing_comparison(
    *,
    seeds: Iterable[int] = COMPARISON_SEEDS,
    horizon: int = COMPARISON_HORIZON,
    wave_interval: int = WAVE_INTERVAL,
    disruption_step: int | None = None,
    restore_step: int | None = None,
    routing_policies: Iterable[str] = ROUTING_POLICIES,
    demand_levels: dict[str, int] | None = None,
    adoption_rates: Iterable[float] = ADOPTION_RATES,
    disruptions: Iterable[str] = DISRUPTION_TYPES,
) -> dict[str, list[dict[str, Any]]]:
    """Run the routing-comparison matrix and return raw summary rows.

    The returned mapping has ``summary`` and ``congestion`` lists. No files
    are written here so callers can test and analyse the exact raw rows before
    handing them to a thin output script. The default configuration is the
    complete 675-scenario matrix from the routing-comparison specification.

    ``horizon`` is an inclusive timestep index: a horizon of 100 produces
    congestion observations for timesteps 0 through 100 and releases waves at
    steps 0, 10, ..., 100.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if wave_interval < 1:
        raise ValueError(f"wave_interval must be >= 1, got {wave_interval}")
    # Keep short test/demo horizons useful while retaining the full
    # experiment's 30/60 schedule. Explicit caller values always win.
    if disruption_step is None:
        disruption_step = (
            horizon // 4 if horizon < 2 * RESTORE_STEP else DISRUPTION_STEP
        )
    if restore_step is None:
        restore_step = (
            (3 * horizon) // 4 if horizon < 2 * RESTORE_STEP else RESTORE_STEP
        )
    if not 0 <= disruption_step <= restore_step <= horizon:
        raise ValueError("disruption and restore steps must lie within the horizon")

    levels = EXPERIMENT_DEMAND_LEVELS if demand_levels is None else demand_levels
    summary_rows: list[dict[str, Any]] = []
    congestion_rows: list[dict[str, Any]] = []
    for policy in routing_policies:
        if policy not in ROUTING_POLICIES:
            raise ValueError(f"unknown routing policy: {policy!r}")
        for demand_level, vehicles_per_wave in levels.items():
            for adoption_rate in adoption_rates:
                for disruption in disruptions:
                    if disruption not in DISRUPTION_TYPES:
                        raise ValueError(f"unknown disruption: {disruption!r}")
                    for seed in seeds:
                        summary, congestion = _run_routing_scenario(
                            policy=policy,
                            demand_level=demand_level,
                            vehicles_per_wave=vehicles_per_wave,
                            adoption_rate=adoption_rate,
                            disruption=disruption,
                            seed=seed,
                            horizon=horizon,
                            wave_interval=wave_interval,
                            disruption_step=disruption_step,
                            restore_step=restore_step,
                        )
                        summary_rows.append(summary)
                        congestion_rows.extend(congestion)
    return {"summary": summary_rows, "congestion": congestion_rows}


# Descriptive aliases make the public runner easy to find without creating a
# second implementation or changing the existing demand/adoption runner.
run_routing_comparison_grid = run_routing_comparison
run_routing_comparison_experiment = run_routing_comparison


def _run_routing_scenario(
    *,
    policy: str,
    demand_level: str,
    vehicles_per_wave: int,
    adoption_rate: float,
    disruption: str,
    seed: int,
    horizon: int,
    wave_interval: int,
    disruption_step: int,
    restore_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_id = _scenario_id(
        policy, demand_level, adoption_rate, disruption, seed
    )
    graph = create_default_network()
    # Calibrate only the experiment network: the top corridor is deliberately
    # capacity-constrained while the lower corridor remains an available
    # alternative. This makes route choice consequential without changing the
    # baseline model or its validation fixtures.
    for edge in ((0, 1), (1, 0), (1, 2), (2, 1)):
        graph[edge[0]][edge[1]]["capacity"] = EXPERIMENT_PRIMARY_CAPACITY
    pristine = {
        (u, v): copy.deepcopy(attrs)
        for u, v, attrs in graph.edges(data=True)
    }
    vehicles: list[Vehicle] = []
    vehicle_number = 0
    for start_time in range(
        0, min(horizon, DEMAND_RELEASE_END) + 1, wave_interval
    ):
        wave = build_corridor_demand(graph, vehicles_per_wave)
        for vehicle in wave:
            vehicle.vehicle_id = f"{scenario_id}-vehicle-{vehicle_number}"
            vehicle.start_time = start_time
            vehicles.append(vehicle)
            vehicle_number += 1

    assign_navigation_adoption(vehicles, adoption_rate, seed=seed)
    if policy == "static":
        for vehicle in vehicles:
            vehicle.uses_navigation_app = False

    simulation = Simulation(
        graph,
        vehicles,
        routing_policy=(
            "coordinated"
            if policy == "coordinated"
            else "shared_navigation"
            if policy == "shared_navigation"
            else "decentralized"
        ),
    )
    free_flow_od_time = _pristine_free_flow_journey_time(graph)
    congestion_scores: list[float] = []
    congestion_rows: list[dict[str, Any]] = []
    for timestep in range(horizon + 1):
        _apply_scheduled_event(
            graph,
            pristine,
            disruption=disruption,
            timestep=timestep,
            disruption_step=disruption_step,
            restore_step=restore_step,
        )
        simulation.step()
        score = network_congestion_score(graph)
        congestion_scores.append(score)
        for u, v, attrs in graph.edges(data=True):
            congestion_rows.append(
                {
                    **_configuration_fields(
                        scenario_id=scenario_id,
                        policy=policy,
                        demand_level=demand_level,
                        vehicles_per_wave=vehicles_per_wave,
                        adoption_rate=adoption_rate,
                        nav_count=sum(
                            vehicle.uses_navigation_app for vehicle in vehicles
                        ),
                        disruption=disruption,
                        seed=seed,
                        horizon=horizon,
                    ),
                    "timestep": timestep,
                    "edge": (u, v),
                    "edge_u": u,
                    "edge_v": v,
                    "occupancy": attrs["occupancy"],
                    "capacity": attrs["capacity"],
                    "road_congestion": attrs["occupancy"] / attrs["capacity"],
                    "network_congestion_score": score,
                }
            )

    durations = [
        duration
        for vehicle in vehicles
        if (duration := journey_time(vehicle)) is not None
    ]
    recovery = _recovery_fields(
        congestion_scores,
        disruption=disruption,
        disruption_step=disruption_step,
        restore_step=restore_step,
        horizon=horizon,
        free_flow_od_time=free_flow_od_time,
    )
    summary = {
        **_configuration_fields(
            scenario_id=scenario_id,
            policy=policy,
            demand_level=demand_level,
            vehicles_per_wave=vehicles_per_wave,
            adoption_rate=adoption_rate,
            nav_count=sum(vehicle.uses_navigation_app for vehicle in vehicles),
            disruption=disruption,
            seed=seed,
            horizon=horizon,
        ),
        "completed_journeys": len(durations),
        "mean_journey_time": _mean(durations),
        "median_journey_time": _percentile(durations, 50),
        "p95_journey_time": _percentile(durations, 95),
        "total_network_delay": sum(
            duration - free_flow_od_time for duration in durations
        ),
        "peak_network_congestion": max(congestion_scores),
        **recovery,
    }
    return summary, congestion_rows


def _configuration_fields(**kwargs: Any) -> dict[str, Any]:
    fields = dict(kwargs)
    fields["disruption_factor"] = (
        REDUCED_CAPACITY_FACTOR
        if fields["disruption"] == "capacity_reduction"
        else None
    )
    fields["disrupted_road"] = DISRUPTED_ROAD
    return fields


def _pristine_free_flow_journey_time(graph) -> float:
    """Measure the baseline using the simulation's discrete-time semantics."""
    baseline_graph = copy.deepcopy(graph)
    baseline_vehicle = Vehicle.create_for_network(
        baseline_graph, vehicle_id="free-flow-baseline", origin=0, destination=2
    )
    baseline_simulation = Simulation(baseline_graph, [baseline_vehicle])
    baseline_simulation.run(until=100)
    duration = journey_time(baseline_vehicle)
    if duration is None:
        raise ValueError("pristine baseline vehicle did not complete")
    return float(duration)


def _scenario_id(
    policy: str, demand_level: str, adoption_rate: float, disruption: str, seed: int
) -> str:
    rate = f"{float(adoption_rate):g}"
    return (
        f"policy-{policy}__demand-{demand_level}__adoption-{rate}"
        f"__disruption-{disruption}__seed-{seed}"
    )


def _apply_scheduled_event(
    graph,
    pristine: dict[tuple[Any, Any], dict[str, Any]],
    *,
    disruption: str,
    timestep: int,
    disruption_step: int,
    restore_step: int,
) -> None:
    if disruption != "none" and timestep == disruption_step:
        if disruption == "capacity_reduction":
            reduce_road_capacity(
                graph, *DISRUPTED_ROAD, factor=REDUCED_CAPACITY_FACTOR
            )
        elif disruption == "closure":
            # Retain closed edges during live experiments so vehicles that
            # entered just before the event can finish their route safely.
            # The routing layer excludes edges marked ``closed``.
            close_road(graph, *DISRUPTED_ROAD, remove_edges=False)
    if disruption != "none" and timestep == restore_step:
        u, v = DISRUPTED_ROAD
        for a, b in ((u, v), (v, u)):
            occupancy = graph[a].get(b, {}).get("occupancy", 0)
            restored = copy.deepcopy(pristine[(a, b)])
            restored["occupancy"] = occupancy
            restored.pop("closed", None)
            graph.add_edge(a, b, **restored)


def _recovery_fields(
    congestion_scores: list[float],
    *,
    disruption: str,
    disruption_step: int,
    restore_step: int,
    horizon: int,
    free_flow_od_time: float,
) -> dict[str, Any]:
    if disruption == "none":
        return {
            "recovery_time": None,
            "recovered": None,
            "recovery_censored": False,
            "recovery_status": "not_applicable",
        }
    result = congestion_recovery(
        congestion_scores,
        disruption_step,
        restore_step,
        W=max(1, math.ceil(4 * free_flow_od_time)),
        K=max(1, math.ceil(2 * free_flow_od_time)),
        horizon=horizon,
    )
    if result["recovered"]:
        status = "recovered"
    elif result["censored"]:
        status = "censored"
    else:
        status = "no_excursion"
    return {
        "recovery_time": result["recovery_time"],
        "recovered": result["recovered"],
        "recovery_censored": result["censored"],
        "recovery_status": status,
    }


def _mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float | int], percentile: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no values."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
