"""Demand × navigation-adoption experiment grid.

Crosses the named demand presets with the planned adoption fractions, runs
each cell on a fresh default network, and returns journey-metric rows. File
output lives in a thin script; this module does not write results.
"""

from __future__ import annotations

from src.adoption import ADOPTION_RATES, assign_navigation_adoption
from src.demand import DEMAND_LEVELS, build_corridor_demand
from src.metrics import completed_count, mean_journey_time
from src.network import create_default_network
from src.simulation import Simulation

# Same horizon as the uninformed baseline validation script.
GRID_HORIZON = 100


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
