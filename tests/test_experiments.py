"""Demand × adoption grid: row contract, completion, and seed repeatability."""

import pytest

from src.adoption import ADOPTION_RATES
from src.demand import DEMAND_LEVELS, build_corridor_demand
from src.experiments import run_demand_adoption_grid
from src.metrics import completed_count, mean_journey_time
from src.network import create_default_network
from src.simulation import Simulation

# Same horizon as the uninformed baseline validation.
BASELINE_HORIZON = 100
FREE_FLOW_JOURNEY_TIME = 1.0


def _uninformed_baseline(n: int):
    """Run a fresh uninformed corridor batch; no adoption helper."""
    graph = create_default_network()
    vehicles = build_corridor_demand(graph, n)
    sim = Simulation(graph, vehicles)
    sim.run(until=BASELINE_HORIZON)
    return sim


# ---------------------------------------------------------------------------
# Primary seam — run_demand_adoption_grid
# ---------------------------------------------------------------------------


def test_grid_has_fifteen_rows_in_demand_then_adoption_order():
    rows = run_demand_adoption_grid(seed=0)

    expected = [
        (level, n, rate)
        for level, n in DEMAND_LEVELS.items()
        for rate in ADOPTION_RATES
    ]
    assert len(rows) == 15
    assert [
        (row["demand_level"], row["n"], row["adoption_rate"]) for row in rows
    ] == expected


def test_each_row_records_seed_nav_count_and_completion():
    rows = run_demand_adoption_grid(seed=0)

    for row in rows:
        assert row["n"] == DEMAND_LEVELS[row["demand_level"]]
        assert row["seed"] == 0
        assert row["nav_count"] == round(row["n"] * row["adoption_rate"])
        assert row["completed_count"] == row["n"]
        assert row["mean_journey_time"] is not None


def test_seed_zero_calls_return_the_same_rows():
    first = run_demand_adoption_grid(seed=0)
    second = run_demand_adoption_grid(seed=0)
    assert first == second
    assert run_demand_adoption_grid() == first


def test_zero_adoption_matches_uninformed_baseline():
    rows = run_demand_adoption_grid(seed=0)
    zero_rate = [row for row in rows if row["adoption_rate"] == 0.0]
    assert [row["demand_level"] for row in zero_rate] == list(DEMAND_LEVELS)

    for row in zero_rate:
        sim = _uninformed_baseline(row["n"])
        assert row["completed_count"] == completed_count(sim.vehicles)
        assert row["mean_journey_time"] == mean_journey_time(sim.vehicles)


def test_low_demand_mean_journey_time_is_free_flow_at_every_rate():
    rows = run_demand_adoption_grid(seed=0)
    low = [row for row in rows if row["demand_level"] == "low"]

    assert [row["adoption_rate"] for row in low] == list(ADOPTION_RATES)
    # Banker's rounding on a one-vehicle fleet: navigator only at 0.75 and 1.
    assert [row["nav_count"] for row in low] == [0, 0, 0, 1, 1]
    assert all(row["mean_journey_time"] == FREE_FLOW_JOURNEY_TIME for row in low)


def test_medium_half_adoption_uses_bankers_rounding():
    rows = run_demand_adoption_grid(seed=0)
    medium_half = next(
        row
        for row in rows
        if row["demand_level"] == "medium" and row["adoption_rate"] == 0.5
    )
    assert medium_half["n"] == 5
    assert medium_half["nav_count"] == 2
    assert medium_half["mean_journey_time"] is not None


def test_invalid_seed_fails_through_adoption_helper():
    with pytest.raises(TypeError, match="seed must be an integer"):
        run_demand_adoption_grid(seed=1.5)  # type: ignore[arg-type]
