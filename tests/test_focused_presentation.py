"""Focused 6+4 presentation scenario regression tests."""

from src.demand import build_staggered_corridor_demand
from src.experiments import _run_focused_scenario
from src.network import create_default_network


def test_staggered_demand_has_six_releases_at_zero_and_four_at_one():
    vehicles = build_staggered_corridor_demand(create_default_network())

    assert len(vehicles) == 10
    assert [vehicle.start_time for vehicle in vehicles] == [0] * 6 + [1] * 4


def test_focused_shared_navigation_herds_more_than_selfish_entry():
    selfish = _run_focused_scenario(
        policy="selfish", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )
    shared = _run_focused_scenario(
        policy="shared_navigation",
        adoption_rate=1.0,
        disruption="none",
        seed=0,
        horizon=100,
    )

    assert selfish["completed_count"] == shared["completed_count"] == 10
    assert shared["route_diversity"] < selfish["route_diversity"]
    assert shared["mean_journey_time"] > selfish["mean_journey_time"]
    assert shared["maximum_occupancy"] > 2


def test_focused_route_metrics_are_seed_reproducible():
    first = _run_focused_scenario(
        policy="shared_navigation", adoption_rate=1.0,
        disruption="capacity_reduction", seed=0, horizon=100,
    )
    second = _run_focused_scenario(
        policy="shared_navigation", adoption_rate=1.0,
        disruption="capacity_reduction", seed=0, horizon=100,
    )

    assert first == second
    assert first["final_completion_timestep"] == first["makespan"]
    assert first["primary_route_share"] == 1.0
