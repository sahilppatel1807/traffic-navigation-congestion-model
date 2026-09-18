"""Tests for journey-level and road-level metrics."""

import networkx as nx
import pytest

from src.metrics import (
    completed_count,
    journey_time,
    mean_journey_time,
    road_congestion,
    simulation_summary,
)
from src.network import create_default_network
from src.simulation import Simulation
from src.vehicle import Vehicle


# ---------------------------------------------------------------------------
# journey_time — primary seam
# ---------------------------------------------------------------------------


def test_journey_time_for_completed_vehicle():
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=2)
    vehicle.current_position = 1
    vehicle.complete_journey(7)

    assert journey_time(vehicle) == 5


def test_journey_time_zero_when_completes_same_step_as_start():
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=3)
    vehicle.current_position = 1
    vehicle.complete_journey(3)

    assert journey_time(vehicle) == 0


def test_journey_time_none_for_incomplete_vehicle():
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=0)

    assert journey_time(vehicle) is None
    assert vehicle.completion_time is None


def test_journey_time_raises_when_completion_before_start():
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=5)
    vehicle.completion_time = 3  # bypass complete_journey invariants

    with pytest.raises(ValueError, match="completion_time"):
        journey_time(vehicle)


def test_journey_time_raises_for_non_integer_completion_time():
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=0)
    vehicle.completion_time = 4.5  # type: ignore[assignment]

    with pytest.raises(TypeError, match="completion_time"):
        journey_time(vehicle)


# ---------------------------------------------------------------------------
# Aggregates — completed_count / mean_journey_time
# ---------------------------------------------------------------------------


def _completed(vehicle_id: str, start: int, completion: int) -> Vehicle:
    vehicle = Vehicle(
        vehicle_id=vehicle_id, origin=0, destination=1, start_time=start
    )
    vehicle.current_position = 1
    vehicle.complete_journey(completion)
    return vehicle


def test_completed_count_counts_only_finished_vehicles():
    finished_a = _completed("a", start=0, completion=4)
    finished_b = _completed("b", start=1, completion=3)
    pending = Vehicle(vehicle_id="c", origin=0, destination=1, start_time=0)

    assert completed_count([finished_a, pending, finished_b]) == 2
    assert completed_count([]) == 0
    assert completed_count([pending]) == 0


def test_mean_journey_time_excludes_incomplete_vehicles():
    finished_a = _completed("a", start=0, completion=4)  # duration 4
    finished_b = _completed("b", start=2, completion=4)  # duration 2
    pending = Vehicle(vehicle_id="c", origin=0, destination=1, start_time=0)

    assert mean_journey_time([finished_a, pending, finished_b]) == 3.0


def test_mean_journey_time_none_when_no_vehicles_complete():
    pending = Vehicle(vehicle_id="c", origin=0, destination=1, start_time=0)

    assert mean_journey_time([pending]) is None
    assert mean_journey_time([]) is None


# ---------------------------------------------------------------------------
# road_congestion — directed occupancy/capacity ratios
# ---------------------------------------------------------------------------


def test_road_congestion_ratio_and_cap_at_one():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=5)
    graph.add_edge(1, 2, capacity=4, occupancy=10)  # over capacity
    graph.add_edge(2, 0, capacity=8, occupancy=0)

    ratios = road_congestion(graph)

    assert ratios[(0, 1)] == 0.5
    assert ratios[(1, 2)] == 1.0
    assert ratios[(2, 0)] == 0.0
    assert set(ratios) == {(0, 1), (1, 2), (2, 0)}


def test_road_congestion_keys_are_directed_edges():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=2)
    graph.add_edge(1, 0, capacity=10, occupancy=8)

    ratios = road_congestion(graph)

    assert ratios[(0, 1)] == 0.2
    assert ratios[(1, 0)] == 0.8


def test_road_congestion_raises_for_non_positive_capacity():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=0, occupancy=1)

    with pytest.raises(ValueError, match="capacity"):
        road_congestion(graph)


def test_road_congestion_raises_for_negative_occupancy():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=-1)

    with pytest.raises(ValueError, match="occupancy"):
        road_congestion(graph)


def test_road_congestion_raises_for_non_numeric_capacity():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity="ten", occupancy=1)

    with pytest.raises(TypeError, match="capacity"):
        road_congestion(graph)


# ---------------------------------------------------------------------------
# simulation_summary — integrated snapshot
# ---------------------------------------------------------------------------


def test_simulation_summary_after_run():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=2, start_time=0)
    vehicle.set_route([0, 1, 2], graph)
    sim = Simulation(graph, [vehicle])
    sim.run(until=100)

    summary = simulation_summary(sim)

    assert summary["completed_count"] == 1
    assert summary["mean_journey_time"] == float(journey_time(vehicle))
    assert isinstance(summary["road_congestion"], dict)
    assert all(0.0 <= ratio <= 1.0 for ratio in summary["road_congestion"].values())
    # Network idle after completion — all edges empty.
    assert all(ratio == 0.0 for ratio in summary["road_congestion"].values())
    assert len(summary["road_congestion"]) == graph.number_of_edges()


def test_simulation_summary_does_not_mutate_state():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=1, start_time=0)
    vehicle.set_route([0, 1], graph)
    sim = Simulation(graph, [vehicle])
    sim.step()  # may complete on free-flow time 1

    step_before = sim.current_step
    completion_before = vehicle.completion_time
    occupancy_before = {
        (u, v): attrs["occupancy"] for u, v, attrs in graph.edges(data=True)
    }

    _ = simulation_summary(sim)

    assert sim.current_step == step_before
    assert vehicle.completion_time == completion_before
    occupancy_after = {
        (u, v): attrs["occupancy"] for u, v, attrs in graph.edges(data=True)
    }
    assert occupancy_after == occupancy_before


def test_simulation_summary_with_incomplete_and_complete_mix():
    finished = _completed("done", start=0, completion=5)
    pending = Vehicle(vehicle_id="wait", origin=0, destination=1, start_time=10)

    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=3, free_flow_time=1.0, current_travel_time=1.0)

    # Build a Simulation with only the pending vehicle (completed agents rejected
    # at construction), then compute aggregates from the mixed list directly and
    # via a hand-rolled summary shape for the finished agent on the same graph.
    sim = Simulation(graph, [pending])
    summary = simulation_summary(sim)

    assert summary["completed_count"] == 0
    assert summary["mean_journey_time"] is None
    assert summary["road_congestion"][(0, 1)] == 0.3

    assert completed_count([finished, pending]) == 1
    assert mean_journey_time([finished, pending]) == 5.0
