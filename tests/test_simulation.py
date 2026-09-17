"""Tests for the discrete simulation clock and vehicle movement."""

import networkx as nx
import pytest

from src.network import create_default_network
from src.simulation import Simulation
from src.vehicle import Vehicle


# ---------------------------------------------------------------------------
# Smoke on the default network — Primary Seam
# ---------------------------------------------------------------------------


def test_smoke_one_vehicle_travels_origin_to_destination():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=2, start_time=0)
    sim = Simulation(graph, [vehicle])

    completed = sim.run(until=100)

    assert completed == [vehicle]
    assert vehicle in sim.completed
    assert vehicle not in sim.active
    assert vehicle.current_position == 2
    assert isinstance(vehicle.completion_time, int)
    assert vehicle.completion_time >= vehicle.start_time
    for _u, _v, attrs in graph.edges(data=True):
        assert attrs["occupancy"] == 0


# ---------------------------------------------------------------------------
# step() observable progression — Secondary Seam
# ---------------------------------------------------------------------------


def test_current_step_starts_at_zero_and_increments_after_step():
    graph = create_default_network()
    sim = Simulation(graph, [])

    assert sim.current_step == 0
    sim.step()
    assert sim.current_step == 1


def test_vehicle_with_later_start_time_not_active_before_release():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="late", origin=0, destination=1, start_time=2)
    sim = Simulation(graph, [vehicle])

    sim.step()  # step 0
    assert vehicle not in sim.active
    assert vehicle not in sim.completed
    assert sim.current_step == 1

    sim.step()  # step 1
    assert vehicle not in sim.active
    assert sim.current_step == 2

    sim.step()  # step 2 — release
    # Free-flow edge time 1: enter + consume in same step → may complete at dest.
    assert vehicle.start_time == 2
    assert vehicle in sim.active or vehicle in sim.completed


def test_position_hops_along_known_route():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="hop", origin=0, destination=2, start_time=0)
    vehicle.set_route([0, 1, 2], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()  # enter 0→1, consume dwell → leave to 1, enter 1→2
    assert vehicle.current_position == 1
    assert vehicle in sim.active
    assert vehicle.completion_time is None

    sim.step()  # consume 1→2 → arrive at 2 and complete
    assert vehicle.current_position == 2
    assert vehicle in sim.completed
    assert vehicle.completion_time == 1


def test_occupancy_rises_while_in_transit_and_returns_to_zero():
    graph = nx.DiGraph()
    # Travel time 2 so the vehicle stays on the edge after the first step.
    graph.add_edge(0, 1, free_flow_time=2.0, capacity=10, occupancy=0, current_travel_time=2.0)
    vehicle = Vehicle(vehicle_id="occ", origin=0, destination=1, start_time=0)
    vehicle.set_route([0, 1], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()
    assert graph[0][1]["occupancy"] == 1
    assert vehicle in sim.active
    assert vehicle.current_position == 0  # still on edge; not yet arrived

    sim.step()
    assert vehicle.current_position == 1
    assert vehicle in sim.completed
    assert graph[0][1]["occupancy"] == 0


def test_unrouted_vehicle_is_auto_routed_on_entry():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="auto", origin=0, destination=2, start_time=0)
    assert vehicle.route is None

    sim = Simulation(graph, [vehicle])
    sim.step()

    assert vehicle.route is not None
    assert vehicle.route[0] == 0
    assert vehicle.route[-1] == 2


def test_pre_routed_vehicle_keeps_caller_assigned_route():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="pre", origin=0, destination=2, start_time=0)
    forced = [0, 1, 2]
    vehicle.set_route(forced, graph)

    sim = Simulation(graph, [vehicle])
    sim.step()

    assert vehicle.route == forced


def test_run_stops_early_when_network_idle():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="early", origin=0, destination=1, start_time=0)
    sim = Simulation(graph, [vehicle])

    completed = sim.run(until=1000)

    assert completed == [vehicle]
    # Single free-flow edge of time 1 completes during step 0; clock advances to 1.
    assert sim.current_step == 1
    assert sim.current_step < 1000


def test_pending_vehicles_outside_active_and_completed():
    graph = create_default_network()
    pending = Vehicle(vehicle_id="p", origin=0, destination=1, start_time=5)
    sim = Simulation(graph, [pending])

    assert pending not in sim.active
    assert pending not in sim.completed
    assert pending in sim.vehicles


def test_start_time_beyond_horizon_never_enters():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="far", origin=0, destination=1, start_time=50)
    sim = Simulation(graph, [vehicle])

    completed = sim.run(until=10)

    assert completed == []
    assert vehicle not in sim.completed
    assert vehicle not in sim.active
    assert vehicle.completion_time is None
    assert vehicle.current_position == 0


def test_travel_times_refresh_after_occupancy_change():
    graph = nx.DiGraph()
    graph.add_edge(
        0, 1,
        free_flow_time=1.0,
        capacity=1,
        occupancy=0,
        current_travel_time=1.0,
    )
    # Dwell 2 so vehicle remains on edge after step 0 for occupancy observation.
    graph.add_edge(
        "A", "B",
        free_flow_time=2.0,
        capacity=1,
        occupancy=0,
        current_travel_time=2.0,
    )
    vehicle = Vehicle(vehicle_id="bpr", origin="A", destination="B", start_time=0)
    vehicle.set_route(["A", "B"], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()
    # Occupancy 1 on capacity 1 → BPR: t = 2 * (1 + 0.15 * 1^4) = 2.3
    assert graph["A"]["B"]["occupancy"] == 1
    assert graph["A"]["B"]["current_travel_time"] == pytest.approx(2.3)


# ---------------------------------------------------------------------------
# Constructor / guardrails — Tertiary Seam
# ---------------------------------------------------------------------------


def test_until_less_than_one_raises_value_error():
    graph = create_default_network()
    sim = Simulation(graph, [])

    with pytest.raises(ValueError, match="until"):
        sim.run(until=0)

    with pytest.raises(ValueError, match="until"):
        sim.run(until=-3)


def test_already_completed_vehicle_rejected_at_construction():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="done", origin=0, destination=1, start_time=0)
    vehicle.update_position(1, graph)
    vehicle.complete_journey(0)

    with pytest.raises(ValueError, match="already completed"):
        Simulation(graph, [vehicle])


def test_missing_edge_on_route_raises_when_entering():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="bad", origin=0, destination=2, start_time=0)
    # Nodes exist but edge 0→2 does not on the default network.
    vehicle.route = [0, 2]

    sim = Simulation(graph, [vehicle])
    with pytest.raises(ValueError, match="missing directed edge"):
        sim.step()


def test_empty_vehicle_list_is_allowed():
    graph = create_default_network()
    sim = Simulation(graph, [])

    completed = sim.run(until=5)

    assert completed == []
    assert sim.active == []
    assert sim.completed == []
    # Idle immediately — no steps needed.
    assert sim.current_step == 0
