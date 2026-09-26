"""Public in-transit snapshot: who is on which edge, and that reads are pure."""

import math

import networkx as nx

from src.network import create_default_network
from src.simulation import InTransitRecord, Simulation
from src.vehicle import Vehicle


def _slow_edge_graph() -> nx.DiGraph:
    """One edge whose free-flow dwell keeps a vehicle aboard after entry."""
    graph = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        free_flow_time=2.0,
        capacity=10,
        occupancy=0,
        current_travel_time=2.0,
    )
    return graph


def test_snapshot_empty_before_first_step():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=2, start_time=0)
    sim = Simulation(graph, [vehicle])

    assert sim.current_step == 0
    assert sim.in_transit_snapshot() == ()


def test_later_release_absent_until_start_time():
    graph = _slow_edge_graph()
    vehicle = Vehicle(vehicle_id="late", origin=0, destination=1, start_time=2)
    vehicle.set_route([0, 1], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()
    assert sim.in_transit_snapshot() == ()
    sim.step()
    assert sim.current_step == 2
    assert sim.in_transit_snapshot() == ()

    sim.step()
    assert [record.vehicle_id for record in sim.in_transit_snapshot()] == ["late"]


def test_entry_step_records_assigned_dwell_and_one_consumed_tick():
    graph = _slow_edge_graph()
    vehicle = Vehicle(vehicle_id="occ", origin=0, destination=1, start_time=0)
    vehicle.set_route([0, 1], graph)
    expected_dwell = max(1, math.ceil(graph[0][1]["current_travel_time"]))
    sim = Simulation(graph, [vehicle])

    sim.step()

    snapshot = sim.in_transit_snapshot()
    assert snapshot == (
        InTransitRecord(
            vehicle_id="occ",
            edge=(0, 1),
            assigned_dwell=expected_dwell,
            steps_remaining=expected_dwell - 1,
        ),
    )
    assert vehicle in sim.active
    assert vehicle.current_position == 0
    assert vehicle.completion_time is None
    assert graph[0][1]["occupancy"] == 1

    occupancy = graph[0][1]["occupancy"]
    sim.in_transit_snapshot()
    assert graph[0][1]["occupancy"] == occupancy


def test_completing_step_removes_vehicle_from_snapshot():
    graph = _slow_edge_graph()
    vehicle = Vehicle(vehicle_id="occ", origin=0, destination=1, start_time=0)
    vehicle.set_route([0, 1], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()
    assert len(sim.in_transit_snapshot()) == 1

    sim.step()
    assert sim.in_transit_snapshot() == ()
    assert vehicle in sim.completed
    assert vehicle.current_position == 1
    assert graph[0][1]["occupancy"] == 0
    # Dwell of 2: entered on clock 0, finished while the clock still read 1.
    assert vehicle.completion_time == 1


def test_free_flow_hop_lands_on_next_edge_with_full_assigned_dwell():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="hop", origin=0, destination=2, start_time=0)
    vehicle.set_route([0, 1, 2], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()

    assert vehicle.current_position == 1
    assert vehicle in sim.active
    snapshot = sim.in_transit_snapshot()
    assert len(snapshot) == 1
    record = snapshot[0]
    assert record.vehicle_id == "hop"
    assert record.edge == (1, 2)
    assert record.assigned_dwell >= 1
    assert record.steps_remaining == record.assigned_dwell


def test_two_vehicles_on_same_edge_follow_active_order():
    graph = _slow_edge_graph()
    first = Vehicle(vehicle_id="a", origin=0, destination=1, start_time=0)
    second = Vehicle(vehicle_id="b", origin=0, destination=1, start_time=0)
    first.set_route([0, 1], graph)
    second.set_route([0, 1], graph)
    sim = Simulation(graph, [first, second])

    sim.step()

    snapshot = sim.in_transit_snapshot()
    assert [record.vehicle_id for record in snapshot] == [
        vehicle.vehicle_id for vehicle in sim.active
    ]
    assert [record.edge for record in snapshot] == [(0, 1), (0, 1)]
    assert len(snapshot) == 2


def test_snapshot_read_does_not_change_run_state():
    graph = _slow_edge_graph()
    vehicle = Vehicle(vehicle_id="occ", origin=0, destination=1, start_time=0)
    vehicle.set_route([0, 1], graph)
    sim = Simulation(graph, [vehicle])
    sim.step()

    clock = sim.current_step
    occupancy = graph[0][1]["occupancy"]
    route = list(vehicle.route)
    completion = vehicle.completion_time
    active = list(sim.active)
    completed = list(sim.completed)

    first = sim.in_transit_snapshot()
    second = sim.in_transit_snapshot()

    assert second == first
    assert sim.current_step == clock
    assert graph[0][1]["occupancy"] == occupancy
    assert vehicle.route == route
    assert vehicle.completion_time == completion
    assert sim.active == active
    assert sim.completed == completed


def test_known_arrival_and_occupancy_unchanged_by_stored_dwell():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="hop", origin=0, destination=2, start_time=0)
    vehicle.set_route([0, 1, 2], graph)
    sim = Simulation(graph, [vehicle])

    sim.step()
    assert vehicle.current_position == 1
    assert vehicle.completion_time is None
    assert any(attrs["occupancy"] == 1 for *_, attrs in graph.edges(data=True))

    sim.step()
    assert vehicle.current_position == 2
    assert vehicle.completion_time == 1
    assert sim.in_transit_snapshot() == ()
    for _u, _v, attrs in graph.edges(data=True):
        assert attrs["occupancy"] == 0
