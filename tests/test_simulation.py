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


def _diverging_weight_graph() -> nx.DiGraph:
    """Hand-built graph where free-flow and live-time shortest paths differ.

    Free-flow prefers A→B→C (cost 2) over direct A→C (cost 10).
    Live travel times reverse that: A→C costs 1; A→B→C costs 10.
    """
    graph = nx.DiGraph()
    graph.add_edge(
        "A", "C",
        free_flow_time=10.0,
        capacity=10,
        occupancy=0,
        current_travel_time=1.0,
    )
    graph.add_edge(
        "A", "B",
        free_flow_time=1.0,
        capacity=10,
        occupancy=0,
        current_travel_time=5.0,
    )
    graph.add_edge(
        "B", "C",
        free_flow_time=1.0,
        capacity=10,
        occupancy=0,
        current_travel_time=5.0,
    )
    return graph


def test_uninformed_auto_route_uses_free_flow_shortest_path():
    graph = _diverging_weight_graph()
    vehicle = Vehicle(
        vehicle_id="uninformed",
        origin="A",
        destination="C",
        start_time=0,
        uses_navigation_app=False,
    )
    assert vehicle.route is None

    sim = Simulation(graph, [vehicle])
    sim.step()

    assert vehicle.route == ["A", "B", "C"]


def test_nav_app_auto_route_uses_live_travel_time_shortest_path():
    graph = _diverging_weight_graph()
    vehicle = Vehicle(
        vehicle_id="nav",
        origin="A",
        destination="C",
        start_time=0,
        uses_navigation_app=True,
    )
    assert vehicle.route is None

    sim = Simulation(graph, [vehicle])
    sim.step()

    assert vehicle.route == ["A", "C"]


def test_shared_navigation_assigns_identical_routes_from_one_snapshot():
    """Same-step navigation users in one OD group share one live route."""
    graph = _diverging_weight_graph()
    vehicles = [
        Vehicle(
            vehicle_id=f"shared-{i}",
            origin="A",
            destination="C",
            uses_navigation_app=True,
        )
        for i in range(3)
    ]
    # With the live snapshot, A→C is shortest for all three. Sequential
    # selfish entry would make later users see a different edge cost.
    sim = Simulation(graph, vehicles, routing_policy="shared_navigation")
    sim.step()

    assert [vehicle.route for vehicle in vehicles] == [["A", "C"]] * 3


def test_shared_navigation_keeps_uninformed_vehicles_on_static_routes():
    graph = _diverging_weight_graph()
    nav = Vehicle("nav", "A", "C", uses_navigation_app=True)
    uninformed = Vehicle("static", "A", "C", uses_navigation_app=False)
    sim = Simulation(graph, [nav, uninformed], routing_policy="shared_navigation")
    sim.step()

    assert nav.route == ["A", "C"]
    assert uninformed.route == ["A", "B", "C"]


def test_shared_navigation_excludes_closed_roads_for_new_vehicles():
    graph = _diverging_weight_graph()
    graph["A"]["C"]["closed"] = True
    vehicles = [
        Vehicle(f"shared-{i}", "A", "C", uses_navigation_app=True)
        for i in range(2)
    ]
    sim = Simulation(graph, vehicles, routing_policy="shared_navigation")
    sim.step()

    assert [vehicle.route for vehicle in vehicles] == [["A", "B", "C"]] * 2


def test_shared_navigation_high_demand_concentrates_on_capacity_limited_corridor():
    from src.adoption import assign_navigation_adoption
    from src.demand import build_corridor_demand

    graph = create_default_network()
    for u, v in ((0, 1), (1, 2), (1, 0), (2, 1)):
        graph[u][v]["capacity"] = 2
    vehicles = build_corridor_demand(graph, 10)
    assign_navigation_adoption(vehicles, 1.0, seed=0)
    sim = Simulation(graph, vehicles, routing_policy="shared_navigation")
    sim.step()

    assert all(vehicle.route == [0, 1, 2] for vehicle in vehicles)
    assert graph[0][1]["occupancy"] > graph[0][1]["capacity"]


def test_pre_set_route_wins_over_nav_app_flag():
    graph = _diverging_weight_graph()
    vehicle = Vehicle(
        vehicle_id="preset",
        origin="A",
        destination="C",
        start_time=0,
        uses_navigation_app=True,
    )
    forced = ["A", "B", "C"]
    vehicle.set_route(forced, graph)

    sim = Simulation(graph, [vehicle])
    sim.step()

    assert vehicle.route == forced


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


# ---------------------------------------------------------------------------
# Coordinated routing policy — Primary Seam
# ---------------------------------------------------------------------------


def _halved_corridor_scenario():
    """15-vehicle high-demand scenario with road 1-2 halved."""
    from src.demand import build_high_demand
    from src.disruption import reduce_road_capacity

    graph = create_default_network()
    vehicles = build_high_demand(graph)
    reduce_road_capacity(graph, 1, 2, factor=0.5)
    return graph, vehicles


def test_coordinated_routing_reduces_total_journey_time_on_halved_corridor():
    """Coordinated routing achieves total journey time 53 vs selfish 58."""
    from src.adoption import assign_navigation_adoption

    # Selfish baseline.
    graph_s, vehicles_s = _halved_corridor_scenario()
    assign_navigation_adoption(vehicles_s, 1.0, seed=0)
    sim_s = Simulation(graph_s, vehicles_s)
    sim_s.run(until=200)
    selfish_total = sum(v.completion_time - v.start_time for v in vehicles_s)

    # Coordinated.
    graph_c, vehicles_c = _halved_corridor_scenario()
    sim_c = Simulation(graph_c, vehicles_c, routing_policy="coordinated")
    sim_c.run(until=200)
    coordinated_total = sum(v.completion_time - v.start_time for v in vehicles_c)

    assert selfish_total == 58
    assert coordinated_total == 53
    assert coordinated_total < selfish_total


def test_coordinated_routing_diverts_last_two_vehicles_to_lower_corridor():
    """The last two vehicles should take the lower-corridor route 0→3→4→5→2."""
    graph, vehicles = _halved_corridor_scenario()
    sim = Simulation(graph, vehicles, routing_policy="coordinated")
    sim.run(until=200)

    lower_route = [0, 3, 4, 5, 2]
    diverted = [v for v in vehicles if v.route == lower_route]
    # Exactly the last two vehicles (demand-13 and demand-14) should be diverted.
    assert len(diverted) == 2
    diverted_ids = {v.vehicle_id for v in diverted}
    assert diverted_ids == {"demand-13", "demand-14"}


def test_coordinated_routing_policy_attribute_is_stored():
    graph = create_default_network()
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=2, start_time=0)
    sim = Simulation(graph, [vehicle], routing_policy="coordinated")
    assert sim.routing_policy == "coordinated"


def test_coordinated_routing_all_vehicles_complete():
    graph, vehicles = _halved_corridor_scenario()
    sim = Simulation(graph, vehicles, routing_policy="coordinated")
    sim.run(until=200)
    for v in vehicles:
        assert v.completion_time is not None


def test_coordinated_routing_raises_value_error_when_no_path():
    """ValueError raised when a vehicle has no path to its destination."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, free_flow_time=1.0, capacity=10, occupancy=0, current_travel_time=1.0)
    graph.add_node(2)  # disconnected
    vehicle = Vehicle(vehicle_id="v0", origin=0, destination=2, start_time=0)

    sim = Simulation(graph, [vehicle], routing_policy="coordinated")
    with pytest.raises(ValueError, match="no path exists"):
        sim.step()


def test_coordinated_routing_raises_value_error_when_search_space_too_large():
    """ValueError raised when the number of route combinations exceeds 32 768.

    Graph has 4 simple paths (1 direct + 3 via intermediate nodes).
    stars-and-bars: C(57+4-1, 4-1) = C(60, 3) = 34220 > 32768.
    """
    graph = nx.DiGraph()
    attrs = {"free_flow_time": 1.0, "capacity": 10, "occupancy": 0, "current_travel_time": 1.0}
    # 3 intermediate nodes (10, 11, 12) give 3 indirect routes; plus a direct 0→2.
    graph.add_edge(0, 2, **attrs)
    for mid in (10, 11, 12):
        graph.add_edge(0, mid, **attrs)
        graph.add_edge(mid, 2, **attrs)
    # 57 vehicles on 0→2: C(60, 3) = 34220 > 32768.
    vehicles = [
        Vehicle(vehicle_id=f"v{i}", origin=0, destination=2, start_time=0)
        for i in range(57)
    ]
    sim = Simulation(graph, vehicles, routing_policy="coordinated")
    with pytest.raises(ValueError, match="32.768|32768"):
        sim.step()
