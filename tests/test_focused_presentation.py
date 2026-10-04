"""Focused 6+4 presentation scenario regression tests.

These tests verify both the mechanism (adoption controls nav-user eligibility,
each release cohort's route behaviour) and the outcomes (journey-time ordering,
route diversity, herding contrast) for the focused presentation network.
"""

from src.adoption import assign_navigation_adoption
from src.demand import build_staggered_corridor_demand
from src.experiments import _run_focused_scenario, configure_focused_network
from src.network import create_default_network
from src.simulation import Simulation


def test_staggered_demand_has_six_releases_at_zero_and_four_at_one():
    vehicles = build_staggered_corridor_demand(create_default_network())

    assert len(vehicles) == 10
    assert [vehicle.start_time for vehicle in vehicles] == [0] * 6 + [1] * 4


# ---------------------------------------------------------------------------
# Acceptance criteria: adoption controls eligibility
# ---------------------------------------------------------------------------

def test_coordinated_zero_adoption_matches_uninformed_baseline():
    """Coordinated at 0% adoption must follow the uninformed baseline (not 7.2).

    When no vehicle uses the navigation app, the coordinated controller is
    bypassed entirely and all vehicles receive uninformed free-flow routes.
    """
    coord = _run_focused_scenario(
        policy="coordinated", adoption_rate=0.0, disruption="none", seed=0, horizon=100
    )
    uninformed = _run_focused_scenario(
        policy="uninformed", adoption_rate=0.0, disruption="none", seed=0, horizon=100
    )

    assert coord["mean_journey_time"] == uninformed["mean_journey_time"]
    assert coord["route_diversity"] == uninformed["route_diversity"]
    assert coord["primary_route_share"] == uninformed["primary_route_share"] == 1.0
    # Must NOT report the (previously erroneous) 7.2-step optimal result.
    assert coord["mean_journey_time"] != 7.2


def test_coordinated_full_adoption_produces_system_level_assignments():
    """Coordinated routing at 100% adoption can divert vehicles to the alternative."""
    coord = _run_focused_scenario(
        policy="coordinated", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )

    assert coord["completed_count"] == 10
    # Controller must produce at least one diverted vehicle.
    assert coord["primary_route_share"] < 1.0
    assert coord["route_diversity"] >= 2


# ---------------------------------------------------------------------------
# Acceptance criteria: mixed waves leave non-nav on free-flow routes
# ---------------------------------------------------------------------------

def test_mixed_coordinated_wave_non_nav_vehicles_get_free_flow_routes():
    """Non-navigation vehicles in a mixed coordinated wave receive free-flow routes.

    The free-flow shortest route in this network is the primary corridor
    0→1→2 (travel time 2 steps at free flow vs ~4 for the alternative).
    Non-nav vehicles must be on that primary route regardless of the
    coordinated controller's nav-user assignments.
    """
    graph = create_default_network()
    configure_focused_network(graph)
    vehicles = build_staggered_corridor_demand(graph)
    # 50% adoption: a mix of nav and non-nav in the same release wave.
    assign_navigation_adoption(vehicles, 0.5, seed=0)

    sim = Simulation(graph, vehicles, routing_policy="coordinated")
    sim.run(until=100)

    primary_route = [0, 1, 2]
    for vehicle in vehicles:
        if not vehicle.uses_navigation_app:
            assert vehicle.route == primary_route, (
                f"non-nav vehicle {vehicle.vehicle_id!r} expected free-flow route "
                f"{primary_route}, got {vehicle.route}"
            )


# ---------------------------------------------------------------------------
# Acceptance criteria: shared navigation cohort herding mechanism
# ---------------------------------------------------------------------------

def test_shared_navigation_assigns_same_route_within_each_release_cohort():
    """Vehicles in the same release wave get one synchronized recommendation.

    All 6 vehicles released at t=0 must share a single route, and all 4
    vehicles released at t=1 must share a single (possibly different) route.
    """
    graph = create_default_network()
    configure_focused_network(graph)
    vehicles = build_staggered_corridor_demand(graph)
    assign_navigation_adoption(vehicles, 1.0, seed=0)

    sim = Simulation(graph, vehicles, routing_policy="shared_navigation")
    sim.run(until=100)

    wave0_routes = set(tuple(v.route) for v in vehicles if v.start_time == 0)
    wave1_routes = set(tuple(v.route) for v in vehicles if v.start_time == 1)

    assert len(wave0_routes) == 1, (
        f"Wave 0 vehicles should all get the same route; got {len(wave0_routes)} distinct routes"
    )
    assert len(wave1_routes) == 1, (
        f"Wave 1 vehicles should all get the same route; got {len(wave1_routes)} distinct routes"
    )


def test_shared_navigation_has_at_least_one_diverted_vehicle():
    """At 100% adoption, shared navigation must divert at least one vehicle."""
    graph = create_default_network()
    configure_focused_network(graph)
    vehicles = build_staggered_corridor_demand(graph)
    assign_navigation_adoption(vehicles, 1.0, seed=0)

    sim = Simulation(graph, vehicles, routing_policy="shared_navigation")
    sim.run(until=100)

    primary_route = (0, 1, 2)
    diverted = [v for v in vehicles if tuple(v.route or ()) != primary_route]
    assert len(diverted) >= 1, "shared navigation must divert at least one vehicle"


def test_shared_navigation_differs_from_uninformed_routing():
    """Shared navigation must not reproduce the all-primary uninformed result."""
    shared = _run_focused_scenario(
        policy="shared_navigation", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )
    uninformed = _run_focused_scenario(
        policy="uninformed", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )

    # Shared navigation should have lower primary-route share (some vehicles diverted).
    assert shared["primary_route_share"] < uninformed["primary_route_share"]


# ---------------------------------------------------------------------------
# Acceptance criteria: journey-time and diversity ordering
# ---------------------------------------------------------------------------

def test_shared_navigation_has_higher_mean_journey_time_than_selfish():
    """Shared navigation herds, causing higher mean journey time than selfish routing."""
    selfish = _run_focused_scenario(
        policy="selfish", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )
    shared = _run_focused_scenario(
        policy="shared_navigation", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )

    assert selfish["completed_count"] == shared["completed_count"] == 10
    assert shared["mean_journey_time"] > selfish["mean_journey_time"]
    assert shared["maximum_occupancy"] > 2


def test_shared_navigation_lower_within_cohort_diversity_than_selfish():
    """Shared navigation gives each cohort one route; selfish splits within cohorts.

    Although total route diversity across all vehicles may be equal, shared
    navigation minimises per-wave diversity to 1 while selfish routing
    produces 2 distinct routes within each wave.
    """
    # --- Shared navigation ---
    graph_shared = create_default_network()
    configure_focused_network(graph_shared)
    vehicles_shared = build_staggered_corridor_demand(graph_shared)
    assign_navigation_adoption(vehicles_shared, 1.0, seed=0)
    sim_shared = Simulation(graph_shared, vehicles_shared, routing_policy="shared_navigation")
    sim_shared.run(until=100)

    shared_wave0_diversity = len({tuple(v.route) for v in vehicles_shared if v.start_time == 0})
    shared_wave1_diversity = len({tuple(v.route) for v in vehicles_shared if v.start_time == 1})

    # --- Selfish routing ---
    graph_selfish = create_default_network()
    configure_focused_network(graph_selfish)
    vehicles_selfish = build_staggered_corridor_demand(graph_selfish)
    assign_navigation_adoption(vehicles_selfish, 1.0, seed=0)
    sim_selfish = Simulation(graph_selfish, vehicles_selfish, routing_policy="decentralized")
    sim_selfish.run(until=100)

    selfish_wave0_diversity = len({tuple(v.route) for v in vehicles_selfish if v.start_time == 0})
    selfish_wave1_diversity = len({tuple(v.route) for v in vehicles_selfish if v.start_time == 1})

    # Each shared cohort has exactly one route; selfish cohorts split.
    assert shared_wave0_diversity == 1
    assert shared_wave1_diversity == 1
    assert selfish_wave0_diversity > shared_wave0_diversity
    assert selfish_wave1_diversity > shared_wave1_diversity


def test_selfish_has_lower_mean_journey_time_than_shared_navigation():
    """Selfish routing has lower mean journey time than shared navigation."""
    selfish = _run_focused_scenario(
        policy="selfish", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )
    shared = _run_focused_scenario(
        policy="shared_navigation", adoption_rate=1.0, disruption="none", seed=0, horizon=100
    )

    assert selfish["mean_journey_time"] < shared["mean_journey_time"]


# ---------------------------------------------------------------------------
# Acceptance criteria: reproducibility
# ---------------------------------------------------------------------------

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


def test_repeated_coordinated_runs_with_same_seed_are_deterministic():
    """Repeated coordinated runs with the same seed must return identical rows."""
    first = _run_focused_scenario(
        policy="coordinated", adoption_rate=1.0, disruption="none", seed=2, horizon=100
    )
    second = _run_focused_scenario(
        policy="coordinated", adoption_rate=1.0, disruption="none", seed=2, horizon=100
    )

    assert first == second
