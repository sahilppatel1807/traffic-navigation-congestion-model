"""End-to-end baseline demand validation: rising demand → rising delay."""

from src.demand import (
    DEMAND_HIGH,
    DEMAND_LOW,
    DEMAND_MEDIUM,
    build_corridor_demand,
    build_high_demand,
    build_low_demand,
    build_medium_demand,
)
from src.metrics import completed_count, mean_journey_time
from src.network import create_default_network
from src.simulation import Simulation

# Free-flow discrete journey time for default corridor 0→1→2 under the
# simulation clock (completion_time - start_time). Two free-flow edges of
# time 1 complete with journey_time == 1 (see Simulation edge-dwell rules).
FREE_FLOW_JOURNEY_TIME = 1.0
VALIDATION_HORIZON = 100


def _run_baseline(n: int, builder=None):
    """Build demand on a fresh default network and run the baseline simulation."""
    graph = create_default_network()
    if builder is None:
        vehicles = build_corridor_demand(graph, n)
    else:
        vehicles = builder(graph)
    sim = Simulation(graph, vehicles)
    sim.run(until=VALIDATION_HORIZON)
    return sim


# ---------------------------------------------------------------------------
# Primary seam — demand builder → Simulation.run → journey metrics
# ---------------------------------------------------------------------------


def test_low_demand_matches_free_flow_journey_time():
    sim = _run_baseline(DEMAND_LOW, builder=build_low_demand)

    assert completed_count(sim.vehicles) == DEMAND_LOW
    assert mean_journey_time(sim.vehicles) == FREE_FLOW_JOURNEY_TIME


def test_mean_journey_time_rises_strictly_across_demand_levels():
    sim_low = _run_baseline(DEMAND_LOW, builder=build_low_demand)
    sim_medium = _run_baseline(DEMAND_MEDIUM, builder=build_medium_demand)
    sim_high = _run_baseline(DEMAND_HIGH, builder=build_high_demand)

    assert completed_count(sim_low.vehicles) == DEMAND_LOW
    assert completed_count(sim_medium.vehicles) == DEMAND_MEDIUM
    assert completed_count(sim_high.vehicles) == DEMAND_HIGH

    jt_low = mean_journey_time(sim_low.vehicles)
    jt_medium = mean_journey_time(sim_medium.vehicles)
    jt_high = mean_journey_time(sim_high.vehicles)

    assert jt_low is not None and jt_medium is not None and jt_high is not None
    assert jt_low < jt_medium < jt_high


def test_explicit_batch_size_completes_within_horizon():
    sim = _run_baseline(3)

    assert completed_count(sim.vehicles) == 3
    assert all(v.completion_time is not None for v in sim.vehicles)


# ---------------------------------------------------------------------------
# Secondary — batch shape only (not a second behavioural oracle)
# ---------------------------------------------------------------------------


def test_corridor_batch_shape_defaults():
    graph = create_default_network()
    vehicles = build_corridor_demand(graph, n=4)

    assert len(vehicles) == 4
    for vehicle in vehicles:
        assert vehicle.origin == 0
        assert vehicle.destination == 2
        assert vehicle.start_time == 0
        assert vehicle.route is None
