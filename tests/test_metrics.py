"""Tests for journey-level and road-level metrics."""

import math

import networkx as nx
import pytest

from src.metrics import (
    RECOVERY_THRESHOLD_FACTOR,
    completed_count,
    congestion_recovery,
    journey_time,
    mean_journey_time,
    network_congestion_score,
    pre_disruption_congestion,
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


# ---------------------------------------------------------------------------
# network_congestion_score — uncapped capacity-weighted mean
# ---------------------------------------------------------------------------


def test_network_congestion_score_matches_hand_computed_weighted_mean():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=5)
    graph.add_edge(1, 2, capacity=5, occupancy=5)
    # Hand: (5 + 5) / (10 + 5) = 10/15
    assert network_congestion_score(graph) == pytest.approx(10 / 15)


def test_network_congestion_score_uncapped_above_one():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=4, occupancy=10)
    graph.add_edge(1, 2, capacity=6, occupancy=0)
    # Combined: 10 / 10 = 1.0
    assert network_congestion_score(graph) == pytest.approx(1.0)

    graph2 = nx.DiGraph()
    graph2.add_edge(0, 1, capacity=4, occupancy=10)
    assert network_congestion_score(graph2) == pytest.approx(2.5)
    # Visualisation/capped road_congestion stays at 1.0 on the same edge.
    assert road_congestion(graph2)[(0, 1)] == 1.0


def test_network_congestion_score_excludes_absent_closed_edges():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, occupancy=2)
    graph.add_edge(1, 2, capacity=10, occupancy=8)
    # Closing removes the edge from the graph — score uses remaining open edges.
    graph.remove_edge(1, 2)
    assert network_congestion_score(graph) == pytest.approx(0.2)


def test_network_congestion_score_nan_when_no_open_edges():
    graph = nx.DiGraph()
    graph.add_node(0)
    assert math.isnan(network_congestion_score(graph))


# ---------------------------------------------------------------------------
# pre_disruption_congestion — window mean
# ---------------------------------------------------------------------------


def test_pre_disruption_congestion_window_mean():
    # indices: 0 1 2 3 4 5 6
    C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8]
    t_star, W = 5, 3
    # Window [2, 5): 0.3, 0.4, 0.5 → mean 0.4
    assert pre_disruption_congestion(C, t_star, W) == pytest.approx(0.4)


def test_pre_disruption_congestion_raises_for_bad_window_params():
    C = [0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match="W"):
        pre_disruption_congestion(C, t_star=2, W=0)
    with pytest.raises(ValueError, match="t_star"):
        pre_disruption_congestion(C, t_star=-1, W=1)
    with pytest.raises(ValueError, match="t_star"):
        pre_disruption_congestion(C, t_star=1, W=2)
    with pytest.raises(ValueError, match="pre-disruption window"):
        pre_disruption_congestion([0.2, 0.2], t_star=3, W=2)


# ---------------------------------------------------------------------------
# congestion_recovery — primary seam (synthetic series)
# ---------------------------------------------------------------------------


def test_congestion_recovery_happy_path_post_restore_streak():
    # C_pre window [0,4): all 1.0 → C_pre=1.0, tau=1.05
    # t*=4, t_r=7, K=2, H=12
    # Excursion in [4,7): C[4]=1.2, C[5]=1.3, C[6]=1.1
    # Post-restore: C[7]=1.2 (still high), C[8]=1.0, C[9]=1.0 → streak at 8
    C = [
        1.0, 1.0, 1.0, 1.0,  # 0..3 pre
        1.2, 1.3, 1.1,  # 4..6 disrupted
        1.2, 1.0, 1.0, 1.0, 1.0, 1.0,  # 7..12
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=2, horizon=12)

    assert result["valid"] is True
    assert result["invalid_reason"] is None
    assert result["recovered"] is True
    assert result["no_excursion"] is False
    assert result["censored"] is False
    assert result["C_pre"] == pytest.approx(1.0)
    assert result["tau"] == pytest.approx(RECOVERY_THRESHOLD_FACTOR)
    assert result["excursion_start"] == 4
    assert result["recovery_start"] == 8
    assert result["confirmation_time"] == 9
    assert result["recovery_time"] == 4  # 8 - 4
    assert result["censor_time"] is None
    assert result["t_star"] == 4
    assert result["t_r"] == 7
    assert result["K"] == 2
    assert result["W"] == 4
    assert result["horizon"] == 12


def test_congestion_recovery_confirmation_exactly_at_horizon():
    # Inclusive horizon: streak start=9, K=2 → confirmation=10 == H.
    C = [
        1.0, 1.0, 1.0, 1.0,
        1.2, 1.2, 1.2,
        1.2, 1.2, 1.0, 1.0,
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=2, horizon=10)

    assert result["recovered"] is True
    assert result["recovery_start"] == 9
    assert result["confirmation_time"] == 10
    assert result["recovery_time"] == 5
    assert result["censored"] is False


def test_congestion_recovery_no_excursion_in_disrupted_interval():
    # Pre window all 1.0 → tau=1.05; disrupted interval stays at 1.0
    C = [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,  # [4,7) never > tau
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=2, horizon=12)

    assert result["valid"] is True
    assert result["no_excursion"] is True
    assert result["recovered"] is False
    assert result["censored"] is False
    assert result["recovery_time"] is None
    assert result["recovery_start"] is None
    assert result["confirmation_time"] is None
    assert result["excursion_start"] is None
    assert result["censor_time"] is None
    assert result["tau"] == pytest.approx(1.05)
    assert result["C_pre"] == pytest.approx(1.0)


def test_congestion_recovery_no_excursion_when_restore_equals_disruption():
    # Zero-width disrupted interval [t*, t*) cannot contain an excursion.
    C = [1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0]
    result = congestion_recovery(C, t_star=4, t_r=4, W=4, K=2, horizon=8)

    assert result["valid"] is True
    assert result["no_excursion"] is True
    assert result["recovered"] is False
    assert result["censored"] is False
    assert result["excursion_start"] is None
    assert result["recovery_time"] is None


def test_congestion_recovery_ignores_pre_restore_streak():
    # Excursion at t*=4; then C drops below tau before restore — must ignore.
    # Qualifying streak only after t_r=7.
    C = [
        1.0, 1.0, 1.0, 1.0,
        1.2, 1.0, 1.0,  # excursion then pre-restore streak at 5..6
        1.0, 1.0, 0.9, 0.9, 0.9, 0.9,
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=2, horizon=12)

    assert result["recovered"] is True
    assert result["excursion_start"] == 4
    assert result["recovery_start"] == 7  # not 5
    assert result["confirmation_time"] == 8
    assert result["recovery_time"] == 3


def test_congestion_recovery_censored_when_streak_confirms_after_horizon():
    # K=3; keep C > tau through horizon so no streak fits.
    C = [
        1.0, 1.0, 1.0, 1.0,
        1.2, 1.2, 1.2,
        1.2, 1.2, 1.2, 1.2,
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=3, horizon=10)

    assert result["valid"] is True
    assert result["recovered"] is False
    assert result["censored"] is True
    assert result["recovery_time"] is None
    assert result["recovery_start"] is None
    assert result["confirmation_time"] is None
    assert result["censor_time"] == 6  # 10 - 4
    assert result["no_excursion"] is False
    assert result["excursion_start"] == 4


def test_congestion_recovery_partial_end_streak_does_not_count():
    # K=3, H=10; C drops only at 9..10 (two steps) — cannot confirm.
    C = [
        1.0, 1.0, 1.0, 1.0,
        1.2, 1.2, 1.2,
        1.2, 1.2, 1.0, 1.0,
    ]
    result = congestion_recovery(C, t_star=4, t_r=7, W=4, K=3, horizon=10)

    assert result["censored"] is True
    assert result["recovered"] is False
    assert result["recovery_time"] is None
    assert result["censor_time"] == 6


def test_congestion_recovery_invalid_restore_time():
    C = [1.0] * 13
    for bad_t_r in (3, 13, -1, 7.0, None, True):  # out of range / non-int
        result = congestion_recovery(C, t_star=4, t_r=bad_t_r, W=4, K=2, horizon=12)  # type: ignore[arg-type]
        assert result["valid"] is False
        assert result["invalid_reason"] == "invalid_restore_time"
        assert result["recovered"] is False
        assert result["censored"] is False
        assert result["no_excursion"] is False
        assert result["tau"] is None
        assert result["recovery_time"] is None
        assert result["recovery_start"] is None
        assert result["confirmation_time"] is None
        assert result["censor_time"] is None
        assert result["C_pre"] is None
        assert result["t_r"] == bad_t_r


def test_congestion_recovery_non_positive_pre_disruption():
    # Window of zeros → C_pre=0
    C = [0.0, 0.0, 0.0, 0.0, 1.2, 1.2, 1.0, 1.0, 1.0]
    result = congestion_recovery(C, t_star=4, t_r=6, W=4, K=2, horizon=8)

    assert result["valid"] is False
    assert result["invalid_reason"] == "non_positive_pre_disruption_congestion"
    assert result["C_pre"] == pytest.approx(0.0)
    assert result["tau"] is None
    assert result["recovery_time"] is None
    assert result["recovery_start"] is None
    assert result["confirmation_time"] is None
    assert result["censor_time"] is None
    assert result["recovered"] is False
    assert result["censored"] is False
    assert result["no_excursion"] is False


def test_congestion_recovery_raises_for_bad_parameters():
    C = [1.0] * 13
    with pytest.raises(ValueError, match="K"):
        congestion_recovery(C, t_star=4, t_r=7, W=4, K=0, horizon=12)
    with pytest.raises(ValueError, match="t_star"):
        congestion_recovery(C, t_star=-1, t_r=7, W=4, K=2, horizon=12)
    with pytest.raises(ValueError, match="W"):
        congestion_recovery(C, t_star=4, t_r=7, W=0, K=2, horizon=12)
    with pytest.raises(ValueError, match="t_star"):
        congestion_recovery(C, t_star=3, t_r=7, W=4, K=2, horizon=12)
    with pytest.raises(ValueError, match="horizon"):
        congestion_recovery([1.0] * 12, t_star=4, t_r=7, W=4, K=2, horizon=12)
    with pytest.raises(TypeError, match="Sequence"):
        congestion_recovery({0: 1.0}, t_star=4, t_r=7, W=4, K=2, horizon=12)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Sequence"):
        congestion_recovery("not-a-series", t_star=4, t_r=7, W=4, K=2, horizon=12)  # type: ignore[arg-type]
