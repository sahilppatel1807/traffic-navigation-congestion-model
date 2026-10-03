"""Focused tests for the reproducible routing-comparison experiment."""

from src.experiments import run_routing_comparison


def _small_result(**overrides):
    config = {
        "seeds": (0,),
        "horizon": 40,
        "wave_interval": 10,
        "routing_policies": ("static", "selfish", "coordinated"),
        "demand_levels": {"low": 1},
        "adoption_rates": (0.0, 1.0),
        "disruptions": ("none", "capacity_reduction", "closure"),
    }
    config.update(overrides)
    return run_routing_comparison(**config)


def test_small_matrix_records_all_factors_and_expected_shape():
    result = _small_result()

    assert len(result["summary"]) == 18
    assert {row["policy"] for row in result["summary"]} == {
        "static", "selfish", "coordinated"
    }
    assert all(row["horizon"] == 40 for row in result["summary"])
    assert all("seed" in row and "scenario_id" in row for row in result["summary"])
    assert all("edge" in row and "timestep" in row for row in result["congestion"])


def test_same_seed_returns_identical_raw_rows():
    assert _small_result() == _small_result()


def test_static_routing_is_invariant_to_adoption():
    result = _small_result(
        routing_policies=("static",), disruptions=("none",), adoption_rates=(0.0, 1.0)
    )
    rows = {row["adoption_rate"]: row for row in result["summary"]}
    assert rows[0.0]["mean_journey_time"] == rows[1.0]["mean_journey_time"]
    assert rows[0.0]["total_network_delay"] == rows[1.0]["total_network_delay"]


def test_closure_is_removed_during_disruption_and_restored_afterward():
    result = _small_result(
        routing_policies=("static",), adoption_rates=(0.0,), disruptions=("closure",)
    )
    rows = result["congestion"]
    assert len([row for row in rows if row["timestep"] == 9]) == 12
    # Live experiments retain the closed edge so vehicles already on it can
    # finish safely; new route searches exclude the marked edge.
    assert len([row for row in rows if row["timestep"] == 10]) == 12
    assert len([row for row in rows if row["timestep"] == 30]) == 12


def test_recovery_status_is_not_applicable_without_disruption():
    result = _small_result(
        routing_policies=("static",), adoption_rates=(0.0,), disruptions=("none",)
    )
    row = result["summary"][0]
    assert row["recovery_status"] == "not_applicable"
    assert row["recovered"] is None


def test_shared_navigation_is_available_in_routing_comparison_runner():
    result = _small_result(
        routing_policies=("shared_navigation",),
        demand_levels={"high": 10},
        adoption_rates=(1.0,),
        disruptions=("none",),
    )
    row = result["summary"][0]
    assert row["policy"] == "shared_navigation"
    assert row["nav_count"] == 10
