"""Tests for the BPR congestion travel-time module."""

import pytest

from src.congestion import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    calculate_travel_time,
    update_all_travel_times,
    update_road_travel_time,
)
from src.network import create_default_network


# ---------------------------------------------------------------------------
# calculate_travel_time — primary seam
# ---------------------------------------------------------------------------


def test_zero_occupancy_returns_free_flow_time():
    result = calculate_travel_time(free_flow_time=5.0, capacity=10, occupancy=0)

    assert result == 5.0


def test_travel_time_is_non_decreasing_with_occupancy():
    free_flow_time = 2.0
    capacity = 10

    occupancies = [0, capacity / 2, capacity, 2 * capacity]
    times = [
        calculate_travel_time(free_flow_time, capacity, occ) for occ in occupancies
    ]

    for earlier, later in zip(times, times[1:]):
        assert later >= earlier


def test_travel_time_at_capacity_exceeds_free_flow():
    free_flow_time = 3.0
    capacity = 20

    result = calculate_travel_time(free_flow_time, capacity, occupancy=capacity)

    assert result > free_flow_time


def test_travel_time_at_double_capacity_significantly_exceeds_free_flow():
    free_flow_time = 2.0
    capacity = 10

    result = calculate_travel_time(free_flow_time, capacity, occupancy=2 * capacity)

    # At 2× capacity with default alpha=0.15, beta=4:
    # t = 2.0 * (1 + 0.15 * 2**4) = 2.0 * (1 + 2.4) = 6.8
    assert result > free_flow_time * 2


def test_custom_alpha_beta_differs_from_default():
    free_flow_time = 1.0
    capacity = 10
    occupancy = 5

    default_result = calculate_travel_time(free_flow_time, capacity, occupancy)
    custom_result = calculate_travel_time(
        free_flow_time, capacity, occupancy, alpha=1.0, beta=2.0
    )

    assert custom_result != default_result


def test_raises_for_non_positive_capacity():
    with pytest.raises(ValueError):
        calculate_travel_time(free_flow_time=1.0, capacity=0, occupancy=5)

    with pytest.raises(ValueError):
        calculate_travel_time(free_flow_time=1.0, capacity=-5, occupancy=5)


def test_raises_for_negative_occupancy():
    with pytest.raises(ValueError):
        calculate_travel_time(free_flow_time=1.0, capacity=10, occupancy=-1)


def test_raises_for_negative_free_flow_time():
    with pytest.raises(ValueError):
        calculate_travel_time(free_flow_time=-1.0, capacity=10, occupancy=5)


# ---------------------------------------------------------------------------
# update_road_travel_time — secondary seam
# ---------------------------------------------------------------------------


def test_update_road_travel_time_mutates_current_travel_time():
    road = {"free_flow_time": 2.0, "capacity": 10, "occupancy": 5}

    update_road_travel_time(road)

    expected = calculate_travel_time(
        free_flow_time=2.0, capacity=10, occupancy=5
    )
    assert road["current_travel_time"] == expected


def test_update_road_travel_time_forwards_alpha_and_beta():
    road = {"free_flow_time": 2.0, "capacity": 10, "occupancy": 5}

    update_road_travel_time(road, alpha=1.0, beta=2.0)

    expected = calculate_travel_time(
        free_flow_time=2.0, capacity=10, occupancy=5, alpha=1.0, beta=2.0
    )
    assert road["current_travel_time"] == expected


# ---------------------------------------------------------------------------
# update_all_travel_times — integration with real network
# ---------------------------------------------------------------------------


def test_update_all_travel_times_sets_correct_values():
    graph = create_default_network()

    # Manually set distinct occupancies on a subset of edges
    edges = list(graph.edges())
    for i, (u, v) in enumerate(edges):
        graph[u][v]["occupancy"] = i * 2  # 0, 2, 4, 6, ...

    update_all_travel_times(graph)

    for u, v, attrs in graph.edges(data=True):
        expected = calculate_travel_time(
            free_flow_time=attrs["free_flow_time"],
            capacity=attrs["capacity"],
            occupancy=attrs["occupancy"],
        )
        assert attrs["current_travel_time"] == expected


def test_update_all_travel_times_zero_occupancy_equals_free_flow():
    graph = create_default_network()
    # All edges already have occupancy=0 by default

    update_all_travel_times(graph)

    for _u, _v, attrs in graph.edges(data=True):
        assert attrs["current_travel_time"] == attrs["free_flow_time"]


def test_update_all_travel_times_forwards_alpha_and_beta():
    graph = create_default_network()

    # Set a non-zero occupancy so the formula produces a distinct value
    for u, v in graph.edges():
        graph[u][v]["occupancy"] = 5

    update_all_travel_times(graph, alpha=1.0, beta=2.0)

    for u, v, attrs in graph.edges(data=True):
        expected = calculate_travel_time(
            free_flow_time=attrs["free_flow_time"],
            capacity=attrs["capacity"],
            occupancy=attrs["occupancy"],
            alpha=1.0,
            beta=2.0,
        )
        assert attrs["current_travel_time"] == expected
