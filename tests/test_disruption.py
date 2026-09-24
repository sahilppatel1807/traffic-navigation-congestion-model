"""Tests for the reduced-road-capacity disruption helper."""

import networkx as nx
import pytest

from src.congestion import calculate_travel_time
from src.disruption import reduce_road_capacity
from src.network import create_default_network


def _tiny_two_way(
    *,
    capacity_uv: float = 10.0,
    capacity_vu: float = 10.0,
    occupancy_uv: float = 0.0,
    occupancy_vu: float = 0.0,
    free_flow: float = 2.0,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    for u, v, capacity, occupancy in (
        (0, 1, capacity_uv, occupancy_uv),
        (1, 0, capacity_vu, occupancy_vu),
    ):
        graph.add_edge(
            u,
            v,
            free_flow_time=free_flow,
            capacity=capacity,
            occupancy=occupancy,
            current_travel_time=free_flow,
        )
    return graph


# ---------------------------------------------------------------------------
# Primary seam — reduce_road_capacity
# ---------------------------------------------------------------------------


def test_both_directions_scaled_by_factor():
    graph = _tiny_two_way(capacity_uv=10.0, capacity_vu=8.0)
    result = reduce_road_capacity(graph, 0, 1, factor=0.5)

    assert result is graph
    assert graph[0][1]["capacity"] == 5.0
    assert graph[1][0]["capacity"] == 4.0


def test_factor_one_leaves_capacity_unchanged():
    graph = _tiny_two_way(capacity_uv=10.0, capacity_vu=10.0)
    reduce_road_capacity(graph, 0, 1, factor=1.0)
    assert graph[0][1]["capacity"] == 10.0
    assert graph[1][0]["capacity"] == 10.0


def test_integer_factor_one_accepted():
    graph = _tiny_two_way()
    reduce_road_capacity(graph, 0, 1, factor=1)
    assert graph[0][1]["capacity"] == 10.0


def test_travel_time_refreshed_after_cut_with_occupancy():
    graph = _tiny_two_way(
        capacity_uv=10.0,
        capacity_vu=10.0,
        occupancy_uv=5.0,
        occupancy_vu=8.0,
        free_flow=2.0,
    )
    # Stale travel times before the cut
    graph[0][1]["current_travel_time"] = 999.0
    graph[1][0]["current_travel_time"] = 999.0

    reduce_road_capacity(graph, 0, 1, factor=0.5)

    assert graph[0][1]["capacity"] == 5.0
    assert graph[1][0]["capacity"] == 5.0
    assert graph[0][1]["occupancy"] == 5.0
    assert graph[1][0]["occupancy"] == 8.0
    assert graph[0][1]["current_travel_time"] == calculate_travel_time(
        free_flow_time=2.0, capacity=5.0, occupancy=5.0
    )
    assert graph[1][0]["current_travel_time"] == calculate_travel_time(
        free_flow_time=2.0, capacity=5.0, occupancy=8.0
    )


def test_reapply_compounds_capacity():
    graph = _tiny_two_way(capacity_uv=10.0, capacity_vu=10.0)
    reduce_road_capacity(graph, 0, 1, factor=0.5)
    reduce_road_capacity(graph, 0, 1, factor=0.5)
    assert graph[0][1]["capacity"] == 2.5
    assert graph[1][0]["capacity"] == 2.5


def test_default_network_road_pair():
    graph = create_default_network()
    original_uv = graph[0][1]["capacity"]
    original_vu = graph[1][0]["capacity"]

    reduce_road_capacity(graph, 0, 1, factor=0.25)

    assert graph[0][1]["capacity"] == float(original_uv) * 0.25
    assert graph[1][0]["capacity"] == float(original_vu) * 0.25


def test_missing_opposite_edge_raises():
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(
        0,
        1,
        free_flow_time=1.0,
        capacity=10,
        occupancy=0,
        current_travel_time=1.0,
    )

    with pytest.raises(ValueError, match="missing directed edge"):
        reduce_road_capacity(graph, 0, 1, factor=0.5)


def test_missing_forward_edge_raises():
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(
        1,
        0,
        free_flow_time=1.0,
        capacity=10,
        occupancy=0,
        current_travel_time=1.0,
    )

    with pytest.raises(ValueError, match="missing directed edge"):
        reduce_road_capacity(graph, 0, 1, factor=0.5)


@pytest.mark.parametrize("bad_factor", [0.0, -0.1, 1.01, 2, -1])
def test_factor_out_of_range_raises_value_error(bad_factor):
    with pytest.raises(ValueError, match="factor must be in"):
        reduce_road_capacity(_tiny_two_way(), 0, 1, factor=bad_factor)


@pytest.mark.parametrize("bad_factor", [True, False])
def test_bool_factor_raises_type_error(bad_factor):
    with pytest.raises(TypeError, match="factor must be"):
        reduce_road_capacity(_tiny_two_way(), 0, 1, factor=bad_factor)


@pytest.mark.parametrize("bad_factor", ["0.5", None, [0.5]])
def test_non_numeric_factor_raises_type_error(bad_factor):
    with pytest.raises(TypeError, match="factor must be"):
        reduce_road_capacity(_tiny_two_way(), 0, 1, factor=bad_factor)


def test_non_numeric_capacity_raises_type_error():
    graph = _tiny_two_way()
    graph[0][1]["capacity"] = "10"

    with pytest.raises(TypeError, match="capacity on edge"):
        reduce_road_capacity(graph, 0, 1, factor=0.5)


@pytest.mark.parametrize("bad_capacity", [0, -1, 0.0])
def test_non_positive_capacity_raises_value_error(bad_capacity):
    graph = _tiny_two_way()
    graph[1][0]["capacity"] = bad_capacity

    with pytest.raises(ValueError, match="capacity on edge"):
        reduce_road_capacity(graph, 0, 1, factor=0.5)


def test_validation_fails_before_writing():
    graph = _tiny_two_way(capacity_uv=10.0, capacity_vu=10.0)
    graph[1][0]["capacity"] = 0
    original_uv = graph[0][1]["capacity"]

    with pytest.raises(ValueError):
        reduce_road_capacity(graph, 0, 1, factor=0.5)

    assert graph[0][1]["capacity"] == original_uv


def test_missing_factor_raises_type_error():
    with pytest.raises(TypeError):
        reduce_road_capacity(_tiny_two_way(), 0, 1)  # type: ignore[call-arg]
