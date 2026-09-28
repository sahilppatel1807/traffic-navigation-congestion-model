"""Tests for static shortest-path routing and route-cost estimation."""

import networkx as nx
import pytest

from src.network import create_default_network
from src.routing import estimate_route_cost, find_shortest_route


# ---------------------------------------------------------------------------
# Happy path on the default network — Primary Seam
# ---------------------------------------------------------------------------


def test_default_network_route_endpoints_and_connectivity():
    graph = create_default_network()
    origin, destination = 0, 5

    route = find_shortest_route(graph, origin, destination)

    assert route[0] == origin
    assert route[-1] == destination
    for u, v in zip(route, route[1:]):
        assert graph.has_edge(u, v)


def test_default_network_route_cost_matches_networkx_shortest_path_length():
    graph = create_default_network()
    origin, destination = 0, 5

    route = find_shortest_route(graph, origin, destination)
    expected_length = nx.shortest_path_length(
        graph, source=origin, target=destination, weight="free_flow_time"
    )

    assert estimate_route_cost(graph, route, weight="free_flow_time") == expected_length


def test_route_is_pure_and_does_not_mutate_graph():
    graph = create_default_network()
    edge_snapshot = {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)}
    node_snapshot = set(graph.nodes)

    find_shortest_route(graph, 0, 5)

    assert set(graph.nodes) == node_snapshot
    assert {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)} == edge_snapshot


# ---------------------------------------------------------------------------
# Custom weighted graph — Secondary Seam
# ---------------------------------------------------------------------------


def test_weighted_graph_prefers_free_flow_cheapest_over_fewest_hops():
    """Fewest hops A→C costs 10; A→B→C costs 2 and must be chosen."""
    graph = nx.DiGraph()
    graph.add_edge("A", "C", free_flow_time=10.0)
    graph.add_edge("A", "B", free_flow_time=1.0)
    graph.add_edge("B", "C", free_flow_time=1.0)

    route = find_shortest_route(graph, "A", "C")

    assert route == ["A", "B", "C"]
    assert estimate_route_cost(graph, route, weight="free_flow_time") == 2.0
    assert estimate_route_cost(graph, ["A", "C"], weight="free_flow_time") == 10.0


def test_custom_weight_parameter_uses_supplied_attribute():
    graph = nx.DiGraph()
    # Free-flow prefers A→B→C; current travel time prefers direct A→C.
    graph.add_edge("A", "C", free_flow_time=10.0, current_travel_time=1.0)
    graph.add_edge("A", "B", free_flow_time=1.0, current_travel_time=5.0)
    graph.add_edge("B", "C", free_flow_time=1.0, current_travel_time=5.0)

    route = find_shortest_route(graph, "A", "C", weight="current_travel_time")

    assert route == ["A", "C"]
    assert estimate_route_cost(graph, route) == 1.0


# ---------------------------------------------------------------------------
# estimate_route_cost — Primary Seam
# ---------------------------------------------------------------------------


def test_estimate_route_cost_default_weight_sums_current_travel_time():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", current_travel_time=2.5, free_flow_time=1.0)
    graph.add_edge("B", "C", current_travel_time=3.5, free_flow_time=1.0)

    cost = estimate_route_cost(graph, ["A", "B", "C"])

    assert isinstance(cost, float)
    assert cost == 6.0


def test_estimate_route_cost_custom_weight_sums_alternate_attribute():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", current_travel_time=9.0, free_flow_time=1.0)
    graph.add_edge("B", "C", current_travel_time=9.0, free_flow_time=2.0)

    cost = estimate_route_cost(graph, ["A", "B", "C"], weight="free_flow_time")

    assert isinstance(cost, float)
    assert cost == 3.0


def test_estimate_route_cost_is_pure_and_does_not_mutate_graph():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", current_travel_time=2.0, occupancy=3)
    graph.add_edge("B", "C", current_travel_time=4.0, occupancy=1)
    edge_snapshot = {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)}
    node_snapshot = set(graph.nodes)

    estimate_route_cost(graph, ["A", "B", "C"])

    assert set(graph.nodes) == node_snapshot
    assert {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)} == edge_snapshot


def test_estimate_route_cost_empty_route_raises_value_error():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", current_travel_time=1.0)

    with pytest.raises(ValueError, match="empty"):
        estimate_route_cost(graph, [])


def test_estimate_route_cost_single_node_route_raises_value_error():
    graph = nx.DiGraph()
    graph.add_node("A")

    with pytest.raises(ValueError, match="at least two nodes"):
        estimate_route_cost(graph, ["A"])


def test_estimate_route_cost_missing_edge_raises_value_error():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", current_travel_time=1.0)
    # No A→C edge.

    with pytest.raises(ValueError, match="no directed edge"):
        estimate_route_cost(graph, ["A", "C"])


def test_estimate_route_cost_missing_weight_raises_key_error():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", free_flow_time=1.0)  # no current_travel_time

    with pytest.raises(KeyError):
        estimate_route_cost(graph, ["A", "B"])


# ---------------------------------------------------------------------------
# Error cases — Tertiary Seam
# ---------------------------------------------------------------------------


def test_missing_origin_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="origin"):
        find_shortest_route(graph, origin=99, destination=5)


def test_missing_destination_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="destination"):
        find_shortest_route(graph, origin=0, destination=99)


def test_origin_equals_destination_raises_value_error():
    graph = create_default_network()

    with pytest.raises(ValueError, match="cannot be the same"):
        find_shortest_route(graph, origin=0, destination=0)


def test_unreachable_pair_raises_value_error():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, free_flow_time=1.0)
    graph.add_node(2)  # disconnected

    with pytest.raises(ValueError, match="no path exists"):
        find_shortest_route(graph, origin=0, destination=2)


# ---------------------------------------------------------------------------
# compare_selfish_and_coordinated — Primary Seam
# ---------------------------------------------------------------------------


def _halved_corridor_scenario():
    """15-vehicle high-demand scenario with road 1–2 halved."""
    from src.demand import build_high_demand
    from src.disruption import reduce_road_capacity

    graph = create_default_network()
    vehicles = build_high_demand(graph)
    reduce_road_capacity(graph, 1, 2, factor=0.5)
    return graph, vehicles


def test_compare_returns_correct_keys():
    from src.routing import compare_selfish_and_coordinated

    graph, vehicles = _halved_corridor_scenario()
    result = compare_selfish_and_coordinated(graph, vehicles, until=200)

    assert set(result.keys()) == {
        "selfish_total_journey_time",
        "selfish_mean_journey_time",
        "coordinated_total_journey_time",
        "coordinated_mean_journey_time",
    }


def test_compare_coordinated_beats_selfish_on_halved_corridor():
    """Coordinated total = 53, selfish total = 58 on the halved-corridor scenario."""
    from src.routing import compare_selfish_and_coordinated

    graph, vehicles = _halved_corridor_scenario()
    result = compare_selfish_and_coordinated(graph, vehicles, until=200)

    assert result["selfish_total_journey_time"] == 58.0
    assert result["coordinated_total_journey_time"] == 53.0
    assert result["coordinated_total_journey_time"] < result["selfish_total_journey_time"]


def test_compare_mean_times_consistent_with_totals():
    """Mean journey times equal total / vehicle count."""
    from src.routing import compare_selfish_and_coordinated

    graph, vehicles = _halved_corridor_scenario()
    n = len(vehicles)
    result = compare_selfish_and_coordinated(graph, vehicles, until=200)

    assert result["selfish_mean_journey_time"] == pytest.approx(
        result["selfish_total_journey_time"] / n
    )
    assert result["coordinated_mean_journey_time"] == pytest.approx(
        result["coordinated_total_journey_time"] / n
    )


def test_compare_does_not_mutate_original_graph_or_vehicles():
    """compare_selfish_and_coordinated leaves the caller's graph and vehicle list intact."""
    from src.routing import compare_selfish_and_coordinated

    graph, vehicles = _halved_corridor_scenario()
    edge_snapshot = {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)}
    original_routes = [v.route for v in vehicles]
    original_completion = [v.completion_time for v in vehicles]

    compare_selfish_and_coordinated(graph, vehicles, until=200)

    assert {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)} == edge_snapshot
    for v, orig_route, orig_completion in zip(vehicles, original_routes, original_completion):
        assert v.route == orig_route
        assert v.completion_time == orig_completion


def test_compare_raises_value_error_if_vehicle_cannot_complete():
    """ValueError raised if a vehicle cannot reach its destination within until steps."""
    from src.routing import compare_selfish_and_coordinated

    graph = create_default_network()
    from src.vehicle import Vehicle
    # Give a very tight step limit so the vehicle cannot complete.
    vehicles = [Vehicle(vehicle_id="v0", origin=0, destination=5, start_time=0)]
    # until=1 is way too few steps for a 0→5 journey.
    with pytest.raises(ValueError, match="did not complete"):
        compare_selfish_and_coordinated(graph, vehicles, until=1)
