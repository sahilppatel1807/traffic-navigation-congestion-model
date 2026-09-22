"""Tests for the Vehicle agent representation and journey state invariants."""

import pytest
import networkx as nx

from src.vehicle import Vehicle
from src.network import create_default_network


# ---------------------------------------------------------------------------
# Vehicle Constructor & Invariants — Primary Seam
# ---------------------------------------------------------------------------


def test_vehicle_instantiation_with_string_and_integer_ids():
    # String ID
    v1 = Vehicle(vehicle_id="v_101", origin=0, destination=1, start_time=5)
    assert v1.id == "v_101"
    assert v1.vehicle_id == "v_101"
    assert v1.origin == 0
    assert v1.destination == 1
    assert v1.start_time == 5

    # Integer ID
    v2 = Vehicle(vehicle_id=42, origin="A", destination="B", start_time=0)
    assert v2.id == 42
    assert v2.vehicle_id == 42
    assert v2.origin == "A"
    assert v2.destination == "B"
    assert v2.start_time == 0


def test_invalid_vehicle_id_type_raises_type_error():
    with pytest.raises(TypeError):
        Vehicle(vehicle_id=3.14, origin="A", destination="B")  # type: ignore

    with pytest.raises(TypeError):
        Vehicle(vehicle_id=True, origin="A", destination="B")  # type: ignore


def test_invalid_start_time_type_raises_type_error():
    with pytest.raises(TypeError):
        Vehicle(vehicle_id="v1", origin="A", destination="B", start_time="5")  # type: ignore

    with pytest.raises(TypeError):
        Vehicle(vehicle_id="v1", origin="A", destination="B", start_time=True)  # type: ignore


def test_origin_equal_destination_raises_value_error():
    with pytest.raises(ValueError, match="origin and destination cannot be the same"):
        Vehicle(vehicle_id="v1", origin="A", destination="A")


def test_negative_start_time_raises_value_error():
    with pytest.raises(ValueError, match="start_time must be >= 0"):
        Vehicle(vehicle_id="v1", origin="A", destination="B", start_time=-1)


def test_vehicle_initial_state():
    v = Vehicle(vehicle_id="v1", origin="A", destination="B", start_time=10)

    assert v.current_position == "A"
    assert v.route is None
    assert v.completion_time is None
    assert v.uses_navigation_app is False


def test_uses_navigation_app_default_is_false():
    v = Vehicle(vehicle_id="v1", origin="A", destination="B")
    assert v.uses_navigation_app is False


def test_uses_navigation_app_true_is_stored():
    v = Vehicle(vehicle_id="v1", origin="A", destination="B", uses_navigation_app=True)
    assert v.uses_navigation_app is True


def test_uses_navigation_app_non_boolean_raises_type_error():
    with pytest.raises(TypeError, match="uses_navigation_app must be a bool"):
        Vehicle(vehicle_id="v1", origin="A", destination="B", uses_navigation_app=1)  # type: ignore

    with pytest.raises(TypeError, match="uses_navigation_app must be a bool"):
        Vehicle(vehicle_id="v1", origin="A", destination="B", uses_navigation_app=0)  # type: ignore

    with pytest.raises(TypeError, match="uses_navigation_app must be a bool"):
        Vehicle(vehicle_id="v1", origin="A", destination="B", uses_navigation_app="yes")  # type: ignore


# ---------------------------------------------------------------------------
# create_for_network Factory Method — Secondary Seam
# ---------------------------------------------------------------------------


def test_create_for_network_successful():
    graph = create_default_network()  # default network has nodes 0 to 5
    v = Vehicle.create_for_network(
        graph=graph,
        vehicle_id="v1",
        origin=1,
        destination=4,
        start_time=2,
    )

    assert v.id == "v1"
    assert v.origin == 1
    assert v.destination == 4
    assert v.start_time == 2
    assert v.current_position == 1
    assert v.uses_navigation_app is False


def test_create_for_network_forwards_uses_navigation_app():
    graph = create_default_network()
    v = Vehicle.create_for_network(
        graph=graph,
        vehicle_id="nav",
        origin=0,
        destination=2,
        uses_navigation_app=True,
    )

    assert v.uses_navigation_app is True


def test_create_for_network_raises_for_non_existent_origin_or_destination():
    graph = create_default_network()

    # Invalid origin
    with pytest.raises(ValueError, match="origin .* is not a valid node"):
        Vehicle.create_for_network(graph=graph, vehicle_id="v1", origin=99, destination=4)

    # Invalid destination
    with pytest.raises(ValueError, match="destination .* is not a valid node"):
        Vehicle.create_for_network(graph=graph, vehicle_id="v1", origin=1, destination=99)


# ---------------------------------------------------------------------------
# Route & Position Mutation — Tertiary Seam
# ---------------------------------------------------------------------------


def test_set_route_boundaries_valid():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")
    route = ["A", "B", "C"]
    v.set_route(route)

    assert v.route == route


def test_set_route_raises_for_invalid_boundaries():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")

    # Does not start at origin
    with pytest.raises(ValueError, match="Route must start at origin"):
        v.set_route(["B", "C"])

    # Does not end at destination
    with pytest.raises(ValueError, match="Route must end at destination"):
        v.set_route(["A", "B"])


def test_set_route_raises_for_empty_route():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")
    with pytest.raises(ValueError, match="Route cannot be empty"):
        v.set_route([])


def test_set_route_with_graph_validation():
    graph = create_default_network()
    v = Vehicle(vehicle_id="v1", origin=0, destination=2)

    # Valid route with all nodes in graph
    v.set_route([0, 1, 2], graph=graph)
    assert v.route == [0, 1, 2]

    # Node in route not in graph
    with pytest.raises(ValueError, match="does not exist in the graph"):
        v.set_route([0, 99, 2], graph=graph)


def test_set_route_raises_if_current_position_not_in_route():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")
    v.current_position = "D"  # manually simulate position elsewhere

    with pytest.raises(ValueError, match="Current position .* is not in the assigned route"):
        v.set_route(["A", "B", "C"])


def test_update_position_without_route_or_graph():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")
    v.update_position("B")

    assert v.current_position == "B"


def test_update_position_with_route():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C")
    v.set_route(["A", "B", "C"])

    # Move to a node in the route
    v.update_position("B")
    assert v.current_position == "B"

    # Try moving to a node NOT in the route
    with pytest.raises(ValueError, match="is not part of the assigned route"):
        v.update_position("D")


def test_update_position_with_graph():
    graph = create_default_network()
    v = Vehicle(vehicle_id="v1", origin=0, destination=2)

    # Move to valid node in graph
    v.update_position(1, graph=graph)
    assert v.current_position == 1

    # Try moving to invalid node in graph
    with pytest.raises(ValueError, match="does not exist in the graph"):
        v.update_position(99, graph=graph)


def test_complete_journey_successful():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C", start_time=5)
    v.set_route(["A", "B", "C"])

    # Move to destination
    v.update_position("C")

    # Complete journey at later time
    v.complete_journey(12)
    assert v.completion_time == 12


def test_complete_journey_raises_for_invalid_type_or_bounds():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C", start_time=5)
    v.set_route(["A", "B", "C"])
    v.update_position("C")

    # Non-integer completion_time
    with pytest.raises(TypeError, match="completion_time must be an integer"):
        v.complete_journey("12")  # type: ignore

    with pytest.raises(TypeError, match="completion_time must be an integer"):
        v.complete_journey(True)  # type: ignore

    # completion_time < start_time
    with pytest.raises(ValueError, match="cannot be less than start_time"):
        v.complete_journey(4)


def test_complete_journey_raises_if_not_at_destination():
    v = Vehicle(vehicle_id="v1", origin="A", destination="C", start_time=5)
    v.set_route(["A", "B", "C"])

    # Attempt to complete journey while still at "A" (or "B")
    with pytest.raises(ValueError, match="is not the destination"):
        v.complete_journey(10)
