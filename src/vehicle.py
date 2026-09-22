"""Vehicle agent representation and journey state invariants for the traffic model."""

from __future__ import annotations

from typing import Any
import networkx as nx


class Vehicle:
    """Represents an individual vehicle agent in the traffic simulation.

    An agent tracks its unique identity, trip parameters (origin, destination,
    assigned route), and its journey state (current position, start time, and
    completion time) while enforcing strict physical invariants.
    """

    def __init__(
        self,
        vehicle_id: str | int,
        origin: Any,
        destination: Any,
        start_time: int = 0,
        uses_navigation_app: bool = False,
    ) -> None:
        """Initialize a Vehicle agent.

        Parameters
        ----------
        vehicle_id:
            A unique identifier, either a string or an integer.
        origin:
            The starting node of the vehicle's journey.
        destination:
            The ending node of the vehicle's journey.
        start_time:
            The discrete simulation time-step when the journey begins. Must be >= 0.
        uses_navigation_app:
            If ``True``, the simulation auto-routes this vehicle at entry using
            live congested travel times; if ``False`` (default), free-flow
            shortest paths are used. Ignored when a route is already assigned.
            Must be a real ``bool`` (integers and other truthy values are rejected).

        Raises
        ------
        TypeError
            If ``vehicle_id`` is not a string or integer.
            If ``start_time`` is not an integer.
            If ``uses_navigation_app`` is not a ``bool``.
        ValueError
            If ``origin == destination``.
            If ``start_time < 0``.
        """
        if not isinstance(vehicle_id, (str, int)) or isinstance(vehicle_id, bool):
            raise TypeError(
                f"vehicle_id must be a string or integer, got {type(vehicle_id).__name__}"
            )
        if not isinstance(start_time, int) or isinstance(start_time, bool):
            raise TypeError(
                f"start_time must be an integer, got {type(start_time).__name__}"
            )
        if not isinstance(uses_navigation_app, bool):
            raise TypeError(
                f"uses_navigation_app must be a bool, got "
                f"{type(uses_navigation_app).__name__}"
            )
        if start_time < 0:
            raise ValueError(f"start_time must be >= 0, got {start_time}")
        if origin == destination:
            raise ValueError(
                f"origin and destination cannot be the same: {origin!r}"
            )

        self.id = vehicle_id
        self.vehicle_id = vehicle_id
        self.origin = origin
        self.destination = destination
        self.start_time = start_time
        self.uses_navigation_app = uses_navigation_app

        # Journey state
        self.current_position = origin
        self.route: list[Any] | None = None
        self.completion_time: int | None = None

    @classmethod
    def create_for_network(
        cls,
        graph: nx.DiGraph,
        vehicle_id: str | int,
        origin: Any,
        destination: Any,
        start_time: int = 0,
        uses_navigation_app: bool = False,
    ) -> Vehicle:
        """Create a Vehicle agent after validating that origin and destination exist in the network.

        Parameters
        ----------
        graph:
            The NetworkX directed graph representing the road network.
        vehicle_id:
            A unique identifier, either a string or an integer.
        origin:
            The starting node. Must exist in ``graph.nodes``.
        destination:
            The ending node. Must exist in ``graph.nodes``.
        start_time:
            The starting time-step. Defaults to 0.
        uses_navigation_app:
            Forwarded to the constructor. Defaults to ``False``.

        Returns
        -------
        Vehicle
            A validated Vehicle instance.

        Raises
        ------
        ValueError
            If ``origin`` or ``destination`` is not in ``graph.nodes``.
        """
        if origin not in graph.nodes:
            raise ValueError(f"origin {origin!r} is not a valid node in the graph")
        if destination not in graph.nodes:
            raise ValueError(f"destination {destination!r} is not a valid node in the graph")
        return cls(
            vehicle_id=vehicle_id,
            origin=origin,
            destination=destination,
            start_time=start_time,
            uses_navigation_app=uses_navigation_app,
        )

    def set_route(self, route: list[Any], graph: nx.DiGraph | None = None) -> None:
        """Set the vehicle's planned path through the network.

        Parameters
        ----------
        route:
            A list of nodes representing the planned path.
        graph:
            An optional NetworkX graph to validate node existence against.

        Raises
        ------
        ValueError
            If ``route`` is empty.
            If ``route[0]`` is not the vehicle's origin.
            If ``route[-1]`` is not the vehicle's destination.
            If a graph is provided and any node in the route does not exist.
            If the vehicle's current position is not in the assigned route.
        """
        if not route:
            raise ValueError("Route cannot be empty")

        if route[0] != self.origin:
            raise ValueError(
                f"Route must start at origin {self.origin!r}, got {route[0]!r}"
            )
        if route[-1] != self.destination:
            raise ValueError(
                f"Route must end at destination {self.destination!r}, got {route[-1]!r}"
            )

        if graph is not None:
            for node in route:
                if node not in graph.nodes:
                    raise ValueError(f"Node {node!r} in route does not exist in the graph")

        if self.current_position not in route:
            raise ValueError(
                f"Current position {self.current_position!r} is not in the assigned route"
            )

        self.route = list(route)

    def update_position(self, node: Any, graph: nx.DiGraph | None = None) -> None:
        """Update the vehicle's current position.

        Parameters
        ----------
        node:
            The target node to move the vehicle to.
        graph:
            An optional NetworkX graph to validate node existence against.

        Raises
        ------
        ValueError
            If a route is assigned and ``node`` is not part of the route.
            If a graph is provided and ``node`` does not exist in the graph.
        """
        if self.route is not None:
            if node not in self.route:
                raise ValueError(
                    f"Target node {node!r} is not part of the assigned route"
                )

        if graph is not None:
            if node not in graph.nodes:
                raise ValueError(f"Target node {node!r} does not exist in the graph")

        self.current_position = node

    def complete_journey(self, completion_time: int) -> None:
        """Mark the vehicle's journey as complete and record the completion time.

        Parameters
        ----------
        completion_time:
            The discrete simulation time-step when the journey was completed.

        Raises
        ------
        TypeError
            If ``completion_time`` is not an integer.
        ValueError
            If ``completion_time`` is less than ``start_time``.
            If the vehicle's current position is not its destination.
        """
        if not isinstance(completion_time, int) or isinstance(completion_time, bool):
            raise TypeError(
                f"completion_time must be an integer, got {type(completion_time).__name__}"
            )
        if completion_time < self.start_time:
            raise ValueError(
                f"completion_time {completion_time} cannot be less than start_time {self.start_time}"
            )
        if self.current_position != self.destination:
            raise ValueError(
                f"Cannot complete journey: current position {self.current_position!r} "
                f"is not the destination {self.destination!r}"
            )

        self.completion_time = completion_time
