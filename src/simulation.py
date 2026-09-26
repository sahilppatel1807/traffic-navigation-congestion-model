"""Discrete simulation clock and vehicle movement on a directed road network.

Callers construct a :class:`Simulation` with a graph and a flat list of
pre-built :class:`~src.vehicle.Vehicle` agents. Each :meth:`Simulation.step`
releases due vehicles, advances in-transit vehicles along their routes,
updates edge occupancy, refreshes congested travel times, then advances the
clock by one. :meth:`Simulation.run` drives the clock with a soft step cap.

In-transit edge and remaining-dwell state live in a private simulation map —
the vehicle agent remains a pure journey-state object. Edge dwell is taken from
the travel time implied by vehicles already on the road; occupancy is then
incremented so later simultaneous entrants see congestion. The dwell assigned
at entry is kept on that private record so :meth:`Simulation.in_transit_snapshot`
can still report it after the entry tick is consumed. Reading the snapshot
does not advance the clock or change occupancy, routes, or completion.

Entry auto-routing chooses Dijkstra weights from each vehicle's
``uses_navigation_app`` flag (free-flow vs live congested travel time). Routes
are locked at entry; same-step due vehicles are still routed then entered in
list order, so later selfish entrants can see earlier occupancy.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import networkx as nx

from src.congestion import update_all_travel_times, update_road_travel_time
from src.routing import find_shortest_route
from src.vehicle import Vehicle


class InTransitRecord(NamedTuple):
    """One vehicle currently on a directed edge.

    Pending and completed vehicles are omitted. ``assigned_dwell`` is the dwell
    set when the vehicle entered ``edge`` and does not shrink as ticks are
    consumed. ``steps_remaining`` is the dwell still left and is at least 1
    while the record exists. Position along the edge is
    ``(assigned_dwell - steps_remaining) / assigned_dwell``.
    """

    vehicle_id: str | int
    edge: tuple[Any, Any]
    assigned_dwell: int
    steps_remaining: int


class _Transit(NamedTuple):
    """Private in-transit bookkeeping for one active vehicle."""

    edge_u: Any
    edge_v: Any
    route_idx: int
    steps_remaining: int
    assigned_dwell: int


class Simulation:
    """Discrete-time traffic simulation over a directed road graph.

    Public attributes
    -----------------
    graph:
        The road network being simulated.
    vehicles:
        Caller-supplied vehicle list (includes pending, active, and completed).
    current_step:
        Discrete clock; starts at ``0``. Each ``step()`` processes the current
        value then increments by one.
    active:
        Vehicles currently in transit on an edge (not pending, not completed).
    completed:
        Vehicles that have finished their journeys.

    ``in_transit_snapshot()`` returns read-only records for vehicles currently
    on an edge, in ``active`` order.
    """

    def __init__(self, graph: nx.DiGraph, vehicles: list[Vehicle]) -> None:
        """Create a simulation from a graph and a list of vehicles.

        Parameters
        ----------
        graph:
            Directed road network with edge attributes used by congestion and
            routing (``free_flow_time``, ``capacity``, ``occupancy``,
            ``current_travel_time``).
        vehicles:
            Pre-constructed vehicles. Already-completed agents
            (``completion_time is not None``) are rejected.

        Raises
        ------
        ValueError
            If any vehicle already has a non-``None`` ``completion_time``.
        """
        for vehicle in vehicles:
            if vehicle.completion_time is not None:
                raise ValueError(
                    f"vehicle {vehicle.vehicle_id!r} is already completed "
                    f"(completion_time={vehicle.completion_time})"
                )

        self.graph = graph
        self.vehicles = vehicles
        self.current_step = 0
        self.active: list[Vehicle] = []
        self.completed: list[Vehicle] = []

        # Private in-transit map. route_idx is the index of edge_u in
        # vehicle.route (avoids first-match scans). assigned_dwell is fixed
        # when the vehicle enters the edge.
        self._in_transit: dict[Vehicle, _Transit] = {}

    def step(self) -> None:
        """Process the current step, then advance the clock by one.

        Phase order (strict):
        1. Enter all vehicles due at ``current_step``.
        2. Advance all in-transit vehicles (consume dwell; transfer or complete).
        3. Refresh all edge travel times after occupancy changes.
        """
        self._enter_due_vehicles()
        self._advance_in_transit()
        update_all_travel_times(self.graph)
        self.current_step += 1

    def run(self, until: int) -> list[Vehicle]:
        """Execute up to ``until`` steps and return completed vehicles.

        Stops early when no vehicles are active and no remaining vehicle has
        ``start_time >= current_step``. At most ``until`` steps are processed.

        Parameters
        ----------
        until:
            Soft upper bound on the number of ``step()`` calls. Must be >= 1.

        Returns
        -------
        list[Vehicle]
            Vehicles that completed during the run (same as ``self.completed``).

        Raises
        ------
        ValueError
            If ``until < 1``.
        """
        if until < 1:
            raise ValueError(f"until must be >= 1, got {until}")

        steps_done = 0
        while steps_done < until:
            if not self.active and not self._has_pending_or_future():
                break
            self.step()
            steps_done += 1

        return list(self.completed)

    def in_transit_snapshot(self) -> tuple[InTransitRecord, ...]:
        """Return vehicles currently on an edge, in active-vehicle order.

        Each record names the vehicle, the directed edge it occupies, the dwell
        assigned when it entered that edge, and the steps still remaining.
        Vehicles that have not entered and vehicles that have finished are
        omitted. On the opening frame, before the first step, the result is
        empty.

        This read does not advance ``current_step`` or change occupancy,
        routes, or completion.
        """
        records: list[InTransitRecord] = []
        for vehicle in self.active:
            state = self._in_transit[vehicle]
            records.append(
                InTransitRecord(
                    vehicle_id=vehicle.vehicle_id,
                    edge=(state.edge_u, state.edge_v),
                    assigned_dwell=state.assigned_dwell,
                    steps_remaining=state.steps_remaining,
                )
            )
        return tuple(records)

    def _has_pending_or_future(self) -> bool:
        """True if any vehicle has not yet entered and is still due to enter."""
        for vehicle in self.vehicles:
            if vehicle in self._in_transit or vehicle in self.completed:
                continue
            if vehicle.start_time >= self.current_step:
                return True
        return False

    def _enter_due_vehicles(self) -> None:
        """Release vehicles whose ``start_time`` equals ``current_step``."""
        for vehicle in self.vehicles:
            if vehicle in self._in_transit or vehicle in self.completed:
                continue
            if vehicle.start_time != self.current_step:
                continue

            if vehicle.route is None:
                weight = (
                    "current_travel_time"
                    if vehicle.uses_navigation_app
                    else "free_flow_time"
                )
                route = find_shortest_route(
                    self.graph,
                    vehicle.origin,
                    vehicle.destination,
                    weight=weight,
                )
                vehicle.set_route(route, self.graph)

            assert vehicle.route is not None
            if len(vehicle.route) < 2:
                raise ValueError(
                    f"vehicle {vehicle.vehicle_id!r} route must contain at "
                    f"least two nodes, got {vehicle.route!r}"
                )

            u, v = vehicle.route[0], vehicle.route[1]
            self._enter_edge(vehicle, u, v, route_idx=0)
            self.active.append(vehicle)

    def _enter_edge(
        self, vehicle: Vehicle, u: Any, v: Any, *, route_idx: int
    ) -> None:
        """Place ``vehicle`` onto directed edge ``u → v`` and set dwell.

        Dwell uses the edge's travel time from vehicles **already** on the road
        (excluding this entrant). Occupancy is then incremented and the edge
        travel time refreshed so later simultaneous entrants see congestion.
        """
        if not self.graph.has_edge(u, v):
            raise ValueError(
                f"missing directed edge {u!r} → {v!r} on route for vehicle "
                f"{vehicle.vehicle_id!r}"
            )

        edge = self.graph[u][v]
        update_road_travel_time(edge)
        travel_time = edge["current_travel_time"]
        assigned_dwell = max(1, math.ceil(travel_time))
        edge["occupancy"] += 1
        update_road_travel_time(edge)
        self._in_transit[vehicle] = _Transit(
            edge_u=u,
            edge_v=v,
            route_idx=route_idx,
            steps_remaining=assigned_dwell,
            assigned_dwell=assigned_dwell,
        )

    def _leave_edge(self, vehicle: Vehicle, u: Any, v: Any) -> None:
        """Decrement occupancy and clear in-transit state for ``vehicle``."""
        self.graph[u][v]["occupancy"] -= 1
        del self._in_transit[vehicle]

    def _advance_in_transit(self) -> None:
        """Consume one dwell tick for each active vehicle; transfer or complete.

        Vehicles that entered an edge in this step (phase 1) consume their first
        dwell tick here. Intermediate transfers leave the previous edge and enter
        the next in the same advance, but do **not** consume a further tick on
        the new edge until a later step — so each free-flow edge of travel time
        ``1`` takes exactly one simulation step.
        """
        # Snapshot so transfers that re-enter _in_transit are not advanced again.
        to_advance = list(self.active)

        for vehicle in to_advance:
            if vehicle not in self._in_transit:
                continue

            state = self._in_transit[vehicle]
            u, v, route_idx = state.edge_u, state.edge_v, state.route_idx
            steps_remaining = state.steps_remaining - 1

            if steps_remaining > 0:
                self._in_transit[vehicle] = state._replace(
                    steps_remaining=steps_remaining
                )
                continue

            # Dwell expired: leave current edge and arrive at node v.
            self._leave_edge(vehicle, u, v)
            vehicle.update_position(v, self.graph)

            assert vehicle.route is not None
            if v == vehicle.destination:
                vehicle.complete_journey(self.current_step)
                self.active.remove(vehicle)
                self.completed.append(vehicle)
                continue

            # Instantaneous transfer onto the next edge; no further tick this step.
            next_node = vehicle.route[route_idx + 2]
            self._enter_edge(vehicle, v, next_node, route_idx=route_idx + 1)
