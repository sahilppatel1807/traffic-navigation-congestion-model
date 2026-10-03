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

When ``routing_policy="shared_navigation"`` is set, navigation users due on
the same step are grouped by origin/destination and receive one route from a
single pre-entry traffic snapshot. Uninformed vehicles still use static
free-flow routing. Routes remain locked after entry.

When ``routing_policy="coordinated"`` is set on a :class:`Simulation`, a
central controller groups vehicles releasing on the same step, enumerates
candidate route distributions, evaluates each by running a deep-copied
sub-simulation to completion, and assigns the distribution that minimises the
sum of journey times for the releasing cohort. Tie-breaks prefer maximum
agreement with selfish routes then lexicographically smallest path lists.
"""

from __future__ import annotations

import copy
import itertools
import math
from math import comb
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

    def __init__(self, graph: nx.DiGraph, vehicles: list[Vehicle], routing_policy: str = "decentralized") -> None:
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
        routing_policy:
            The routing policy to use for unrouted vehicles at release.
            Defaults to ``"decentralized"``. Use ``"shared_navigation"`` for
            synchronized live recommendations or ``"coordinated"`` for the
            central coordinated routing policy.

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
        self.routing_policy = routing_policy
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
        if self.routing_policy == "coordinated":
            self._enter_due_vehicles_coordinated()
        elif self.routing_policy == "shared_navigation":
            self._enter_due_vehicles_shared_navigation()
        else:
            self._enter_due_vehicles_decentralized()

    def _enter_due_vehicles_shared_navigation(self) -> None:
        """Enter due vehicles using synchronized recommendations for nav users.

        All live routes are calculated before any due vehicle enters. This is
        the shared traffic snapshot: vehicles in one origin/destination group
        receive the same route, while uninformed vehicles retain static
        free-flow routing. A route already assigned by a caller remains
        locked, matching the behaviour of the other policies.
        """
        due = [
            vehicle
            for vehicle in self.vehicles
            if vehicle not in self._in_transit
            and vehicle not in self.completed
            and vehicle.start_time == self.current_step
        ]

        shared_routes: dict[tuple[Any, Any], list[Any]] = {}
        for vehicle in due:
            if not vehicle.uses_navigation_app or vehicle.route is not None:
                continue
            key = (vehicle.origin, vehicle.destination)
            if key not in shared_routes:
                shared_routes[key] = find_shortest_route(
                    self.graph,
                    vehicle.origin,
                    vehicle.destination,
                    weight="current_travel_time",
                )

        for vehicle in due:
            if vehicle.route is None:
                route = (
                    shared_routes.get((vehicle.origin, vehicle.destination))
                    if vehicle.uses_navigation_app
                    else None
                )
                if route is None:
                    route = find_shortest_route(
                        self.graph,
                        vehicle.origin,
                        vehicle.destination,
                        weight="free_flow_time",
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

    def _enter_due_vehicles_decentralized(self) -> None:
        """Decentralized (selfish/uninformed) entry: route each vehicle independently."""
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

    def _enter_due_vehicles_coordinated(self) -> None:
        """Coordinated entry: central controller assigns routes to minimize total journey time.

        Algorithm
        ---------
        1. Collect all vehicles releasing this step.
        2. For each vehicle, list all simple paths (origin → destination).
           Raise ``ValueError`` if any vehicle has no path.
        3. Sort each vehicle's paths: selfish path (Dijkstra on
           ``current_travel_time``) at index 0, rest sorted lexicographically.
        4. Enumerate all distributions of vehicles over paths within each
           (origin, destination) group. A distribution specifies how many
           vehicles take each path; vehicles are assigned in order (first
           ``c_0`` take path 0, next ``c_1`` take path 1, …).
        5. The Cartesian product across groups must not exceed 32 768.
           Raise ``ValueError`` if it does.
        6. For each candidate global assignment, deep-copy the simulation,
           pre-set routes, and run until all releasing vehicles complete.
           Raise ``ValueError`` if any releasing vehicle fails to complete.
        7. Choose the assignment minimising the sum of journey times. Ties
           broken by: (a) maximise selfish-route agreement; (b) minimise the
           concatenated route node lists lexicographically.
        8. Apply the chosen routes on the real vehicle agents and enter them.
        """
        # --- 1. Collect releasing vehicles (preserve list order) ---
        releasing: list[Vehicle] = []
        for vehicle in self.vehicles:
            if vehicle in self._in_transit or vehicle in self.completed:
                continue
            if vehicle.start_time == self.current_step:
                releasing.append(vehicle)

        if not releasing:
            return

        # --- 2 & 3. Per-vehicle: all simple paths, sorted with selfish first ---
        selfish_routes: list[list[Any]] = []
        all_paths_per_vehicle: list[list[list[Any]]] = []
        for vehicle in releasing:
            if vehicle.route is not None:
                # Pre-set route: treat it as the only option.
                selfish_routes.append(list(vehicle.route))
                all_paths_per_vehicle.append([list(vehicle.route)])
                continue

            # Selfish route (index 0).
            selfish = find_shortest_route(
                self.graph, vehicle.origin, vehicle.destination,
                weight="current_travel_time"
            )
            selfish_routes.append(selfish)

            # All simple paths.
            try:
                raw_paths = [
                    path for path in nx.all_simple_paths(
                        self.graph, vehicle.origin, vehicle.destination
                    )
                    if all(
                        not self.graph[u][v].get("closed", False)
                        for u, v in zip(path, path[1:])
                    )
                ]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raw_paths = []

            if not raw_paths:
                raise ValueError(
                    f"no path exists for vehicle {vehicle.vehicle_id!r} "
                    f"from {vehicle.origin!r} to {vehicle.destination!r}"
                )

            # Sort: selfish first, then others lexicographically.
            others = sorted(
                [p for p in raw_paths if p != selfish],
                key=lambda p: [str(n) for n in p],
            )
            sorted_paths = [selfish] + others
            all_paths_per_vehicle.append(sorted_paths)

        # --- 4 & 5. Group by (origin, destination) and build distributions ---
        # Map each releasing vehicle to its index in `releasing`.
        od_groups: dict[tuple[Any, Any], list[int]] = {}
        for idx, vehicle in enumerate(releasing):
            key = (vehicle.origin, vehicle.destination)
            od_groups.setdefault(key, []).append(idx)

        # For each OD group, enumerate all distributions (stars-and-bars):
        # how many vehicles take path 0, path 1, etc.
        # A distribution is a tuple of integers summing to n.
        def _distributions(n: int, k: int) -> list[tuple[int, ...]]:
            """All non-negative integer tuples of length k summing to n."""
            if k == 1:
                return [(n,)]
            result = []
            for first in range(n + 1):
                for rest in _distributions(n - first, k - 1):
                    result.append((first,) + rest)
            return result

        # Collect per-group distribution lists and vehicle-index lists.
        group_keys = list(od_groups.keys())
        group_vehicle_indices: list[list[int]] = [od_groups[k] for k in group_keys]
        group_paths: list[list[list[Any]]] = []

        # --- Check search-space size using stars-and-bars formula BEFORE enumeration ---
        # Number of distributions for n vehicles over k paths = C(n+k-1, k-1).
        _MAX_COMBINATIONS = 32_768
        total_combinations = 1
        for vid_indices in group_vehicle_indices:
            rep_idx = vid_indices[0]
            paths = all_paths_per_vehicle[rep_idx]
            n, k = len(vid_indices), len(paths)
            group_dist_count = comb(n + k - 1, k - 1)
            total_combinations *= group_dist_count
            if total_combinations > _MAX_COMBINATIONS:
                raise ValueError(
                    f"coordinated routing search space ({total_combinations}+ combinations) "
                    f"exceeds the limit of {_MAX_COMBINATIONS}"
                )

        # Now actually enumerate distributions (safe since total_combinations <= limit).
        group_distributions: list[list[tuple[int, ...]]] = []
        for vid_indices in group_vehicle_indices:
            rep_idx = vid_indices[0]
            paths = all_paths_per_vehicle[rep_idx]
            group_paths.append(paths)
            group_distributions.append(_distributions(len(vid_indices), len(paths)))

        # --- 6. Evaluate each global assignment ---
        releasing_ids = {v.vehicle_id for v in releasing}

        # Determine a sub-simulation step limit: generous upper bound.
        _SUB_SIM_LIMIT = max(200, 10 * len(releasing))

        best_assignment: list[list[Any]] | None = None
        best_total_time: float = float("inf")
        best_selfish_agreement: int = -1
        best_route_key: list[list[Any]] = []

        for dist_combo in itertools.product(*group_distributions):
            # dist_combo[i] is the distribution tuple for group i.
            # Build the vehicle-index → route mapping.
            assignment: dict[int, list[Any]] = {}
            for group_i, dist in enumerate(dist_combo):
                vid_indices = group_vehicle_indices[group_i]
                paths = group_paths[group_i]
                pointer = 0
                for path_j, count in enumerate(dist):
                    for _ in range(count):
                        assignment[vid_indices[pointer]] = paths[path_j]
                        pointer += 1

            # Build the flattened route list in releasing order.
            assigned_routes: list[list[Any]] = [assignment[i] for i in range(len(releasing))]

            # Deep-copy the simulation and pre-set routes.
            sim_copy = copy.deepcopy(self)
            sim_copy.routing_policy = "decentralized"  # sub-sim uses decentralized
            releasing_copy = [
                v for v in sim_copy.vehicles
                if v.vehicle_id in releasing_ids
                and v not in sim_copy._in_transit
                and v not in sim_copy.completed
                and v.start_time == sim_copy.current_step
            ]
            # Sort releasing_copy in same order as releasing (by vehicle_id match).
            id_to_copy: dict[Any, Vehicle] = {v.vehicle_id: v for v in releasing_copy}
            for i, real_vehicle in enumerate(releasing):
                copy_vehicle = id_to_copy.get(real_vehicle.vehicle_id)
                if copy_vehicle is not None:
                    copy_vehicle.set_route(assigned_routes[i], sim_copy.graph)

            # Run sub-simulation until all releasing vehicles complete.
            sim_copy.run(until=_SUB_SIM_LIMIT)

            # Collect journey times for releasing vehicles.
            total_time = 0.0
            for real_vehicle in releasing:
                copy_vehicle = id_to_copy.get(real_vehicle.vehicle_id)
                if copy_vehicle is None or copy_vehicle.completion_time is None:
                    raise ValueError(
                        f"coordinated routing: vehicle {real_vehicle.vehicle_id!r} "
                        f"did not complete within {_SUB_SIM_LIMIT} sub-simulation steps"
                    )
                total_time += copy_vehicle.completion_time - copy_vehicle.start_time

            # Compute selfish-route agreement count.
            selfish_agreement = sum(
                1 for i, route in enumerate(assigned_routes)
                if route == selfish_routes[i]
            )

            # Compare: minimize total time → maximize selfish agreement →
            # minimize route lists lexicographically (compare as list of lists).
            is_better = False
            if total_time < best_total_time:
                is_better = True
            elif total_time == best_total_time:
                if selfish_agreement > best_selfish_agreement:
                    is_better = True
                elif selfish_agreement == best_selfish_agreement:
                    if assigned_routes < best_route_key:
                        is_better = True

            if is_better:
                best_assignment = assigned_routes
                best_total_time = total_time
                best_selfish_agreement = selfish_agreement
                best_route_key = assigned_routes

        assert best_assignment is not None

        # --- 7. Apply chosen routes and enter vehicles ---
        for vehicle, route in zip(releasing, best_assignment):
            vehicle.set_route(route, self.graph)
            if len(vehicle.route) < 2:  # type: ignore[arg-type]
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
