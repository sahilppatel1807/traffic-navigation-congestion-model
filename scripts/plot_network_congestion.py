#!/usr/bin/env python3
"""Deterministic Week 9 demo: congested network figure under results/.

Builds the default network, stacks hard-coded seed vehicles on an overlapping
corridor (bottleneck intensity), runs a short Simulation, then saves
``results/network_congestion.png``.

Run from the repository root::

    python scripts/plot_network_congestion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.network import create_default_network
from src.routing import find_shortest_route
from src.simulation import Simulation
from src.vehicle import Vehicle
from src.visualisation import plot_network_congestion

# Fixed demo recipe — no randomness.
DEMO_STEP_COUNT = 1
DEMO_OUTPUT = _REPO_ROOT / "results" / "network_congestion.png"
# Capacity on 0→1 / 1→2 is 10; ten vehicles on the same corridor → ratio 1.0.
DEMO_CORRIDOR_ORIGIN = 0
DEMO_CORRIDOR_DESTINATION = 2
DEMO_VEHICLE_COUNT = 10


def build_demo_vehicles(graph) -> list[Vehicle]:
    """Hard-coded seed vehicles sharing the top corridor 0→1→2."""
    route = find_shortest_route(
        graph,
        DEMO_CORRIDOR_ORIGIN,
        DEMO_CORRIDOR_DESTINATION,
        weight="free_flow_time",
    )
    vehicles: list[Vehicle] = []
    for i in range(DEMO_VEHICLE_COUNT):
        vehicle = Vehicle.create_for_network(
            graph,
            vehicle_id=f"demo-{i}",
            origin=DEMO_CORRIDOR_ORIGIN,
            destination=DEMO_CORRIDOR_DESTINATION,
            start_time=0,
        )
        vehicle.set_route(route, graph)
        vehicles.append(vehicle)
    return vehicles


def main() -> None:
    graph = create_default_network()
    vehicles = build_demo_vehicles(graph)
    simulation = Simulation(graph, vehicles)
    # After one step, free-flow dwell on 0→1 expires and all ten vehicles sit on
    # 1→2 at occupancy == capacity (ratio 1.0) — the hot corridor for the figure.
    simulation.run(until=DEMO_STEP_COUNT)
    plot_network_congestion(simulation.graph, DEMO_OUTPUT)
    print(f"Wrote {DEMO_OUTPUT}")


if __name__ == "__main__":
    main()
