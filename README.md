# Traffic Navigation and Congestion Model

An agent-based computational model that investigates whether real-time navigation information improves traffic conditions or, when widely adopted, creates new congestion bottlenecks.

## Research question

**How do traffic demand and real-time navigation-app adoption affect system-wide congestion and travel times after a road disruption?**

## Motivation

Traffic congestion is an emergent property of a road network: each driver makes an individual route choice, but the combined choices determine congestion for everyone. Navigation applications can help a driver avoid a busy road. However, if many drivers receive the same recommendation, they may reroute to the same alternative road and create a new bottleneck.

This project uses a small synthetic road network rather than a real city map. This makes the assumptions explicit and lets the effect of routing decisions be studied systematically.

## Hypothesis

Real-time navigation will improve travel times at low and moderate adoption rates. At high adoption rates, simultaneous rerouting may concentrate vehicles on the same roads and increase total network delay, particularly after a road disruption.

## Model overview

The model represents a road network as a graph:

- **Nodes** represent intersections.
- **Edges** represent roads with a free-flow travel time and capacity.
- **Vehicles** are individual agents with an origin, destination, route, and travel time.
- Road travel times increase as the number of vehicles approaches or exceeds capacity.
- A disruption is represented by reducing the capacity of a road or closing it.

At each simulation time step, new vehicles enter the network, choose a route, travel along roads, and completed trips are recorded.

The default topology is a small synthetic six-node corridor with two parallel paths:

```text
0 <----> 1 <----> 2
^                   ^
|                   |
v                   v
3 <----> 4 <----> 5
```

It is created by `src.network.create_default_network()` as a reproducible `networkx.DiGraph`. Two-way roads are represented explicitly as opposite directed edges. Each edge stores `free_flow_time`, `capacity`, `occupancy`, and `current_travel_time`. Time is measured in simulation steps; capacity and occupancy are vehicle counts. Use `src.network.summarize_network()` for a compact debug summary.

## Congestion model

Road travel times are updated using the **Bureau of Public Roads (BPR)** formula:

```
t = t0 × (1 + α × (occupancy / capacity) ^ β)
```

| Symbol | Meaning | Default |
|---|---|---|
| `t0` | Free-flow travel time (simulation steps) | per edge |
| `occupancy` | Vehicles currently on the road | per edge |
| `capacity` | Design-capacity vehicle count | per edge |
| `α` (`alpha`) | Sensitivity coefficient | `0.15` |
| `β` (`beta`) | Shape exponent | `4.0` |

At zero occupancy, `t` equals `t0` exactly. Travel time is non-decreasing as occupancy rises. At capacity (`occupancy == capacity`) the penalty is `+15%`; at twice capacity it reaches `+240%` with default parameters.

### `src/congestion.py` — public API

```python
from src.congestion import (
    DEFAULT_ALPHA,          # 0.15
    DEFAULT_BETA,           # 4.0
    calculate_travel_time,
    update_road_travel_time,
    update_all_travel_times,
)
```

**`calculate_travel_time(free_flow_time, capacity, occupancy, alpha, beta) → float`**
Pure BPR calculator. Raises `ValueError` for `capacity <= 0`, `occupancy < 0`, or `free_flow_time < 0`.

**`update_road_travel_time(road, alpha, beta) → None`**
Reads `free_flow_time`, `capacity`, and `occupancy` from a NetworkX edge-attribute `dict` and writes the result to `road["current_travel_time"]`.

**`update_all_travel_times(graph, alpha, beta) → None`**
Iterates over all edges of a `networkx.DiGraph` and calls `update_road_travel_time` on each. Call this after any step in which vehicle occupancy changes.

## Vehicle Agents

Individual vehicles are represented as stateful agents that track their identity, trip parameters, and journey status while enforcing strict physical invariants.

### `src/vehicle.py` — public API

```python
from src.vehicle import Vehicle
```

**`Vehicle(vehicle_id: str | int, origin: Any, destination: Any, start_time: int = 0)`**
Pure constructor that initializes a vehicle.
- Raises `TypeError` if `vehicle_id` is not a string or integer, or if `start_time` is not an integer.
- Raises `ValueError` if `origin == destination` or if `start_time < 0`.
- Sets initial journey state: `current_position` at `origin`, `route` to `None`, and `completion_time` to `None`.

**`Vehicle.create_for_network(graph, vehicle_id, origin, destination, start_time=0) → Vehicle`**
Factory classmethod that verifies that the `origin` and `destination` nodes exist in the NetworkX directed graph before constructing the vehicle. Raises `ValueError` if either node is missing.

**`set_route(route: list[Any], graph: nx.DiGraph | None = None) -> None`**
Sets the vehicle's planned path.
- Raises `ValueError` if the route does not start at `origin` or end at `destination`.
- If `graph` is provided, raises `ValueError` if any node in the route is not in `graph.nodes`.
- Raises `ValueError` if the vehicle's `current_position` is not in the assigned route.

**`update_position(node: Any, graph: nx.DiGraph | None = None) -> None`**
Updates the vehicle's `current_position`.
- If a route is assigned, raises `ValueError` if `node` is not part of the route.
- If `graph` is provided, raises `ValueError` if `node` does not exist in `graph.nodes`.

**`complete_journey(completion_time: int) -> None`**
Marks the journey as complete and records the completion time-step.
- Raises `TypeError` if `completion_time` is not an integer.
- Raises `ValueError` if `completion_time` is less than `start_time` or if the vehicle is not currently at its `destination`.

## Routing behaviours

### Uninformed routing

Drivers choose the shortest route using normal, uncongested road travel times. They do not receive live traffic updates.

### `src/routing.py` — public API (static / uninformed)

```python
from src.routing import find_shortest_route
```

**`find_shortest_route(graph, origin, destination, weight="free_flow_time") → list`**
Pure Dijkstra shortest path on a `networkx.DiGraph`. Returns an ordered node list from `origin` to `destination` inclusive — suitable for `Vehicle.set_route`. Does not mutate the graph or assign routes to vehicles.

- Default `weight="free_flow_time"` implements **uninformed / static routing** (shortest path on free-flow times).
- Optional `weight` (e.g. `"current_travel_time"`) is reserved for later selfish routing; every edge must carry the chosen attribute.
- Raises `ValueError` if `origin` or `destination` is missing, if they are equal, or if no directed path exists.

### Selfish real-time routing

Navigation-app users choose the route with the lowest *currently estimated personal travel time*. This is decentralised routing: every driver tries to minimise their own trip time. Not implemented yet; the same `find_shortest_route` helper is intended to be reused with `weight="current_travel_time"`.

### Coordinated routing (extension)

A central controller assigns or recommends routes to minimise total travel time across all vehicles. It may assign a car to a slightly slower route if that prevents a major bottleneck and improves network-wide performance.

## Simulation clock and vehicle movement

### `src/simulation.py` — public API

```python
from src.simulation import Simulation
```

**`Simulation(graph, vehicles)`**
Constructs a discrete-time simulation from a directed road graph and a flat list of pre-built `Vehicle` agents. Public attributes: `graph`, `vehicles`, `current_step` (starts at `0`), `active` (in-transit only), and `completed`. Raises `ValueError` if any vehicle already has a non-`None` `completion_time`.

**`step() → None`**
Processes the current clock value then increments by one. Phase order: (1) enter vehicles due at `current_step` (auto-route with `free_flow_time` if needed; place onto first edge); (2) advance in-transit vehicles (consume dwell; leave/enter edges or complete); (3) refresh all edge travel times via BPR. Mutates state in place; returns `None`.

**`run(until: int) → list[Vehicle]`**
Executes up to `until` steps. Stops early when nothing is active and no remaining vehicle has `start_time >= current_step`. Returns the list of completed vehicles. Raises `ValueError` if `until < 1`.

Edge dwell uses `max(1, ceil(current_travel_time))` ticks; the first tick is consumed in the same step a vehicle enters an edge from the release phase. Intermediate node transfers are instantaneous (leave previous edge and enter the next in the same advance) without cascading extra dwell ticks, so a free-flow edge of travel time `1` takes exactly one simulation step. In-transit timers are stored inside the simulation, not on the vehicle agent.

## Experimental design

The following factors will be varied systematically:

| Factor | Scenarios |
| --- | --- |
| Traffic demand | Low, medium, high |
| Navigation-app adoption | 0%, 25%, 50%, 75%, 100% |
| Disruption | None, reduced road capacity, road closure |
| Routing policy | Uninformed, selfish real-time, coordinated (extension) |

Each scenario will be repeated using multiple random seeds to account for stochastic variation.

## Measures

The model will record:

- mean journey time;
- median and 95th-percentile journey time;
- total network delay;
- road-level congestion over time;
- congestion recovery time after a disruption; and
- the difference between individual and network-wide routing outcomes.

## Expected outputs

- Network diagrams coloured by congestion level.
- Time-series plots of congestion after a disruption.
- Average travel time versus navigation-app adoption rate.
- Comparisons of selfish and coordinated routing across demand levels.

## Project status

**Current stage:** Synthetic network topology, BPR congestion travel-time calculation, vehicle agents, static (uninformed) shortest-path routing, and the discrete simulation clock with vehicle movement are implemented. Selfish / coordinated routing policies, demand generation, metrics, and road disruptions are not implemented yet.

The first modelling milestone is a working simulation in which vehicles travel through a capacity-constrained network and rising demand produces rising travel times — the network, congestion, vehicle, static-routing, and simulation-clock modules are now in place. The next milestones are real-time route choice, metrics aggregation, and a road-disruption scenario.

## Repository layout

```text
src/            Simulation source code
notebooks/      Experiment analysis and visualisations
results/        Generated data and figures
tests/          Automated tests
report/         Project report and supporting material
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

At this stage, `pytest` runs the project smoke check, network topology tests, congestion calculation tests, vehicle agent tests, static routing tests, and simulation clock / vehicle movement tests. Additional model behaviour tests will be added with later issues.

## Reproducibility

The final repository will include installation instructions, a fixed random-seed option, experiment configuration files, and scripts to regenerate all reported results and figures.