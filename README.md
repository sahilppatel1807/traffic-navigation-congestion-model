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
- A disruption is represented by reducing the capacity of a road (via `src.disruption.reduce_road_capacity`) or fully closing it (via `src.disruption.close_road`).

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

**`Vehicle(vehicle_id: str | int, origin: Any, destination: Any, start_time: int = 0, uses_navigation_app: bool = False)`**
Pure constructor that initializes a vehicle.
- Raises `TypeError` if `vehicle_id` is not a string or integer, if `start_time` is not an integer, or if `uses_navigation_app` is not a real `bool` (integers and other truthy values are rejected).
- Raises `ValueError` if `origin == destination` or if `start_time < 0`.
- Sets initial journey state: `current_position` at `origin`, `route` to `None`, `completion_time` to `None`, and `uses_navigation_app` (default `False` = uninformed).

**`Vehicle.create_for_network(graph, vehicle_id, origin, destination, start_time=0, uses_navigation_app=False) → Vehicle`**
Factory classmethod that verifies that the `origin` and `destination` nodes exist in the NetworkX directed graph before constructing the vehicle. Forwards `uses_navigation_app` to the constructor. Raises `ValueError` if either node is missing.

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

### `src/routing.py` — public API (static / uninformed + route-cost)

```python
from src.routing import find_shortest_route, estimate_route_cost
```

**`find_shortest_route(graph, origin, destination, weight="free_flow_time") → list`**
Pure Dijkstra shortest path on a `networkx.DiGraph`. Returns an ordered node list from `origin` to `destination` inclusive — suitable for `Vehicle.set_route`. Does not mutate the graph or assign routes to vehicles.

- Default `weight="free_flow_time"` implements **uninformed / static routing** (shortest path on free-flow times).
- Optional `weight` (e.g. `"current_travel_time"`) is used by **selfish / navigation-app** entry auto-routing; every edge must carry the chosen attribute.
- Raises `ValueError` if `origin` or `destination` is missing, if they are equal, or if no directed path exists.

**`estimate_route_cost(graph, route, weight="current_travel_time") → float`**
Pure sum of a named edge attribute along consecutive nodes of a planned route. Default `weight="current_travel_time"` scores the path under live congested travel times already stored on each directed edge. Does not mutate the graph and does not call BPR updaters — callers refresh travel times before asking for a live cost.

- Optional `weight` (e.g. `"free_flow_time"`) lets the same helper sum other attributes in tests and experiments.
- Raises `ValueError` for an empty route, a single-node route, or a missing directed edge between consecutive nodes.
- A missing weight attribute on an existing edge surfaces as the raw `KeyError`.

### Selfish real-time routing

Navigation-app users choose the route with the lowest *currently estimated personal travel time*. This is decentralised routing: every driver tries to minimise their own trip time.

Per-vehicle flag `uses_navigation_app` (default `False`) controls entry auto-routing when no route is pre-assigned:
- `False` → Dijkstra on `free_flow_time` (uninformed / static).
- `True` → Dijkstra on live `current_travel_time` (selfish).

The route chosen at entry stays fixed for the whole journey (no mid-trip re-routing in this milestone). A pre-assigned route always wins; the flag is unused for that vehicle’s entry. Same-step due vehicles are still routed then entered in list order, so later selfish entrants can see earlier occupancy and may choose different paths — simultaneous entry is not order-independent.

### Navigation-app adoption rates

`src/adoption.py` assigns who uses the app on an existing vehicle list for a given adoption fraction. Demand builders stay at default uninformed fleets; call the helper after building demand when you need a mixed fleet.

```python
from src.adoption import ADOPTION_RATES, assign_navigation_adoption

ADOPTION_RATES  # (0.0, 0.25, 0.5, 0.75, 1.0) — planned experiment fractions
```

**`assign_navigation_adoption(vehicles, rate, *, seed) → list[Vehicle]`**
- `rate`: fraction in `[0.0, 1.0]` (`int` or `float`, including `0` / `1`); rejects `bool`; out of range → `ValueError`.
- `seed`: required keyword-only integer; rejects `bool` and non-integers → `TypeError`.
- Overwrites every vehicle’s `uses_navigation_app` flag. Exactly `k = round(n * rate)` vehicles are set to `True` (Python banker’s rounding — e.g. `n=5`, `rate=0.5` → `k=2`). Selection is a seeded shuffle of indices (exact count, not Bernoulli). Mutates the list in place and returns the same list object. An empty list is a no-op.

### Road disruption helpers

`src/disruption.py` provides two physical-road disruption helpers. Callers apply them before a run or between steps; there is no simulation-scheduled disruption API yet. Restore/`original_capacity` and exported factor presets are deferred.

```python
from src.disruption import reduce_road_capacity, close_road

reduce_road_capacity(graph, u, v, *, factor)
close_road(graph, u, v)
```

**`reduce_road_capacity(graph, u, v, *, factor) → DiGraph`**
- `factor`: remaining-capacity fraction in `(0.0, 1.0]` (`int` or `float`, including integer `1`); rejects `bool`; out of range or `<= 0` → `ValueError`.
- Requires both directed edges `(u, v)` and `(v, u)` with numeric `capacity > 0` on each; missing edge → `ValueError`; non-numeric capacity → `TypeError`; non-positive capacity → `ValueError`.
- Multiplies each direction’s current capacity by `factor` (float product, no rounding); leaves `occupancy` unchanged; refreshes `current_travel_time` on both edges via `update_road_travel_time`. Mutates the graph in place and returns the same object. Re-applying compounds on current capacity.

**`close_road(graph, u, v) → DiGraph`**
- Requires both directed edges `(u, v)` and `(v, u)`; missing either → `ValueError` before any removal.
- Removes both directed edges (attributes discarded); leaves nodes in place. Does not refresh travel times on remaining edges, check connectivity, or inspect occupancy.
- Mutates the graph in place and returns the same object. Closing an already-absent pair raises (not idempotent). Mid-run use while vehicles may still reference removed edges is unsupported.

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
Processes the current clock value then increments by one. Phase order: (1) enter vehicles due at `current_step` (auto-route if needed — `free_flow_time` for uninformed vehicles, `current_travel_time` for `uses_navigation_app=True`; place onto first edge); (2) advance in-transit vehicles (consume dwell; leave/enter edges or complete); (3) refresh all edge travel times via BPR. Mutates state in place; returns `None`.

**`run(until: int) → list[Vehicle]`**
Executes up to `until` steps. Stops early when nothing is active and no remaining vehicle has `start_time >= current_step`. Returns the list of completed vehicles. Raises `ValueError` if `until < 1`.

Edge dwell uses ``max(1, ceil(current_travel_time))`` ticks taken from the
edge's travel time **before** this vehicle is counted in occupancy, so the
first entrant on an empty road sees free-flow while later simultaneous
entrants see congestion. Occupancy is then incremented and the edge travel
time refreshed. The first dwell tick is consumed in the same step a vehicle
enters an edge from the release phase. Intermediate node transfers are
instantaneous (leave previous edge and enter the next in the same advance)
without cascading extra dwell ticks, so a free-flow edge of travel time ``1``
takes exactly one simulation step. In-transit timers are stored inside the
simulation, not on the vehicle agent.

## Metrics

Journey and network metrics read vehicle and graph state without mutating the simulation. Journey times use discrete simulation steps (`completion_time - start_time`). Road congestion for visualisation is the directed-edge occupancy/capacity ratio, capped at `1.0`. Recovery scoring uses a separate uncapped capacity-weighted network congestion score and pure helpers on a dense `C(t)` series; it does not schedule disruption/restore or claim integrated experiment validity.

### `src/metrics.py` — public API

```python
from src.metrics import (
    journey_time,
    completed_count,
    mean_journey_time,
    road_congestion,
    simulation_summary,
    network_congestion_score,
    pre_disruption_congestion,
    congestion_recovery,
)
```

**`journey_time(vehicle) → int | None`**
Returns `completion_time - start_time` in simulation steps. Incomplete vehicles (`completion_time is None`) return `None`. Raises `TypeError` / `ValueError` for malformed start or completion times.

**`completed_count(vehicles) → int`**
Counts vehicles with a non-`None` `completion_time`.

**`mean_journey_time(vehicles) → float | None`**
Mean of completed journey times only. Incomplete vehicles are excluded; returns `None` when none are complete.

**`road_congestion(graph) → dict[tuple, float]`**
Maps each directed edge `(u, v)` to `min(occupancy / capacity, 1.0)`. Raises `TypeError` / `ValueError` for invalid capacity or occupancy.

**`simulation_summary(simulation) → dict`**
Plain dictionary with keys `completed_count`, `mean_journey_time`, and `road_congestion`. Reads `simulation.vehicles` and `simulation.graph` without changing state.

**`network_congestion_score(graph) → float`**
Uncapped capacity-weighted mean `sum(occupancy) / sum(capacity)` over open edges (closed/absent edges excluded). Returns `math.nan` when there are no open edges.

**`pre_disruption_congestion(C, t_star, W) → float`**
Mean of dense series `C` over the pre-disruption window `[t_star - W, t_star)`.

**`congestion_recovery(C, t_star, t_r, W, K, horizon) → dict`**
Threshold recovery on a dense `C` series after a disruption/restore pair. Computes `C_pre` via `pre_disruption_congestion`, sets `τ = 1.05 C_pre`, requires an excursion in `[t_star, t_r)`, and counts only post-restore `K`-streaks that confirm on or before the inclusive `horizon`. Callers supply `W` and `K` (recommended defaults `W = 4 T_corr`, `K = 2 T_corr` for a configuration-derived free-flow OD time `T_corr`). Invalid `t_r` or non-positive `C_pre` return `valid=False` with null derived fields; programming errors raise.

## Demand generation

Baseline demand is a simultaneous corridor batch. Intensity is a single batch size `n` against the top-corridor capacity of 10. Named presets are **low = 1**, **medium = 5**, **high = 15**. Vehicles are returned bare (`route is None`, `start_time=0`); `Simulation` auto-routes on entry with free-flow weights.

### `src/demand.py` — public API

```python
from src.demand import (
    DEMAND_LOW, DEMAND_MEDIUM, DEMAND_HIGH,  # 1, 5, 15
    DEMAND_LEVELS,
    build_corridor_demand,
    build_low_demand,
    build_medium_demand,
    build_high_demand,
)
```

**`build_corridor_demand(graph, n, *, origin=0, destination=2) → list[Vehicle]`**
Builds `n` bare vehicles for the corridor. Raises `TypeError` if `n` is not an integer; `ValueError` if `n < 1` or the OD nodes are missing.

**`build_low_demand` / `build_medium_demand` / `build_high_demand`**
Thin wrappers around the preset batch sizes.

### Validate the baseline (low / medium / high)

From the repository root (after installing requirements):

```bash
python scripts/validate_baseline_demand.py
```

This runs three static/uninformed scenarios (0% navigation-app adoption, no disruption) with horizon `until=100` and writes `results/baseline_demand_validation.csv` with columns `demand_level,n,completed_count,mean_journey_time`. Automated tests assert every vehicle completes, low-demand mean journey time equals free-flow **1**, and mean journey time rises strictly low < medium < high.

### Regenerate the demand × adoption table

From the repository root (after installing requirements):

```bash
python scripts/run_demand_adoption.py
```

This calls `run_demand_adoption_grid` (seed `0`, horizon `until=100`, both locked inside the runner; the script takes no arguments). It crosses low / medium / high demand with adoption fractions `0`, `0.25`, `0.5`, `0.75`, and `1` (15 cells). Each cell uses a fresh default network, a bare default-corridor batch, then `assign_navigation_adoption`. The script creates `results/` if needed and overwrites `results/demand_adoption.csv` with columns `demand_level,n,adoption_rate,nav_count,seed,completed_count,mean_journey_time`. Journey time is in discrete simulation steps. There is no disruption axis and no travel-time-versus-adoption figure.

Automated tests call the runner (not the script). They lock row count and order, `nav_count = round(n * rate)` (Python banker's rounding), seed `0` repeatability, completion of every vehicle, agreement of the 0% column with an uninformed baseline of the same batch size and horizon, and low-demand mean journey time **1** at every adoption rate (including the one-vehicle navigator rows). Medium- and high-demand means are observed from the run and are not hard-coded. Tests do not claim that mean journey time falls and then rises with adoption.

## Network congestion visualisation

Directed edges can be coloured by the same occupancy/capacity congestion ratios used in metrics. Colour is mapped on a fixed `[0.0, 1.0]` scale (`YlOrRd`) with a colourbar; opposing directions are drawn as offset strokes. The helper is read-only and does not mutate edge attributes.

### `src/visualisation.py` — public API

```python
from src.visualisation import plot_network_congestion

plot_network_congestion(graph, path, pos=None)
```

**`plot_network_congestion(graph, path, pos=None) → None`**
Saves a PNG of the directed road graph coloured by congestion. Default node layout is the documented 2×3 grid for nodes `0`–`5`; pass `pos` for custom layouts. Uses the Agg matplotlib backend so figures can be written headlessly.

### Generate the Week 9 demo figure

From the repository root (after installing requirements):

```bash
python scripts/plot_network_congestion.py
```

This runs a short deterministic simulation that stacks seed vehicles on the top corridor so at least one edge is near capacity, then writes `results/network_congestion.png`.

## Experimental design

The following factors will be varied systematically:

| Factor | Scenarios |
| --- | --- |
| Traffic demand | Low, medium, high |
| Navigation-app adoption | 0%, 25%, 50%, 75%, 100% |
| Disruption | None, reduced road capacity, road closure |
| Routing policy | Uninformed, selfish real-time, coordinated (extension) |

The demand × adoption grid (no disruption) is recorded by `scripts/run_demand_adoption.py` at a single locked seed (`0`). A later issue will repeat stochastic scenarios with multiple seeds and compare distributions. Disruption sweeps and the rise-and-fall travel-time claim stay out of this table while routes remain locked at entry.

## Measures

The model currently records (via `src/metrics.py`):

- completed journey count;
- mean journey time (simulation steps; incomplete vehicles excluded);
- directed-road congestion ratios (occupancy/capacity, capped at `1.0`); and
- pure congestion-recovery helpers (`network_congestion_score`, `pre_disruption_congestion`, `congestion_recovery`) on synthetic or caller-built `C(t)` series.

**Recovery status:** pure helpers are shipped and unit-tested. Integrated recovery experiment claims remain blocked until warm-up, continuous/multi-wave demand, and scheduled disrupt/restore exist. Batch-only demand runs should keep using journey metrics and must not report headline recovery time. Permanent disruptions without restore are undefined for this metric. Later restore semantics (for capacity cuts: restore stored `original_capacity`; for closures: reinsert edges with stored attrs and zero occupancy; instantaneous at `t_r`) are a dependency for integrated experiments, not implemented here.

Still planned for later issues:

- median and 95th-percentile journey time;
- total network delay;
- road-level congestion over time;
- journey-based recovery (secondary); and
- the difference between individual and network-wide routing outcomes.

## Expected outputs

- Network diagrams coloured by congestion level (see `scripts/plot_network_congestion.py` → `results/network_congestion.png`).
- Time-series plots of congestion after a disruption.
- Average travel time versus navigation-app adoption rate.
- Comparisons of selfish and coordinated routing across demand levels.

## Project status

**Current stage:** Synthetic network topology, BPR congestion travel-time calculation, vehicle agents, static (uninformed) shortest-path routing, real-time route-cost estimation (`estimate_route_cost`), selfish entry auto-routing via per-vehicle `uses_navigation_app`, seeded navigation-app adoption-rate helpers (`assign_navigation_adoption`, `ADOPTION_RATES`), reduced-road-capacity disruption (`reduce_road_capacity`), full road-closure disruption (`close_road`), the discrete simulation clock with vehicle movement, journey/network metrics (including pure congestion-recovery helpers), network congestion visualisation, baseline demand generation (low / medium / high corridor batches), and the demand × adoption experiment runner (`run_demand_adoption_grid`, CSV `results/demand_adoption.csv`) are implemented. Mid-trip re-routing, coordinated routing, scheduled disruption / restore, continuous/multi-wave demand, wired recovery series collection, multi-seed repeats, and disruption sweeps inside the experiments runner are not implemented yet.

The first modelling milestone — rising demand produces rising travel times under static routing — is validated by `scripts/validate_baseline_demand.py` and `tests/test_demand.py`. The demand × adoption baseline (seed `0`, no disruption) is produced by `scripts/run_demand_adoption.py` and `tests/test_experiments.py`. The next milestones are multi-seed repeats of that grid, scheduled disruption/restore composed into the same runner, continuous or multi-wave demand, and wiring `C(t)` collection. The rise-and-fall adoption hypothesis stays untested while routes are locked at entry.

## Repository layout

```text
src/            Simulation source code
scripts/        Deterministic demo / figure generation scripts
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

At this stage, `pytest` runs the project smoke check, network topology tests, congestion calculation tests, vehicle agent tests (including `uses_navigation_app`), static routing and route-cost estimation tests, simulation clock / vehicle movement / selfish entry-routing tests, navigation-app adoption assignment tests, reduced-road-capacity and full road-closure disruption tests, journey/network metrics tests (including pure congestion-recovery helpers on synthetic series), visualisation smoke tests, baseline demand validation tests, and the demand × adoption grid tests. Additional model behaviour tests will be added with later issues.

## Reproducibility

The final repository will include installation instructions, a fixed random-seed option, experiment configuration files, and scripts to regenerate all reported results and figures.