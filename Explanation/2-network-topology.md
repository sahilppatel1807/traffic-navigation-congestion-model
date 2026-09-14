# Issue 2: Synthetic road-network topology

## Issue

- Number: #2
- Title: Create synthetic road-network topology
- Link: https://github.com/sahilppatel1807/traffic-navigation-congestion-model/issues/2
- Request: Add a reproducible directed synthetic road network with edge attributes, tests, and documentation.

## Summary

Implemented the default six-node road topology as a deterministic `networkx.DiGraph`. Each physical two-way road is modelled as a pair of opposite directed edges, with every edge carrying `free_flow_time`, `capacity`, `occupancy`, and `current_travel_time`. Added a compact summary helper for debugging and documented the topology and units in the README.

## Files changed

### New

- `src/network.py`: Defines the default road-network constants, graph factory, road helper, and summary helper.
- `tests/test_network.py`: Covers graph type, node and edge counts, required edge attributes, two-way directionality, reproducibility, and summary output.

### Modified

- `README.md`: Documents the default topology, edge-attribute units, exported network helpers, and updated project status.

### Deleted

- None.

## How it connects

- Exports `create_default_network()`, which later simulation, routing, and experiment modules can use as the baseline topology.
- Exports `summarize_network()`, a debugging helper for checking network size and per-edge attributes without plotting.
- Uses `networkx`, already listed in `requirements.txt`.
- Introduces no congestion, vehicle, routing, disruption, or simulation-clock behaviour.

## Tests

- Added `tests/test_network.py`.
- Ran `pytest` in the local `.venv`.
- Result: 6 tests passed.

## Known gaps / follow-ups

- Congestion travel-time updates are deferred to a later issue.
- Vehicle agents, routing policies, road disruptions, and simulation steps are still out of scope.
- `gh issue view 2` could not be used for live issue details because GitHub API access returned `Forbidden`; this document uses the issue title and link from the implementation plan.
