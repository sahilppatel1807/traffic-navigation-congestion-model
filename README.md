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

## Routing behaviours

### Uninformed routing

Drivers choose the shortest route using normal, uncongested road travel times. They do not receive live traffic updates.

### Selfish real-time routing

Navigation-app users choose the route with the lowest *currently estimated personal travel time*. This is decentralised routing: every driver tries to minimise their own trip time.

### Coordinated routing (extension)

A central controller assigns or recommends routes to minimise total travel time across all vehicles. It may assign a car to a slightly slower route if that prevents a major bottleneck and improves network-wide performance.

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

**Current stage:** model design and baseline implementation.

The first milestone is a working simulation in which vehicles travel through a capacity-constrained network and rising demand produces rising travel times. The next milestone is to add real-time route choice and a road-disruption scenario.

## Planned repository layout

```text
src/            Simulation source code
notebooks/      Experiment analysis and visualisations
results/        Generated data and figures
tests/          Automated tests
report/         Project report and supporting material
```

## Reproducibility

The final repository will include installation instructions, a fixed random-seed option, experiment configuration files, and scripts to regenerate all reported results and figures.