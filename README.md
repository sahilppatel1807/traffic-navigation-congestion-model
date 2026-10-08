# Traffic Navigation and Congestion Model

An agent-based computational model that investigates whether real-time navigation information improves traffic conditions or, when widely adopted, creates new congestion bottlenecks.

This project is implemented in Python 3.x for the CITS4403 research project.

## Project overview

### Research question

**How do traffic demand, real-time navigation-app adoption, and routing policy affect system-wide congestion and travel times after a road disruption?**

### Hypothesis

Real-time navigation may improve journey times when adoption is low or moderate. When many drivers receive the same recommendation, they may choose the same alternative road and create a new bottleneck, especially after a disruption.

### What the project does

The model represents a small synthetic road network as a graph. Intersections are nodes, roads are directed edges, and cars are individual vehicle agents. Each road has a capacity and a congestion-dependent travel time. The simulation advances in discrete time steps and records journey and network-level outcomes.

The project compares uninformed shortest-path routing, selfish real-time navigation, shared navigation recommendations, and coordinated routing. Experiments vary traffic demand, navigation-app adoption, disruption type, routing policy, and random seed. The main outputs are journey time, completion rate, road congestion, route concentration, and comparisons between individual and system-wide outcomes.

This is a controlled synthetic model, not a prediction of real Perth traffic. Its contribution is to study how individual and shared route choices interact with capacity bottlenecks after a disruption.

## Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

The test suite checks the model rules, routing, disruptions, metrics, experiments, and visualisation helpers.

## Open the Streamlit app

After installing the dependencies, run:

```bash
streamlit run app/demo.py
```

The app opens an interactive presentation of the synthetic road network. Use the controls to compare uninformed, selfish navigation, shared navigation, and coordinated routing under different demand, adoption, disruption, and seed settings. Press **Run**, **Step**, or **Reset** to update the simulation.

## Project structure

```text
src/            Core simulation model and metrics
app/            Streamlit interactive demonstration
scripts/        Experiment and figure-generation commands
notebooks/      Final analysis notebook
results/        Generated CSV files and figures
tests/          Automated tests
report/         Report materials
requirements.txt Python dependencies
README.md       Setup, usage, and project overview
```

The project uses a synthetic network and generates experiment inputs programmatically, so separate `data/` and `utils/` directories are not needed.

## Validation and analysis

The automated tests cover the main model rules and boundary cases, including valid routes, congestion behaviour, closed-road routing, vehicle movement, seeded adoption, disruptions, metrics, and repeatability. The baseline demand experiment also checks that increasing demand produces increasing journey times under static routing.

The notebook in `notebooks/final_analysis.ipynb` is the communication layer for the project. It combines explanatory text, experiment outputs, tables, and figures. The reusable model remains in `src/`, while scripts in `scripts/` reproduce the reported CSV files and visualisations.

## Running experiments

These commands regenerate the main validation results and figures:

```bash
python scripts/validate_baseline_demand.py
python scripts/run_demand_adoption.py
python scripts/run_routing_comparison.py
python scripts/plot_routing_comparison.py
python scripts/run_presentation_scenario.py
python scripts/generate_final_figures.py
```

Generated CSV files and figures are written to `results/`. The experiment outputs retain their scenario parameters and seeds. The final analysis notebook is [notebooks/final_analysis.ipynb](notebooks/final_analysis.ipynb).

## Scope and limitations

This is a controlled synthetic model, not a prediction of real Perth traffic. Driver behaviour, road geometry, and disruptions are simplified. The project does not model real map data, lanes, traffic lights, collisions, parking, pedestrians, machine learning, or reinforcement learning. Routes are locked after entry; mid-trip rerouting is outside the current scope.

The project’s contribution is the controlled comparison of routing policies, navigation adoption, demand, and disruption effects in a capacity-constrained network. Conclusions should be interpreted as findings about this model and its assumptions.
