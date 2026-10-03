"""Interactive presentation demo of the traffic model.

Launch from the repository root:

    streamlit run app/demo.py

The app is a shell. Demand, adoption, disruption, routing, the simulation
clock, congestion colours, and journey metrics all come from the existing
model. Dots are placed from :meth:`src.simulation.Simulation.in_transit_snapshot`.
Buttons are the only actions that build or advance a scene. Moving a control
leaves the map and the numbers on the last button press.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from src.adoption import assign_navigation_adoption
from src.demand import build_corridor_demand
from src.disruption import close_road, reduce_road_capacity
from src.experiments import EXPERIMENT_DEMAND_LEVELS, EXPERIMENT_PRIMARY_CAPACITY
from src.metrics import completed_count, mean_journey_time, network_congestion_score, road_congestion
from src.network import create_default_network
from src.simulation import Simulation
from src.visualisation import DEFAULT_NODE_POSITIONS

# Opening scene and the Presentation Demo button: high demand, synchronized
# navigation, full adoption, seed 0, and no disruption.
PRESENTATION_DEMAND = "High"
PRESENTATION_POLICY = "Shared navigation"
PRESENTATION_ADOPTION = 1.0
PRESENTATION_DISRUPTION = "None"
PRESENTATION_SEED = 0

DEMAND_OPTIONS = ("Low", "Medium", "High")
DEMAND_COUNTS = {
    level.title(): count for level, count in EXPERIMENT_DEMAND_LEVELS.items()
}
POLICY_OPTIONS = ("Uninformed", "Selfish", "Shared navigation", "Coordinated")
ADOPTION_OPTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DISRUPTION_OPTIONS = ("None", "Halve road 1–2", "Close road 1–2")

FRAME_INTERVAL_SECONDS = 0.5
# Stops a stuck scene from holding the talk. The presentation run is ~10 frames.
_PLAYBACK_STEP_CAP = 1000

_EDGE_ARC_RAD = 0.12
_CONGESTION_CMAP = "YlOrRd"
_NAVIGATION_COLOR = "#1f77b4"
_UNINFORMED_COLOR = "#333333"
_COORDINATED_COLOR = "#2ca02c"
# Pulls dots off the node discs and onto the visible road stroke.
_EDGE_DRAW_START = 0.12
_EDGE_DRAW_END = 0.88
_SAME_EDGE_OFFSET = 0.045

_DEMAND_BUILDERS = {
    level: (lambda graph, count=count: build_corridor_demand(graph, count))
    for level, count in DEMAND_COUNTS.items()
}


def _build_scenario(
    *,
    demand: str,
    policy: str,
    adoption: float,
    disruption: str,
    seed: int,
) -> Simulation:
    """Build one scene from the controls and return it at the opening frame."""
    graph = create_default_network()
    for edge in ((0, 1), (1, 2)):
        graph[edge[0]][edge[1]]["capacity"] = EXPERIMENT_PRIMARY_CAPACITY
        graph[edge[1]][edge[0]]["capacity"] = EXPERIMENT_PRIMARY_CAPACITY
    vehicles = _DEMAND_BUILDERS[demand](graph)
    if policy == "Coordinated":
        rate = 0.0
    else:
        rate = 0.0 if policy == "Uninformed" else float(adoption)
    assign_navigation_adoption(vehicles, rate, seed=int(seed))
    if disruption == "Halve road 1–2":
        reduce_road_capacity(graph, 1, 2, factor=0.5)
    elif disruption == "Close road 1–2":
        # Keep the edge in the graph: existing vehicles may finish, while the
        # routing layer skips edges marked closed for new route selection.
        close_road(graph, 1, 2, remove_edges=False)
    routing_policy = {
        "Coordinated": "coordinated",
        "Shared navigation": "shared_navigation",
    }.get(policy, "decentralized")
    return Simulation(graph, vehicles, routing_policy=routing_policy)


def _scenario_from_controls() -> Simulation:
    return _build_scenario(
        demand=st.session_state.demand,
        policy=st.session_state.policy,
        adoption=st.session_state.adoption,
        disruption=st.session_state.disruption,
        seed=int(st.session_state.seed),
    )


def _has_more_to_do(simulation: Simulation) -> bool:
    """True while a step would still release or move a vehicle."""
    if simulation.active:
        return True
    for vehicle in simulation.vehicles:
        if vehicle in simulation.completed:
            continue
        if vehicle.start_time >= simulation.current_step:
            return True
    return False


def _on_policy_change() -> None:
    """Only policies with navigation users expose the adoption mix."""
    if st.session_state.policy in ("Uninformed", "Coordinated"):
        st.session_state.adoption = 0.0


def _on_run() -> None:
    st.session_state.simulation = _scenario_from_controls()
    st.session_state.playing = True


def _on_step() -> None:
    st.session_state.playing = False
    simulation = st.session_state.get("simulation")
    if simulation is None:
        st.session_state.simulation = _scenario_from_controls()
        return
    if _has_more_to_do(simulation):
        simulation.step()


def _on_reset() -> None:
    st.session_state.playing = False
    st.session_state.simulation = _scenario_from_controls()


def _on_presentation() -> None:
    st.session_state.demand = PRESENTATION_DEMAND
    st.session_state.policy = PRESENTATION_POLICY
    st.session_state.adoption = PRESENTATION_ADOPTION
    st.session_state.disruption = PRESENTATION_DISRUPTION
    st.session_state.seed = PRESENTATION_SEED
    st.session_state.simulation = _build_scenario(
        demand=PRESENTATION_DEMAND,
        policy=PRESENTATION_POLICY,
        adoption=PRESENTATION_ADOPTION,
        disruption=PRESENTATION_DISRUPTION,
        seed=PRESENTATION_SEED,
    )
    st.session_state.playing = True


def _boot() -> None:
    """Open on the presentation scene's first frame, before any playback."""
    if st.session_state.get("booted"):
        return
    st.session_state.demand = PRESENTATION_DEMAND
    st.session_state.policy = PRESENTATION_POLICY
    st.session_state.adoption = PRESENTATION_ADOPTION
    st.session_state.disruption = PRESENTATION_DISRUPTION
    st.session_state.seed = PRESENTATION_SEED
    st.session_state.simulation = _build_scenario(
        demand=PRESENTATION_DEMAND,
        policy=PRESENTATION_POLICY,
        adoption=PRESENTATION_ADOPTION,
        disruption=PRESENTATION_DISRUPTION,
        seed=PRESENTATION_SEED,
    )
    st.session_state.playing = False
    st.session_state.booted = True


def _point_on_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    fraction: float,
    rad: float,
) -> tuple[float, float]:
    """Point along the same straight line or quadratic arc used to draw the road."""
    x1, y1 = start
    x2, y2 = end
    if rad == 0.0:
        return (x1 + fraction * (x2 - x1), y1 + fraction * (y2 - y1))
    dx, dy = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0 + rad * dy
    cy = (y1 + y2) / 2.0 - rad * dx
    t = fraction
    one_minus = 1.0 - t
    return (
        one_minus**2 * x1 + 2.0 * one_minus * t * cx + t**2 * x2,
        one_minus**2 * y1 + 2.0 * one_minus * t * cy + t**2 * y2,
    )


def _draw_fractions(records) -> list[float]:
    """Edge progress for each snapshot record, with a drawing-only queue offset."""
    peers: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        peers[record.edge].append(index)

    fractions: list[float] = []
    for index, record in enumerate(records):
        progress = (record.assigned_dwell - record.steps_remaining) / record.assigned_dwell
        rank = peers[record.edge].index(index)
        count = len(peers[record.edge])
        spread = (rank - (count - 1) / 2.0) * _SAME_EDGE_OFFSET
        along = progress + spread
        along = min(max(along, 0.0), 1.0)
        fractions.append(_EDGE_DRAW_START + (_EDGE_DRAW_END - _EDGE_DRAW_START) * along)
    return fractions


def _draw_network(simulation: Simulation):
    """Live congestion figure. Does not write the static PNG or mutate the graph."""
    graph = simulation.graph
    congestion = road_congestion(graph)
    layout = {node: DEFAULT_NODE_POSITIONS[node] for node in graph.nodes()}
    by_id = {vehicle.vehicle_id: vehicle for vehicle in simulation.vehicles}

    fig, ax = plt.subplots(figsize=(10, 4.2))
    cmap = plt.get_cmap(_CONGESTION_CMAP)
    norm = Normalize(vmin=0.0, vmax=1.0)

    for u, v in graph.edges():
        rad = _EDGE_ARC_RAD if graph.has_edge(v, u) else 0.0
        closed = graph[u][v].get("closed", False)
        ax.add_patch(
            FancyArrowPatch(
                layout[u],
                layout[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=2.5,
                color="#9ca3af" if closed else cmap(norm(congestion.get((u, v), 0.0))),
                linestyle="--" if closed else "-",
                alpha=0.7 if closed else 1.0,
                shrinkA=14,
                shrinkB=14,
                zorder=1,
            )
        )

    records = simulation.in_transit_snapshot()
    fractions = _draw_fractions(records)
    for record, fraction in zip(records, fractions):
        u, v = record.edge
        rad = _EDGE_ARC_RAD if graph.has_edge(v, u) else 0.0
        x, y = _point_on_edge(layout[u], layout[v], fraction, rad)
        vehicle = by_id[record.vehicle_id]
        if simulation.routing_policy == "coordinated":
            color = _COORDINATED_COLOR
        elif vehicle.uses_navigation_app:
            color = _NAVIGATION_COLOR
        else:
            color = _UNINFORMED_COLOR
        ax.scatter(
            [x],
            [y],
            s=55,
            c=color,
            zorder=2,
            linewidths=0.4,
            edgecolors="white",
        )

    xs = [layout[node][0] for node in graph.nodes()]
    ys = [layout[node][1] for node in graph.nodes()]
    ax.scatter(xs, ys, s=420, c="white", edgecolors="black", zorder=3, linewidths=1.5)
    for node, (x, y) in layout.items():
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Congestion (occupancy / capacity)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _format_mean(mean: float | None) -> str:
    if mean is None:
        return ""
    text = f"{mean:.2f}".rstrip("0").rstrip(".")
    return text


def _render_metrics(slot, simulation: Simulation) -> None:
    score = network_congestion_score(simulation.graph)
    mean = _format_mean(mean_journey_time(simulation.vehicles))
    total_vehicles = len(simulation.vehicles)
    completed_vehicles = completed_count(simulation.vehicles)
    completion_rate = (
        100.0 * completed_vehicles / total_vehicles if total_vehicles else 0.0
    )
    primary_route = [0, 1, 2]
    assigned_routes = [
        vehicle.route for vehicle in simulation.vehicles if vehicle.route is not None
    ]
    primary_count = sum(route == primary_route for route in assigned_routes)
    primary_share = (
        100.0 * primary_count / len(assigned_routes) if assigned_routes else 0.0
    )
    disruption = st.session_state.get("disruption", "None")
    primary_capacity = simulation.graph[0][1]["capacity"]
    if disruption == "Close road 1–2":
        capacity_assumption = "Primary corridor: 2 vehicles before closure"
    else:
        capacity_assumption = (
            f"Primary corridor: {EXPERIMENT_PRIMARY_CAPACITY:g} vehicles "
            f"(active: {primary_capacity:g})"
        )
    rows = (
        ("Timestep", str(simulation.current_step)),
        ("Vehicles", str(total_vehicles)),
        ("In transit", str(len(simulation.active))),
        ("Completed vehicles", str(completed_vehicles)),
        ("Completion rate", f"{completion_rate:.0f}%"),
        ("Mean journey time (steps)", mean),
        ("Network congestion", f"{score:.3f}"),
        ("Primary route", f"{primary_count}/{total_vehicles} ({primary_share:.0f}%)"),
    )
    lines = []
    for label, value in rows:
        lines.append(
            "<p style='margin:0.05rem 0 0;font-size:0.8rem;opacity:0.85'>"
            f"{label}</p>"
            "<p style='margin:0 0 0.35rem;font-size:1.45rem;line-height:1.05'>"
            f"{value or '&nbsp;'}</p>"
        )
    slot.markdown(
        "".join(lines)
        + "<p style='margin:0.8rem 0 0;font-size:0.8rem;opacity:0.85'>"
        f"Active disruption: {disruption}</p>"
        + "<p style='margin:0.05rem 0 0;font-size:0.8rem;opacity:0.85'>"
        f"{capacity_assumption}</p>",
        unsafe_allow_html=True,
    )


def _render(map_slot, metric_slot, simulation: Simulation) -> None:
    map_slot.pyplot(_draw_network(simulation), clear_figure=True)
    _render_metrics(metric_slot, simulation)


def _play(map_slot, metric_slot, simulation: Simulation) -> None:
    """Block until the scene finishes, pausing between frames, then hold the last one."""
    steps = 0
    while _has_more_to_do(simulation) and steps < _PLAYBACK_STEP_CAP:
        time.sleep(FRAME_INTERVAL_SECONDS)
        simulation.step()
        steps += 1
        _render(map_slot, metric_slot, simulation)
    st.session_state.playing = False


def main() -> None:
    st.set_page_config(page_title="Traffic model demo", layout="wide")
    _boot()

    demand_col, policy_col, adoption_col, disruption_col, seed_col = st.columns(
        [1, 1, 1.3, 1.4, 0.7]
    )
    with demand_col:
        st.selectbox(
            "Demand",
            DEMAND_OPTIONS,
            format_func=lambda level: f"{level} ({DEMAND_COUNTS[level]} vehicles)",
            key="demand",
        )
    with policy_col:
        st.selectbox(
            "Routing",
            POLICY_OPTIONS,
            key="policy",
            on_change=_on_policy_change,
        )
    with adoption_col:
        st.select_slider(
            "Adoption",
            options=list(ADOPTION_OPTIONS),
            format_func=lambda rate: f"{int(rate * 100)}%",
            key="adoption",
            disabled=st.session_state.policy in ("Uninformed", "Coordinated"),
        )
    with disruption_col:
        st.selectbox("Disruption", DISRUPTION_OPTIONS, key="disruption")
    with seed_col:
        st.number_input("Seed", min_value=0, step=1, key="seed")

    run_col, step_col, reset_col, demo_col = st.columns(4)
    with run_col:
        st.button("Run", on_click=_on_run, width="stretch")
    with step_col:
        st.button("Step", on_click=_on_step, width="stretch")
    with reset_col:
        st.button("Reset", on_click=_on_reset, width="stretch")
    with demo_col:
        st.button(
            "Presentation Demo",
            on_click=_on_presentation,
            width="stretch",
        )

    map_col, metric_col = st.columns([4, 1])
    with map_col:
        map_slot = st.empty()
        st.caption(
            "Green: coordinated. Blue: navigation users. Dark grey: uninformed. "
            "Shared recommendations can concentrate vehicles on one route. "
            "A closed road is unavailable to new routes; vehicles already on it finish."
        )
    with metric_col:
        metric_slot = st.empty()

    simulation = st.session_state.simulation
    _render(map_slot, metric_slot, simulation)
    if st.session_state.playing:
        _play(map_slot, metric_slot, simulation)


if __name__ == "__main__":
    main()
