"""Network congestion visualisation for directed road graphs.

Public API
----------
plot_network_congestion(graph, path, pos=None) -> None
    Colour directed edges by occupancy/capacity congestion and save a figure.
    Does not mutate graph edge attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

# Non-interactive backend so saving figures works in headless / CI / pytest.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from src.metrics import road_congestion

# Fixed 2×3 grid matching the default synthetic network (0–1–2 over 3–4–5).
DEFAULT_NODE_POSITIONS: dict[int, tuple[float, float]] = {
    0: (0.0, 1.0),
    1: (1.0, 1.0),
    2: (2.0, 1.0),
    3: (0.0, 0.0),
    4: (1.0, 0.0),
    5: (2.0, 0.0),
}

_DEFAULT_COLORMAP = "YlOrRd"
# Arc radius for FancyArrowPatch when both directions exist; same sign on each
# stroke separates A→B from B→A onto opposite sides of the chord.
_EDGE_ARC_RAD = 0.12


def plot_network_congestion(
    graph: nx.DiGraph,
    path: str | Path,
    *,
    pos: Mapping[Any, tuple[float, float]] | None = None,
) -> None:
    """Colour directed edges by congestion and save the figure to ``path``.

    Congestion is each edge's occupancy/capacity ratio capped at ``1.0`` (same
    definition as :func:`src.metrics.road_congestion`). Colour is mapped on a
    fixed ``[0.0, 1.0]`` scale with colormap ``YlOrRd``. Opposing directed
    edges are drawn as offset curved strokes so each direction keeps its own
    colour. Missing or empty occupancy is treated as free (``0.0``) via the
    metrics helper when attributes are present at zero.

    Parameters
    ----------
    graph:
        Directed road network with ``capacity`` and ``occupancy`` on each edge.
    path:
        Destination image path (parent directories are created if needed).
    pos:
        Optional node → ``(x, y)`` layout. When omitted, the default 2×3 grid
        is used if every node is in that layout; otherwise a deterministic
        spring layout is used.

    Notes
    -----
    This function is read-only with respect to edge attributes: it does not
    write congestion values or other fields back onto ``graph``.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    congestion = road_congestion(graph)
    layout = _resolve_positions(graph, pos)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap(_DEFAULT_COLORMAP)
    norm = Normalize(vmin=0.0, vmax=1.0)

    for u, v in graph.edges():
        ratio = congestion.get((u, v), 0.0)
        rad = _EDGE_ARC_RAD if graph.has_edge(v, u) else 0.0
        _draw_directed_edge(
            ax,
            layout[u],
            layout[v],
            color=cmap(norm(ratio)),
            rad=rad,
        )

    xs = [layout[n][0] for n in graph.nodes()]
    ys = [layout[n][1] for n in graph.nodes()]
    ax.scatter(xs, ys, s=400, c="white", edgecolors="black", zorder=3, linewidths=1.5)

    for node, (x, y) in layout.items():
        if node in graph:
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

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Congestion (occupancy / capacity)")

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_positions(
    graph: nx.DiGraph,
    pos: Mapping[Any, tuple[float, float]] | None,
) -> dict[Any, tuple[float, float]]:
    if pos is not None:
        missing = [n for n in graph.nodes() if n not in pos]
        if missing:
            raise ValueError(
                f"pos is missing coordinates for nodes: {missing!r}"
            )
        return {n: tuple(pos[n]) for n in graph.nodes()}

    if all(n in DEFAULT_NODE_POSITIONS for n in graph.nodes()):
        return {n: DEFAULT_NODE_POSITIONS[n] for n in graph.nodes()}

    return nx.spring_layout(graph, seed=0)


def _draw_directed_edge(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: Any,
    rad: float,
) -> None:
    """Draw one directed edge as a curved (or straight) arrow stroke."""
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=2.5,
        color=color,
        shrinkA=12,
        shrinkB=12,
        zorder=1,
    )
    ax.add_patch(arrow)
