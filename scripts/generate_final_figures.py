#!/usr/bin/env python3
"""Generate the final analysis figures from the routing-comparison CSVs.

The figures focus on the project's research comparison: decentralised selfish
routing at different navigation-adoption rates versus coordinated routing.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np

SUMMARY = _ROOT / "results" / "routing_comparison_summary.csv"
CONGESTION = _ROOT / "results" / "routing_comparison_congestion.csv"
OUT = _ROOT / "results" / "figures"
POLICIES = ("selfish", "coordinated")
COLORS = {
    "selfish": "#ff7f00",
    "coordinated": "#4daf4a",
}
DEMAND_ORDER = ("low", "medium", "high")
ADOPTION_ORDER = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row[field] not in ("", "None")]
    return sum(values) / len(values) if values else float("nan")


def _subset(rows, **filters):
    return [row for row in rows if all(row[key] == str(value) for key, value in filters.items())]


def _mean_by_seed(rows: list[dict[str, str]], field: str) -> float:
    """Average a metric across recorded seeds for one scenario cell."""
    return _mean(rows, field)


def figure1_adoption(summary: list[dict[str, str]]) -> None:
    """Show the non-monotone selfish-routing effect across demand levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, demand in zip(axes, DEMAND_ORDER):
        for policy in POLICIES:
            means = []
            for adoption in ADOPTION_ORDER:
                rows = _subset(summary, demand_level=demand, disruption="none", policy=policy, adoption_rate=adoption)
                means.append(_mean(rows, "mean_journey_time"))
            ax.plot([100 * a for a in ADOPTION_ORDER], means, marker="o", linewidth=2, label=policy, color=COLORS[policy])
        ax.set_title(f"{demand.title()} demand")
        ax.set_xlabel("Navigation adoption (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean completed journey time (simulation steps)")
    axes[-1].legend(title="Routing policy", loc="upper right")
    fig.suptitle("Selfish routing becomes less effective when adoption is too high", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_journey_time_vs_adoption.png", dpi=220)
    plt.close(fig)


def figure2_journey_time_uncertainty(summary: list[dict[str, str]]) -> None:
    """Show mean journey time with seed-to-seed variability."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, disruption in zip(axes, ("none", "capacity_reduction", "closure")):
        for policy in POLICIES:
            means = []
            errors = []
            for adoption in ADOPTION_ORDER:
                rows = _subset(summary, demand_level="high", disruption=disruption, policy=policy, adoption_rate=adoption)
                values = [float(row["mean_journey_time"]) for row in rows]
                mean = sum(values) / len(values) if values else float("nan")
                variance = sum((value - mean) ** 2 for value in values) / len(values) if values else float("nan")
                means.append(mean)
                errors.append(variance ** 0.5)
            ax.errorbar(
                [100 * a for a in ADOPTION_ORDER],
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=policy,
                color=COLORS[policy],
            )
        ax.set_title(disruption.replace("_", " ").title())
        ax.set_xlabel("Navigation adoption (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean completed journey time (simulation steps)")
    axes[-1].legend(title="Routing policy", loc="upper right")
    fig.suptitle("High-demand journey time with seed variability", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_journey_time_uncertainty.png", dpi=220)
    plt.close(fig)


def figure3_disruption(summary: list[dict[str, str]]) -> None:
    """Compare network-level delay as adoption rises under disruptions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, disruption in zip(axes, ("none", "capacity_reduction", "closure")):
        for policy in POLICIES:
            values = [
                _mean_by_seed(
                    _subset(summary, demand_level="high", disruption=disruption, policy=policy, adoption_rate=adoption),
                    "total_network_delay",
                )
                for adoption in ADOPTION_ORDER
            ]
            ax.plot([100 * a for a in ADOPTION_ORDER], values, marker="o", linewidth=2, label=policy, color=COLORS[policy])
        ax.set_title(disruption.replace("_", " ").title())
        ax.set_xlabel("Navigation adoption (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Total network delay (simulation steps)")
    axes[-1].legend(title="Routing policy", loc="upper right")
    fig.suptitle("Coordinated routing reduces network-wide delay", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_congestion_after_disruption.png", dpi=220)
    plt.close(fig)


def figure4_alternative_corridor(congestion: list[dict[str, str]]) -> None:
    """Show whether selfish users overload the alternative corridor."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for policy in POLICIES:
        values = []
        for adoption in ADOPTION_ORDER:
            edge_rows = [
                row for row in congestion
                if row["demand_level"] == "high"
                and row["disruption"] == "none"
                and row["policy"] == policy
                and row["adoption_rate"] == str(adoption)
                and row["edge"] == "(0, 3)"
            ]
            by_seed: dict[str, list[float]] = defaultdict(list)
            for row in edge_rows:
                by_seed[row["seed"]].append(float(row["road_congestion"]))
            seed_peaks = [max(v) for v in by_seed.values() if v]
            values.append(sum(seed_peaks) / len(seed_peaks) if seed_peaks else float("nan"))
        ax.plot([100 * a for a in ADOPTION_ORDER], values, marker="o", linewidth=2, label=policy, color=COLORS[policy])
    ax.axhline(1.0, color="crimson", linestyle="--", label="Capacity reached")
    ax.set_xlabel("Navigation adoption (%)")
    ax.set_ylabel("Peak alternative-road congestion (occupancy / capacity)")
    ax.set_title("High adoption concentrates selfish traffic on the alternative route", fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_demand_vs_journey_time.png", dpi=220)
    plt.close(fig)


def figure5_advantage(summary: list[dict[str, str]]) -> None:
    """Compare selfish adoption levels with coordinated routing after closure."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(ADOPTION_ORDER))
    width = 0.36
    values = {}
    for policy in POLICIES:
        values[policy] = [
            _mean(
                _subset(summary, demand_level="high", disruption="closure", policy=policy, adoption_rate=adoption),
                "mean_journey_time",
            )
            for adoption in ADOPTION_ORDER
        ]
    bars_selfish = ax.bar(x - width / 2, values["selfish"], width, label="selfish routing", color=COLORS["selfish"])
    bars_coordinated = ax.bar(x + width / 2, values["coordinated"], width, label="coordinated routing", color=COLORS["coordinated"])
    for bars in (bars_selfish, bars_coordinated):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, [f"{int(100 * adoption)}%" for adoption in ADOPTION_ORDER])
    ax.set_xlabel("Selfish navigation adoption")
    ax.set_ylabel("Mean completed journey time (simulation steps)")
    ax.set_title("High-demand journey time after road closure", fontweight="bold")
    ax.legend(title="Routing policy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_navigation_advantage.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = _load(SUMMARY)
    congestion = _load(CONGESTION)
    figure1_adoption(summary)
    figure2_journey_time_uncertainty(summary)
    figure3_disruption(summary)
    figure4_alternative_corridor(congestion)
    figure5_advantage(summary)
    print(f"Generated five figures in {OUT}")


if __name__ == "__main__":
    main()
