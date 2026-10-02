#!/usr/bin/env python3
"""Generate the final analysis figures from the routing-comparison CSVs.

The figures deliberately show both journey time and completion rate.  In the
most congested scenarios some static vehicles can remain unfinished by the
finite horizon; hiding that fact would make the comparison misleading.
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
POLICIES = ("static", "selfish", "coordinated")
COLORS = {"static": "#377eb8", "selfish": "#ff7f00", "coordinated": "#4daf4a"}
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


def figure1_adoption(summary: list[dict[str, str]]) -> None:
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
    fig.suptitle("Figure 1: Navigation adoption matters only when demand creates congestion", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_journey_time_vs_adoption.png", dpi=220)
    plt.close(fig)


def figure2_completion(summary: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, demand in zip(axes, DEMAND_ORDER):
        for policy in ("static", "selfish"):
            rates = []
            for adoption in ADOPTION_ORDER:
                rows = _subset(summary, demand_level=demand, disruption="none", policy=policy, adoption_rate=adoption)
                fractions = [int(row["completed_journeys"]) / (int(row["vehicles_per_wave"]) * 2) for row in rows]
                rates.append(100 * sum(fractions) / len(fractions) if fractions else float("nan"))
            ax.plot([100 * a for a in ADOPTION_ORDER], rates, marker="o", linewidth=2, label=policy, color=COLORS[policy])
        ax.set_title(f"{demand.title()} demand")
        ax.set_xlabel("Navigation adoption (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Completed vehicles (%)")
    axes[-1].legend(title="Routing policy", loc="lower right")
    fig.suptitle("Figure 2: Completion rate reveals queueing that mean time alone can hide", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_completion_rate.png", dpi=220)
    plt.close(fig)


def figure3_disruption(congestion: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, disruption in zip(axes, ("capacity_reduction", "closure")):
        for policy in ("static", "selfish"):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in congestion:
                if row["demand_level"] == "high" and row["adoption_rate"] == "1.0" and row["disruption"] == disruption and row["policy"] == policy:
                    grouped[int(row["timestep"])].append(float(row["network_congestion_score"]))
            steps = sorted(grouped)
            values = [sum(grouped[t]) / len(grouped[t]) for t in steps]
            ax.plot(steps, values, linewidth=2, label=policy, color=COLORS[policy])
        ax.axvline(15, color="crimson", linestyle="--", label="Disruption onset")
        ax.axvline(45, color="purple", linestyle=":", label="Restoration")
        ax.set_title(disruption.replace("_", " ").title())
        ax.set_xlabel("Timestep (simulation steps)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean network congestion score")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Figure 3: High-adoption navigation changes the congestion response to disruption", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_congestion_after_disruption.png", dpi=220)
    plt.close(fig)


def figure4_demand(summary: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(DEMAND_ORDER))
    width = 0.18
    series = (("static", 0.0), ("selfish", 0.0), ("selfish", 1.0), ("coordinated", 1.0))
    for i, (policy, adoption) in enumerate(series):
        values = [_mean(_subset(summary, demand_level=d, disruption="none", policy=policy, adoption_rate=adoption), "mean_journey_time") for d in DEMAND_ORDER]
        ax.bar(x + (i - 1.5) * width, values, width, label=f"{policy}, adoption {int(100 * adoption)}%", color=COLORS[policy], alpha=0.55 + i * 0.12)
    ax.set_xticks(x, [f"{d.title()}\n({n} veh/wave)" for d, n in (("low", 1), ("medium", 5), ("high", 10))])
    ax.set_ylabel("Mean completed journey time (simulation steps)")
    ax.set_title("Figure 4: Demand amplifies the benefit of route information", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_demand_vs_journey_time.png", dpi=220)
    plt.close(fig)


def figure5_advantage(summary: list[dict[str, str]]) -> None:
    data = np.zeros((3, 5))
    for i, demand in enumerate(DEMAND_ORDER):
        for j, adoption in enumerate(ADOPTION_ORDER):
            static = _mean(_subset(summary, demand_level=demand, disruption="none", policy="static", adoption_rate=adoption), "mean_journey_time")
            selfish = _mean(_subset(summary, demand_level=demand, disruption="none", policy="selfish", adoption_rate=adoption), "mean_journey_time")
            data[i, j] = static - selfish
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(data, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(5), ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(range(3), ["Low", "Medium", "High"])
    ax.set_xlabel("Navigation adoption")
    ax.set_ylabel("Demand")
    ax.set_title("Figure 5: Selfish navigation advantage over static routing\n(positive values = lower journey time)", fontweight="bold")
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Journey-time reduction (simulation steps)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_navigation_advantage.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = _load(SUMMARY)
    congestion = _load(CONGESTION)
    figure1_adoption(summary)
    figure2_completion(summary)
    figure3_disruption(congestion)
    figure4_demand(summary)
    figure5_advantage(summary)
    print(f"Generated five figures in {OUT}")


if __name__ == "__main__":
    main()
