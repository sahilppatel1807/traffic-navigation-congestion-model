#!/usr/bin/env python3
"""Generate the focused 6+4 presentation experiment outputs.

Run from the repository root::

    python scripts/run_presentation_scenario.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import run_focused_presentation_experiment

SUMMARY_OUTPUT = _REPO_ROOT / "results" / "focused_presentation_summary.csv"
FIGURE_OUTPUT = _REPO_ROOT / "results" / "figures" / "focused_presentation_metrics.png"
SUMMARY_COLUMNS = (
    "scenario_id", "policy", "adoption_rate", "nav_count", "disruption",
    "disruption_factor", "seed", "horizon", "vehicle_count",
    "completed_count", "completion_rate", "mean_journey_time",
    "final_completion_timestep", "makespan", "maximum_occupancy",
    "route_diversity", "most_common_route", "most_common_route_count",
    "most_common_route_share", "primary_route_count", "primary_route_share",
    "peak_network_congestion",
)


def _write_csv(rows: list[dict]) -> None:
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialised = dict(row)
            serialised["most_common_route"] = str(row["most_common_route"])
            writer.writerow(serialised)


def _mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else float("nan")


def _write_figure(rows: list[dict]) -> None:
    policies = ("uninformed", "selfish", "shared_navigation", "coordinated")
    labels = {"none": "No disruption", "capacity_reduction": "Half capacity"}
    colours = {
        "uninformed": "#333333",
        "selfish": "#1f77b4",
        "shared_navigation": "#e6550d",
        "coordinated": "#2ca02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for disruption, ax in zip(("none", "capacity_reduction"), axes):
        for policy in policies:
            values = [
                _mean(
                    [
                        row
                        for row in rows
                        if row["policy"] == policy
                        and row["adoption_rate"] == adoption
                        and row["disruption"] == disruption
                    ],
                    "mean_journey_time",
                )
                for adoption in (0.0, 0.5, 1.0)
            ]
            ax.plot(
                (0, 50, 100), values, marker="o", label=policy.replace("_", " "),
                color=colours[policy],
            )
        ax.set_title(labels[disruption])
        ax.set_xlabel("Navigation adoption (%)")
        ax.set_ylabel("Mean completed journey time (steps)")
        ax.set_xticks((0, 50, 100))
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Focused 6 + 4 demand: synchronized recommendations can herd traffic")
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = run_focused_presentation_experiment()
    _write_csv(rows)
    _write_figure(rows)
    print(f"Wrote {SUMMARY_OUTPUT} ({len(rows)} rows)")
    print(f"Wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
