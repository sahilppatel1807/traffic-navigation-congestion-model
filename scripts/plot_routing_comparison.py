#!/usr/bin/env python3
"""Plot labelled policy comparisons from routing-comparison summary output."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt

SUMMARY_INPUT = _REPO_ROOT / "results" / "routing_comparison_summary.csv"
FIGURE_OUTPUT = _REPO_ROOT / "results" / "routing_comparison_journey_times.png"


def main() -> None:
    with SUMMARY_INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {SUMMARY_INPUT}")
    policies = ("static", "selfish", "coordinated")
    labels = []
    means = []
    for policy in policies:
        values = [
            float(row["mean_journey_time"])
            for row in rows
            if row["policy"] == policy and row["mean_journey_time"]
        ]
        labels.append(policy)
        means.append(sum(values) / len(values) if values else float("nan"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means)
    ax.set_xlabel("Routing policy (mean across demand, adoption, disruption, and seeds)")
    ax.set_ylabel("Mean journey time (simulation steps)")
    ax.set_title("Static, selfish, and coordinated routing comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=150)
    print(f"Wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
