#!/usr/bin/env python3
"""Baseline demand validation: low / medium / high → CSV under results/.

Runs three static/uninformed scenarios (0% navigation adoption, no disruption)
on the default network and writes
``results/baseline_demand_validation.csv``.

Run from the repository root::

    python scripts/validate_baseline_demand.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.demand import DEMAND_LEVELS, build_corridor_demand
from src.metrics import completed_count, mean_journey_time
from src.network import create_default_network
from src.simulation import Simulation

VALIDATION_HORIZON = 100
OUTPUT_CSV = _REPO_ROOT / "results" / "baseline_demand_validation.csv"
CSV_COLUMNS = ("demand_level", "n", "completed_count", "mean_journey_time")


def run_level(demand_level: str, n: int) -> dict[str, object]:
    """Run one baseline demand level and return a CSV row dict."""
    graph = create_default_network()
    vehicles = build_corridor_demand(graph, n)
    sim = Simulation(graph, vehicles)
    sim.run(until=VALIDATION_HORIZON)
    return {
        "demand_level": demand_level,
        "n": n,
        "completed_count": completed_count(sim.vehicles),
        "mean_journey_time": mean_journey_time(sim.vehicles),
    }


def main() -> None:
    rows = [
        run_level(level, n)
        for level, n in DEMAND_LEVELS.items()
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}")
    for row in rows:
        print(
            f"  {row['demand_level']}: n={row['n']}, "
            f"completed={row['completed_count']}, "
            f"mean_jt={row['mean_journey_time']}"
        )


if __name__ == "__main__":
    main()
