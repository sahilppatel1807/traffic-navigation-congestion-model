#!/usr/bin/env python3
"""Demand × adoption grid: write journey metrics to results/demand_adoption.csv.

Runs the locked fifteen-cell grid (seed and horizon live in the experiments
runner) and overwrites ``results/demand_adoption.csv``.

Run from the repository root::

    python scripts/run_demand_adoption.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import run_demand_adoption_grid

OUTPUT_CSV = _REPO_ROOT / "results" / "demand_adoption.csv"
CSV_COLUMNS = (
    "demand_level",
    "n",
    "adoption_rate",
    "nav_count",
    "seed",
    "completed_count",
    "mean_journey_time",
)


def main() -> None:
    rows = run_demand_adoption_grid()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}")
    for row in rows:
        print(
            f"  {row['demand_level']}: n={row['n']}, "
            f"adoption={row['adoption_rate']}, "
            f"nav={row['nav_count']}, "
            f"seed={row['seed']}, "
            f"completed={row['completed_count']}, "
            f"mean_jt={row['mean_journey_time']}"
        )


if __name__ == "__main__":
    main()
