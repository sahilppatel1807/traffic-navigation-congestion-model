#!/usr/bin/env python3
"""Generate the reproducible routing-comparison CSV outputs.

Run from the repository root::

    python scripts/run_routing_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import run_routing_comparison

SUMMARY_OUTPUT = _REPO_ROOT / "results" / "routing_comparison_summary.csv"
CONGESTION_OUTPUT = _REPO_ROOT / "results" / "routing_comparison_congestion.csv"

SUMMARY_COLUMNS = (
    "scenario_id", "policy", "demand_level", "vehicles_per_wave",
    "adoption_rate", "nav_count", "disruption", "disruption_factor",
    "disrupted_road", "seed", "horizon", "completed_journeys",
    "mean_journey_time", "median_journey_time", "p95_journey_time",
    "total_network_delay", "peak_network_congestion", "recovery_time",
    "recovered", "recovery_censored", "recovery_status",
)
CONGESTION_COLUMNS = (
    "scenario_id", "policy", "demand_level", "vehicles_per_wave",
    "adoption_rate", "nav_count", "disruption", "disruption_factor",
    "disrupted_road", "seed", "horizon", "timestep", "edge", "edge_u",
    "edge_v", "occupancy", "capacity", "road_congestion",
    "network_congestion_score",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result = run_routing_comparison()
    _write_csv(SUMMARY_OUTPUT, SUMMARY_COLUMNS, result["summary"])
    _write_csv(CONGESTION_OUTPUT, CONGESTION_COLUMNS, result["congestion"])
    print(f"Wrote {SUMMARY_OUTPUT} ({len(result['summary'])} rows)")
    print(f"Wrote {CONGESTION_OUTPUT} ({len(result['congestion'])} rows)")


if __name__ == "__main__":
    main()
