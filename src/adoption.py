"""Seeded navigation-app adoption assignment for mixed fleets.

Given an existing vehicle list and an adoption fraction in ``[0.0, 1.0]``,
overwrites each vehicle's ``uses_navigation_app`` flag so exactly
``round(n * rate)`` vehicles use the app (Python banker's rounding). Demand
builders stay separate: this module only assigns who uses the app.
"""

from __future__ import annotations

import random

from src.vehicle import Vehicle

# Planned experiment fractions for later scenario grids.
ADOPTION_RATES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def assign_navigation_adoption(
    vehicles: list[Vehicle],
    rate: float,
    *,
    seed: int,
) -> list[Vehicle]:
    """Overwrite navigation-app flags for an exact adoption mix.

    Mutates ``vehicles`` in place and returns the same list object. Selection
    uses a seeded shuffle of indices so exactly ``k = round(n * rate)``
    distinct vehicles are flagged (Python built-in banker's rounding — e.g.
    ``n=5``, ``rate=0.5`` → ``k=2``).

    Parameters
    ----------
    vehicles:
        Existing fleet (any list of :class:`~src.vehicle.Vehicle`). Empty lists
        are allowed and return unchanged.
    rate:
        Adoption fraction in ``[0.0, 1.0]``. Accepts ``int`` or ``float``
        (including ``0`` / ``1``); rejects ``bool``.
    seed:
        Required keyword-only integer seed for reproducible assignment.
        Rejects ``bool`` and non-integers.

    Returns
    -------
    list[Vehicle]
        The same ``vehicles`` list after flags are overwritten.

    Raises
    ------
    TypeError
        If ``rate`` is a ``bool``, or ``seed`` is not a real ``int``.
    ValueError
        If ``rate`` is outside ``[0.0, 1.0]``.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError(
            f"rate must be an int or float in [0.0, 1.0], got {type(rate).__name__}"
        )
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"rate must be in [0.0, 1.0], got {rate}")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}"
        )

    n = len(vehicles)
    if n == 0:
        return vehicles

    k = round(n * rate)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    nav_set = set(indices[:k])
    for i, vehicle in enumerate(vehicles):
        vehicle.uses_navigation_app = i in nav_set

    return vehicles
