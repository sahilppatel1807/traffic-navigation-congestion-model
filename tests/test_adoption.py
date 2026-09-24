"""Tests for seeded navigation-app adoption assignment."""

import pytest

from src.adoption import ADOPTION_RATES, assign_navigation_adoption
from src.vehicle import Vehicle


def _fleet(n: int, *, uses_navigation_app: bool = False) -> list[Vehicle]:
    return [
        Vehicle(
            vehicle_id=i,
            origin="A",
            destination="B",
            uses_navigation_app=uses_navigation_app,
        )
        for i in range(n)
    ]


def _nav_flags(vehicles: list[Vehicle]) -> list[bool]:
    return [v.uses_navigation_app for v in vehicles]


def _nav_count(vehicles: list[Vehicle]) -> int:
    return sum(1 for v in vehicles if v.uses_navigation_app)


# ---------------------------------------------------------------------------
# Primary seam — assign_navigation_adoption
# ---------------------------------------------------------------------------


def test_exact_count_for_known_n_and_rate():
    vehicles = _fleet(8)
    assign_navigation_adoption(vehicles, 0.25, seed=1)
    assert _nav_count(vehicles) == 2  # round(8 * 0.25) == 2

    vehicles = _fleet(10)
    assign_navigation_adoption(vehicles, 0.3, seed=1)
    assert _nav_count(vehicles) == 3  # round(10 * 0.3) == 3


def test_bankers_rounding_half_rate_on_odd_n():
    # round(5 * 0.5) == round(2.5) == 2 under Python banker's rounding
    vehicles = _fleet(5)
    assign_navigation_adoption(vehicles, 0.5, seed=7)
    assert _nav_count(vehicles) == 2
    assert sum(1 for v in vehicles if not v.uses_navigation_app) == 3


@pytest.mark.parametrize(
    "rate, expected",
    [
        (0.0, 0),
        (0, 0),
        (1.0, 8),
        (1, 8),
    ],
)
def test_boundary_rates_set_none_or_all(rate, expected):
    vehicles = _fleet(8)
    assign_navigation_adoption(vehicles, rate, seed=42)
    assert _nav_count(vehicles) == expected
    if expected == 0:
        assert all(not v.uses_navigation_app for v in vehicles)
    else:
        assert all(v.uses_navigation_app for v in vehicles)


def test_same_seed_reproducible_flag_pattern():
    a = _fleet(6)
    b = _fleet(6)
    assign_navigation_adoption(a, 0.5, seed=99)
    assign_navigation_adoption(b, 0.5, seed=99)
    assert _nav_flags(a) == _nav_flags(b)
    assert _nav_count(a) == round(6 * 0.5)


def test_different_seed_may_differ():
    a = _fleet(10)
    b = _fleet(10)
    assign_navigation_adoption(a, 0.5, seed=1)
    assign_navigation_adoption(b, 0.5, seed=2)
    assert _nav_count(a) == _nav_count(b) == 5
    assert _nav_flags(a) != _nav_flags(b)


def test_overwrites_preexisting_flags():
    vehicles = _fleet(4, uses_navigation_app=True)
    assert _nav_count(vehicles) == 4

    result = assign_navigation_adoption(vehicles, 0.0, seed=3)
    assert result is vehicles
    assert _nav_count(vehicles) == 0

    assign_navigation_adoption(vehicles, 1.0, seed=3)
    assert _nav_count(vehicles) == 4


def test_mutates_in_place_and_returns_same_list():
    vehicles = _fleet(3)
    result = assign_navigation_adoption(vehicles, 1 / 3, seed=5)
    assert result is vehicles
    assert _nav_count(vehicles) == 1  # round(3 * 1/3) == 1


def test_empty_list_is_noop():
    vehicles: list[Vehicle] = []
    result = assign_navigation_adoption(vehicles, 0.5, seed=0)
    assert result is vehicles
    assert result == []


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, 2, -1])
def test_rate_out_of_range_raises_value_error(bad_rate):
    with pytest.raises(ValueError, match="rate must be in"):
        assign_navigation_adoption(_fleet(2), bad_rate, seed=1)


@pytest.mark.parametrize("bad_rate", [True, False])
def test_bool_rate_raises_type_error(bad_rate):
    with pytest.raises(TypeError, match="rate must be"):
        assign_navigation_adoption(_fleet(2), bad_rate, seed=1)


@pytest.mark.parametrize("bad_seed", [True, False, 1.5, "1", 1.0])
def test_invalid_seed_raises_type_error(bad_seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        assign_navigation_adoption(_fleet(2), 0.5, seed=bad_seed)  # type: ignore[arg-type]


def test_missing_seed_raises_type_error():
    with pytest.raises(TypeError):
        assign_navigation_adoption(_fleet(2), 0.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Secondary seam — ADOPTION_RATES constant
# ---------------------------------------------------------------------------


def test_adoption_rates_constant():
    assert ADOPTION_RATES == (0.0, 0.25, 0.5, 0.75, 1.0)
