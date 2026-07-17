import numpy as np

from src.deployment import calculate_psi


def test_psi_is_zero_for_identical_distributions() -> None:
    values = np.linspace(
        0.0,
        1.0,
        1000,
    )

    psi = calculate_psi(
        values,
        values.copy(),
        bins=10,
    )

    assert psi < 1e-10


def test_psi_detects_shifted_distribution() -> None:
    rng = np.random.default_rng(42)

    reference = rng.normal(
        loc=0.0,
        scale=1.0,
        size=5000,
    )

    current = rng.normal(
        loc=2.0,
        scale=1.0,
        size=5000,
    )

    psi = calculate_psi(
        reference,
        current,
        bins=10,
    )

    assert psi > 0.25


def test_psi_is_non_negative() -> None:
    rng = np.random.default_rng(42)

    reference = rng.normal(
        size=1000,
    )

    current = rng.normal(
        loc=0.2,
        size=1000,
    )

    psi = calculate_psi(
        reference,
        current,
        bins=10,
    )

    assert psi >= 0.0